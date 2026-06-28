"""
Auto-generate ``sem/*.txt`` gesture-type labels from transcriptions using an
LLM (GPT-4o-mini by default).

This script reads word-level transcriptions — either in the BEAT2
``_whisper_relations.json`` format or in standard Whisper JSON output — calls
the same LLM prompt used by the RAG-Gesture retrieval pipeline, and writes
BEAT2-compatible ``sem/*.txt`` files that can be consumed by
``BEATXDataset``.

Usage examples:

  # From BEAT2-style whisper_relations JSON directory
  OPENAI_API_KEY=sk-... python tools/generate_sem_labels.py \
      --input_dir datasets/beat_english_v2.0.0/whisper_transcription \
      --output_dir datasets/beat_english_v2.0.0/sem \
      --input_format whisper_relations

  # From standard Whisper word-level JSON directory
  OPENAI_API_KEY=sk-... python tools/generate_sem_labels.py \
      --input_dir /path/to/teacher_whisper_output \
      --output_dir /path/to/teacher_data/sem \
      --input_format whisper_standard

  # Process a single file
  OPENAI_API_KEY=sk-... python tools/generate_sem_labels.py \
      --input_file /path/to/transcription.json \
      --output_dir /path/to/sem \
      --input_format whisper_standard
"""

import argparse
import glob
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# LLM interaction — mirrors the prompt & parsing from
# ``mogen/models/transformers/rag/llm_retrieval.py`` so that this script can
# run standalone (no mmcv / PyTorch / mogen dependency).
# ---------------------------------------------------------------------------
import warnings

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_GEST_TYPE_EXP_SHORT = """
You are an expert in human gestures. You need to identify words that may elicit semantically meaningful gestures(deictic, iconic, metaphoric) and their types:

Metaphoric Gesture: Represents abstract ideas or concepts physically, creating a vivid mental image.
Iconic Gesture: Mimics the shape or action of the object or concept being described.
Deictic Gesture: Points to or indicates a person, object, or location.

Format your response as a python list of python tuples of (word, type). For example: [('hello', 'beat'), ('world',
'iconic')]
"""

_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    warnings.warn(
        "OPENAI_API_KEY is not set. Set it to enable LLM-based label generation."
    )
    _client = None
else:
    _client = OpenAI(api_key=_openai_api_key)


def get_llm_output(text, model="gpt-4o-mini"):
    if _client is None:
        raise RuntimeError(
            "OpenAI client is not initialised — export OPENAI_API_KEY first."
        )
    completion = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GEST_TYPE_EXP_SHORT},
            {
                "role": "user",
                "content": (
                    "identify at most 2 important words which are more likely "
                    "to elicit semantically meaningful gestures and what are "
                    f'types of those gestures in following text: "{text}".'
                ),
            },
        ],
    )
    return completion.choices[0].message.content


def parse_gesture_labels_from_llm_output(llm_output):
    """Parse ``[(word, type), ...]`` from the LLM text response."""
    match_regex = (
        r"[\"\']*"
        r"([\w \-\']+\w)"
        r"[\"\']*\,\s*[\"\']*"
        r"(?P<gesttype>b*eat|m*etaphoric|iconic|deictic)"
    )
    gesture_labels = []
    for m in re.finditer(match_regex, llm_output, re.MULTILINE):
        gt = m.group("gesttype")
        if "etaphoric" in gt:
            gesttype = "metaphoric"
        elif "eat" in gt:
            gesttype = "beat"
        elif "iconic" in gt:
            gesttype = "iconic"
        elif "deictic" in gt:
            gesttype = "deictic"
        else:
            continue
        gesture_labels.append({"word": m.group(1).strip(), "name": gesttype})

    # Drop beat (non-semantic) and deduplicate
    gesture_labels = [gl for gl in gesture_labels if gl["name"] != "beat"]
    seen = []
    for gl in gesture_labels:
        if gl not in seen:
            seen.append(gl)
    return seen


# ---------------------------------------------------------------------------
# Transcription readers
# ---------------------------------------------------------------------------

def read_whisper_relations(filepath):
    """Read a BEAT2 ``*_whisper_relations.json`` file.

    Returns (full_text, word_records) where each word_record is
    ``{"word": str, "start": float, "end": float}``.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for sent in data["sentences"]:
        for tok in sent["tokens"]:
            word = tok["surface"].replace(" ", "")
            if not word:
                continue
            records.append({
                "word": word,
                "start": float(tok["startSec"]),
                "end": float(tok["endSec"]),
            })

    full_text = " ".join(r["word"] for r in records)
    return full_text, records


def read_whisper_standard(filepath):
    """Read a standard Whisper JSON output with word-level timestamps.

    Expects either:
      - top-level ``"words"`` list, or
      - ``"segments"`` each containing a ``"words"`` list.

    Each word entry has ``"word"`` (or ``"text"``), ``"start"``, ``"end"``.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_words = []
    if "words" in data:
        raw_words = data["words"]
    elif "segments" in data:
        for seg in data["segments"]:
            raw_words.extend(seg.get("words", []))
    else:
        raise ValueError(
            f"Cannot find 'words' or 'segments' in {filepath}. "
            "Make sure the Whisper output includes word-level timestamps "
            "(use --word_timestamps True when running Whisper)."
        )

    records = []
    for w in raw_words:
        word = w.get("word") or w.get("text", "")
        word = word.strip()
        if not word:
            continue
        records.append({
            "word": word,
            "start": float(w["start"]),
            "end": float(w["end"]),
        })

    full_text = " ".join(r["word"] for r in records)
    return full_text, records


READERS = {
    "whisper_relations": read_whisper_relations,
    "whisper_standard": read_whisper_standard,
}


# ---------------------------------------------------------------------------
# Match LLM-identified words back to word-level timestamps
# ---------------------------------------------------------------------------

def match_gesture_words_to_timestamps(gesture_labels, word_records):
    """For each gesture label from the LLM, find matching word timestamps.

    Args:
        gesture_labels: list of ``{"word": str, "name": str}`` from
            ``parse_gesture_labels_from_llm_output``.
        word_records: list of ``{"word": str, "start": float, "end": float}``
            from the transcription reader.

    Returns:
        list of dicts with ``name, start, end, duration, word``.
    """
    results = []

    # Normalise word_records for matching
    norm_records = []
    for r in word_records:
        norm = re.sub(r"[^a-z0-9 ]", "", r["word"].lower())
        norm_records.append(norm)

    for gl in gesture_labels:
        query = re.sub(r"[^a-z0-9 ]", "", gl["word"].lower())
        query_parts = query.split()
        if not query_parts:
            continue

        # Try to find a contiguous span matching all parts
        matched = False
        for i, nw in enumerate(norm_records):
            if nw == query_parts[0] or query_parts[0] in nw:
                # Single-word match
                if len(query_parts) == 1:
                    results.append({
                        "name": gl["name"],
                        "start": word_records[i]["start"],
                        "end": word_records[i]["end"],
                        "word": gl["word"],
                    })
                    matched = True
                    break

                # Multi-word: try to match subsequent words
                span_end = i
                parts_matched = 1
                for j in range(1, len(query_parts)):
                    idx = i + j
                    if idx >= len(norm_records):
                        break
                    if norm_records[idx] == query_parts[j] or query_parts[j] in norm_records[idx]:
                        parts_matched += 1
                        span_end = idx

                if parts_matched == len(query_parts):
                    results.append({
                        "name": gl["name"],
                        "start": word_records[i]["start"],
                        "end": word_records[span_end]["end"],
                        "word": gl["word"],
                    })
                    matched = True
                    break

        if not matched:
            # Fallback: fuzzy single-word match
            for i, nw in enumerate(norm_records):
                if any(qp in nw or nw in qp for qp in query_parts):
                    results.append({
                        "name": gl["name"],
                        "start": word_records[i]["start"],
                        "end": word_records[i]["end"],
                        "word": gl["word"],
                    })
                    break

    return results


# ---------------------------------------------------------------------------
# Write sem/*.txt
# ---------------------------------------------------------------------------

def write_sem_file(output_path, matched_gestures):
    """Write a BEAT2-compatible ``sem/{id}.txt`` file.

    Format (tab-separated, no header):
        name  start_time  end_time  duration  score  keywords
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for g in matched_gestures:
            duration = round(g["end"] - g["start"], 6)
            # score=1.0 as a default; it is only used when sem_rep="score"
            line = f"{g['name']}\t{g['start']:.6f}\t{g['end']:.6f}\t{duration:.6f}\t1.0\t{g['word']}\n"
            f.write(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def derive_sample_id(filepath, input_format):
    """Extract a BEAT2-style sample id from the file path."""
    basename = os.path.basename(filepath)
    if input_format == "whisper_relations":
        # e.g. ``1_wayne_0_1_1_whisper_relations.json`` → ``1_wayne_0_1_1``
        return basename.replace("_whisper_relations.json", "")
    # Generic: strip extension
    return os.path.splitext(basename)[0]


def process_single_file(filepath, output_dir, input_format, model, verbose=False):
    """Process one transcription file → one sem/*.txt file."""
    reader = READERS[input_format]
    full_text, word_records = reader(filepath)

    if not full_text.strip():
        if verbose:
            print(f"  [skip] empty text in {filepath}")
        return False

    # Call LLM
    llm_output = get_llm_output(full_text, model=model)
    gesture_labels = parse_gesture_labels_from_llm_output(llm_output)

    if verbose:
        print(f"  LLM identified {len(gesture_labels)} gesture(s): {gesture_labels}")

    # Match to timestamps
    matched = match_gesture_words_to_timestamps(gesture_labels, word_records)

    # Even if no gestures were found, write an empty file so the dataset
    # loader doesn't crash on a missing file.
    sample_id = derive_sample_id(filepath, input_format)
    out_path = os.path.join(output_dir, f"{sample_id}.txt")
    write_sem_file(out_path, matched)

    if verbose:
        print(f"  → wrote {out_path} ({len(matched)} gesture(s))")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate sem/*.txt gesture labels from transcriptions via LLM."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir",
        help="Directory of transcription files to process.",
    )
    group.add_argument(
        "--input_file",
        help="Single transcription file to process.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write sem/*.txt files into.",
    )
    parser.add_argument(
        "--input_format",
        choices=list(READERS.keys()),
        default="whisper_relations",
        help="Format of input transcription files (default: whisper_relations).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "llama3-8b"],
        help="LLM model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.5,
        help="Seconds to wait between LLM calls to avoid rate limits (default: 0.5).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file progress.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect files
    if args.input_file:
        files = [args.input_file]
    else:
        if args.input_format == "whisper_relations":
            pattern = os.path.join(args.input_dir, "*_whisper_relations.json")
        else:
            pattern = os.path.join(args.input_dir, "*.json")
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching format '{args.input_format}' in {args.input_dir}")
        sys.exit(1)

    print(f"Processing {len(files)} file(s) → {args.output_dir}")
    n_success = 0
    n_error = 0
    for i, fpath in enumerate(files):
        if args.verbose:
            print(f"[{i+1}/{len(files)}] {os.path.basename(fpath)}")
        try:
            ok = process_single_file(
                fpath, args.output_dir, args.input_format, args.model, args.verbose,
            )
            if ok:
                n_success += 1
        except Exception as e:
            n_error += 1
            print(f"  [ERROR] {fpath}: {e}")

        # Rate limit
        if i < len(files) - 1:
            time.sleep(args.rate_limit_delay)

    print(f"\nDone. {n_success} succeeded, {n_error} failed out of {len(files)} files.")


if __name__ == "__main__":
    main()
