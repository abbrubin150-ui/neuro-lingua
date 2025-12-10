"""Dataset preparation helpers for Neuro-Lingua.

This script prepares WikiText-style English encyclopaedic text and Hebrew news
articles for language modelling experiments. It supports lightweight sample
corpora that ship with the repository as well as reproducible downloads from
remote sources or the Hugging Face `datasets` hub (when available).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.request import urlopen

try:  # Optional import that is only required for --hf-name runs
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Local sample files bundled with the repository
SAMPLE_FILES = {
    "wikitext": RAW_DIR / "wikitext" / "sample.txt",
    "hebrew_news": RAW_DIR / "hebrew_news" / "sample.jsonl",
    "hebrew_opinion": RAW_DIR / "hebrew_opinion" / "sample.jsonl",
}


@dataclass
class SplitResult:
    name: str
    records: Sequence[dict]
    text_lines: Sequence[str]


@dataclass
class CorpusStats:
    documents: int
    tokens: int


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file at {path}; run with --mode sample first or provide --source-url.")
    return path


def _download_to(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:  # nosec B310 - trusted URLs documented in README
        payload = response.read()
    path.write_bytes(payload)
    return path


def _split_indices(total: int, ratios: Sequence[float]) -> List[range]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    counts = [int(total * r) for r in ratios]
    remainder = total - sum(counts)
    for i in range(remainder):
        counts[i % len(counts)] += 1
    offsets: List[range] = []
    cursor = 0
    for count in counts:
        offsets.append(range(cursor, cursor + count))
        cursor += count
    return offsets


def _tokenise(text: str) -> List[str]:
    return [tok for tok in text.replace("\n", " ").split(" ") if tok]


def _stats_from_lines(lines: Iterable[str]) -> CorpusStats:
    docs = 0
    tokens = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        docs += 1
        tokens += len(_tokenise(stripped))
    return CorpusStats(docs, tokens)


def _write_split_files(base_dir: Path, split: SplitResult) -> CorpusStats:
    base_dir.mkdir(parents=True, exist_ok=True)
    text_path = base_dir / f"{split.name}.txt"
    jsonl_path = base_dir / f"{split.name}.jsonl"
    text_path.write_text("\n\n".join(split.text_lines), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in split.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _stats_from_lines(split.text_lines)


def _shuffle(records: Sequence[dict], seed: int) -> List[dict]:
    copy = list(records)
    random.Random(seed).shuffle(copy)
    return copy


def build_wikitext(mode: str, output_dir: Path, seed: int, source_url: str | None) -> dict:
    if mode == "sample":
        raw_path = _ensure_exists(SAMPLE_FILES["wikitext"])
        paragraphs = [chunk.strip() for chunk in raw_path.read_text(encoding="utf-8").split("\n\n") if chunk.strip()]
        records = [{"id": idx, "paragraph": para} for idx, para in enumerate(paragraphs, start=1)]
    elif mode == "full":
        if source_url:
            raw_path = _download_to(RAW_DIR / "wikitext" / "full.txt", source_url)
            paragraphs = [chunk.strip() for chunk in raw_path.read_text(encoding="utf-8").split("\n\n") if chunk.strip()]
            records = [{"id": idx, "paragraph": para} for idx, para in enumerate(paragraphs, start=1)]
        elif load_dataset:
            dataset = load_dataset("wikitext", "wikitext-103-v1")  # type: ignore[arg-type]
            paragraphs = [item["text"].strip() for item in dataset["train"] if item["text"].strip()]
            records = [{"id": idx, "paragraph": para} for idx, para in enumerate(paragraphs, start=1)]
        else:
            raise RuntimeError("Full WikiText preparation requires either --source-url or the `datasets` package")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    shuffled = _shuffle(records, seed)
    ratios = [0.8, 0.1, 0.1]
    splits: List[SplitResult] = []
    indices = _split_indices(len(shuffled), ratios)
    for split_range, name in zip(indices, ["train", "validation", "test"]):
        slice_records = [shuffled[i] for i in split_range]
        splits.append(
            SplitResult(
                name=name,
                records=slice_records,
                text_lines=[entry["paragraph"] for entry in slice_records],
            )
        )

    metadata: dict[str, dict[str, int]] = {}
    for split in splits:
        stats = _write_split_files(output_dir, split)
        metadata[split.name] = {"documents": stats.documents, "tokens": stats.tokens}

    meta_path = output_dir / "metadata.json"
    meta_payload = {
        "dataset": "wikitext",
        "mode": mode,
        "seed": seed,
        "source": str(source_url or SAMPLE_FILES["wikitext"]),
        "splits": metadata,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_payload


def build_hebrew_news(mode: str, output_dir: Path, seed: int, source_url: str | None) -> dict:
    import json as _json

    if mode == "sample":
        raw_path = _ensure_exists(SAMPLE_FILES["hebrew_news"])
        records = [_json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif mode == "full":
        if source_url:
            raw_path = _download_to(RAW_DIR / "hebrew_news" / "full.jsonl", source_url)
            records = [_json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif load_dataset:
            dataset = load_dataset("json", data_files={"train": source_url}) if source_url else None  # type: ignore[arg-type]
            if dataset is None:
                raise RuntimeError("Full Hebrew news preparation requires --source-url when using the Hugging Face JSON loader.")
            records = [dict(item) for item in dataset["train"]]
        else:
            raise RuntimeError("Full Hebrew news preparation requires either --source-url or the `datasets` package")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for idx, record in enumerate(records, start=1):
        record.setdefault("id", idx)

    shuffled = _shuffle(records, seed)
    ratios = [0.8, 0.1, 0.1]
    indices = _split_indices(len(shuffled), ratios)
    splits: List[SplitResult] = []
    for split_range, name in zip(indices, ["train", "validation", "test"]):
        slice_records = [shuffled[i] for i in split_range]
        text_lines = [
            f"{entry.get('title', '').strip()}\n{entry.get('summary', '').strip()}\n\n{entry.get('content', '').strip()}".strip()
            for entry in slice_records
            if entry.get("content")
        ]
        splits.append(
            SplitResult(
                name=name,
                records=slice_records,
                text_lines=text_lines,
            )
        )

    metadata: dict[str, dict[str, int]] = {}
    for split in splits:
        stats = _write_split_files(output_dir, split)
        metadata[split.name] = {"documents": stats.documents, "tokens": stats.tokens}

    meta_path = output_dir / "metadata.json"
    meta_payload = {
        "dataset": "hebrew_news",
        "mode": mode,
        "seed": seed,
        "source": str(source_url or SAMPLE_FILES["hebrew_news"]),
        "splits": metadata,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_payload


def build_hebrew_opinion(mode: str, output_dir: Path, seed: int, source_url: str | None) -> dict:
    import json as _json

    if mode == "sample":
        raw_path = _ensure_exists(SAMPLE_FILES["hebrew_opinion"])
        records = [_json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif mode == "full":
        if source_url:
            raw_path = _download_to(RAW_DIR / "hebrew_opinion" / "full.jsonl", source_url)
            records = [_json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif load_dataset:
            dataset = load_dataset("json", data_files={"train": source_url}) if source_url else None  # type: ignore[arg-type]
            if dataset is None:
                raise RuntimeError("Full Hebrew opinion preparation requires --source-url when using the Hugging Face JSON loader.")
            records = [dict(item) for item in dataset["train"]]
        else:
            raise RuntimeError("Full Hebrew opinion preparation requires either --source-url or the `datasets` package")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for idx, record in enumerate(records, start=1):
        record.setdefault("id", idx)

    shuffled = _shuffle(records, seed)
    ratios = [0.8, 0.1, 0.1]
    indices = _split_indices(len(shuffled), ratios)
    splits: List[SplitResult] = []
    for split_range, name in zip(indices, ["train", "validation", "test"]):
        slice_records = [shuffled[i] for i in split_range]
        text_lines = [
            "\n".join(
                filter(
                    None,
                    [
                        entry.get("title", "").strip(),
                        entry.get("thesis", "").strip(),
                        entry.get("argument", "").strip(),
                        entry.get("counterpoint", "").strip(),
                        entry.get("call_to_action", "").strip(),
                    ],
                )
            )
            for entry in slice_records
            if entry.get("argument")
        ]
        splits.append(
            SplitResult(
                name=name,
                records=slice_records,
                text_lines=text_lines,
            )
        )

    metadata: dict[str, dict[str, int]] = {}
    for split in splits:
        stats = _write_split_files(output_dir, split)
        metadata[split.name] = {"documents": stats.documents, "tokens": stats.tokens}

    meta_path = output_dir / "metadata.json"
    meta_payload = {
        "dataset": "hebrew_opinion",
        "mode": mode,
        "seed": seed,
        "source": str(source_url or SAMPLE_FILES["hebrew_opinion"]),
        "splits": metadata,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_payload


BUILDERS = {
    "wikitext": build_wikitext,
    "hebrew_news": build_hebrew_news,
    "hebrew_opinion": build_hebrew_opinion,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare textual corpora for Neuro-Lingua experiments")
    parser.add_argument("--dataset", choices=sorted(BUILDERS.keys()), required=True)
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (defaults to data/processed/<dataset>)",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=None,
        help="Optional remote resource for --mode full runs (e.g. a WikiText mirror or Hebrew news JSONL).",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (PROCESSED_DIR / args.dataset)
    builder = BUILDERS[args.dataset]

    metadata = builder(args.mode, output_dir, seed=args.seed, source_url=args.source_url)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
