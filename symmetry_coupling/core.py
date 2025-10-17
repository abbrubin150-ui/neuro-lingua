"""Core implementation of the Symmetry Coupling metric.

This module provides an offline, dependency-free implementation of the
"Symmetry Coupling Metric v1.0" specification.  It exposes a small, stable API
that can be embedded inside data pipelines or used interactively from a REPL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import re
import string

__all__ = [
    "Config",
    "Metrics",
    "compute_coupling",
    "compute_coupling_batch",
    "load_synonym_dictionary",
]


_DATA_DIR = Path(__file__).resolve().parent
_HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")
_TOKEN_RE = re.compile(r"[\w\-']+", re.UNICODE)


DEFAULT_DIVERGENCE_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "general": {
        "strongly_coupled": 0.25,
        "coupled": 0.5,
        "weakly_coupled": 0.75,
    },
    "legal": {
        "strongly_coupled": 0.15,
        "coupled": 0.35,
        "weakly_coupled": 0.6,
    },
    "medical": {
        "strongly_coupled": 0.2,
        "coupled": 0.4,
        "weakly_coupled": 0.65,
    },
    "technical": {
        "strongly_coupled": 0.18,
        "coupled": 0.38,
        "weakly_coupled": 0.6,
    },
}


@dataclass
class Config:
    """Runtime configuration for the coupling metric.

    Attributes
    ----------
    language:
        Two letter language identifier used to load default synonym dictionaries.
    domain:
        Domain profile that controls the divergence thresholds used for
        classification.
    normalize:
        Whether to lowercase and strip punctuation before tokenisation.
    strip_diacritics:
        When ``True`` removes Hebrew niqqud/te'amim characters before analysis.
    synonyms_path:
        Optional path to a JSON synonyms file.  If not supplied, the loader will
        fall back to the package defaults for the chosen language.
    synonyms:
        In-memory synonyms dictionary that overrides ``synonyms_path``.
    thresholds:
        Optional mapping that overrides the default divergence thresholds for the
        selected domain.
    max_edit_distance:
        Maximum edit distance (Levenshtein) tolerated for fuzzy token matches.
    """

    language: str = "en"
    domain: str = "general"
    normalize: bool = True
    strip_diacritics: bool = True
    synonyms_path: Optional[str] = None
    synonyms: Optional[Mapping[str, Sequence[str]]] = None
    thresholds: Optional[Mapping[str, float]] = None
    max_edit_distance: int = 1


@dataclass
class Metrics:
    """Container for the computed metric values."""

    alignment: float
    reorder_penalty: float
    coupling: float
    divergence: float
    classification: str
    lcs_length: int
    token_count_a: int
    token_count_b: int

    def as_dict(self) -> Dict[str, float]:
        """Return a serialisable representation of the metric values."""

        return {
            "alignment": self.alignment,
            "reorder_penalty": self.reorder_penalty,
            "coupling": self.coupling,
            "divergence": self.divergence,
        }


def compute_coupling(text_a: str, text_b: str, config: Optional[Config] = None) -> Metrics:
    """Compute the symmetry coupling metrics for a pair of texts.

    Parameters
    ----------
    text_a, text_b:
        Input strings that will be tokenised and compared.
    config:
        Optional :class:`Config` object.  When omitted, the default configuration
        for English/general usage is applied.
    """

    config = config or Config()
    tokens_a = tokenize(text_a, config)
    tokens_b = tokenize(text_b, config)

    canonical_map = _resolve_canonical_map(config)
    return _compute_metrics(tokens_a, tokens_b, config, canonical_map)


def compute_coupling_batch(
    pairs: Iterable[Tuple[str, str]], config: Optional[Config] = None
) -> List[Metrics]:
    """Vectorised version of :func:`compute_coupling`.

    Parameters
    ----------
    pairs:
        Iterable of ``(text_a, text_b)`` tuples.
    config:
        Shared :class:`Config` object used for all comparisons.
    """
    config = config or Config()
    canonical_map = _resolve_canonical_map(config)
    results: List[Metrics] = []
    for text_a, text_b in pairs:
        tokens_a = tokenize(text_a, config)
        tokens_b = tokenize(text_b, config)
        results.append(_compute_metrics(tokens_a, tokens_b, config, canonical_map))
    return results


def _compute_metrics(
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    config: Config,
    canonical_map: Mapping[str, str],
) -> Metrics:
    lcs_len = lcs_length_fuzzy(tokens_a, tokens_b, canonical_map, config.max_edit_distance)

    max_len = max(len(tokens_a), len(tokens_b), 1)
    alignment = lcs_len / max_len

    reorder_penalty = _compute_reorder_penalty(tokens_a, tokens_b)
    coupling = alignment * (1.0 - 0.5 * reorder_penalty)
    coupling = max(0.0, min(1.0, coupling))
    divergence = 1.0 - coupling

    classification = classify(divergence, config)

    return Metrics(
        alignment=alignment,
        reorder_penalty=reorder_penalty,
        coupling=coupling,
        divergence=divergence,
        classification=classification,
        lcs_length=lcs_len,
        token_count_a=len(tokens_a),
        token_count_b=len(tokens_b),
    )


def tokenize(text: str, config: Config) -> List[str]:
    """Tokenise the provided text according to the configuration."""

    if not text:
        return []

    processed = text
    if config.strip_diacritics and config.language.lower() in {"he", "hebrew"}:
        processed = _HEBREW_DIACRITICS_RE.sub("", processed)

    if config.normalize:
        processed = processed.lower()

    processed = _normalise_punctuation(processed)

    tokens = _TOKEN_RE.findall(processed)
    return tokens


def _normalise_punctuation(text: str) -> str:
    translator = str.maketrans({ch: " " for ch in string.punctuation})
    return text.translate(translator)


def _resolve_canonical_map(config: Config) -> Mapping[str, str]:
    if config.synonyms is not None:
        return canonicalise_synonyms(config.synonyms)

    if config.synonyms_path:
        return load_synonym_dictionary(config.synonyms_path)

    candidate = _DATA_DIR / f"synonyms_{config.language.lower()}.json"
    if candidate.exists():
        return load_synonym_dictionary(str(candidate))

    return {}


def load_synonym_dictionary(path: str) -> Mapping[str, str]:
    """Load a synonyms file and produce a canonical lookup map."""

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return canonicalise_synonyms(data)


def canonicalise_synonyms(data: Mapping[str, Sequence[str]]) -> Mapping[str, str]:
    """Canonicalise a raw synonyms mapping.

    The canonicalisation step ensures bi-directional lookup: each token becomes
    part of a connected component where every member maps to a deterministic
    representative string.  This avoids asymmetries in the original mapping.
    """

    adjacency: Dict[str, set[str]] = {}
    for key, values in data.items():
        adjacency.setdefault(key, set()).update(values)
        for value in values:
            adjacency.setdefault(value, set()).add(key)

    canonical: Dict[str, str] = {}
    visited: set[str] = set()

    for token in adjacency:
        if token in visited:
            continue
        stack = [token]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency.get(current, ()))
        representative = sorted(component)[0]
        for entry in component:
            canonical[entry] = representative

    # Include isolated tokens that did not appear on the left-hand side.
    for token, values in data.items():
        if token not in canonical:
            canonical[token] = token
        for value in values:
            canonical.setdefault(value, value)

    return canonical


def lcs_length_fuzzy(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    canonical_map: Mapping[str, str],
    max_edit_distance: int,
) -> int:
    """Compute an LCS length that tolerates synonyms and near matches."""

    if not seq_a or not seq_b:
        return 0

    len_a, len_b = len(seq_a), len(seq_b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a):
        for j in range(len_b):
            if _tokens_equivalent(seq_a[i], seq_b[j], canonical_map, max_edit_distance):
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[len_a][len_b]


def _tokens_equivalent(
    token_a: str,
    token_b: str,
    canonical_map: Mapping[str, str],
    max_edit_distance: int,
) -> bool:
    if token_a == token_b:
        return True

    if canonical_map:
        if canonical_map.get(token_a, token_a) == canonical_map.get(token_b, token_b):
            return True

    if max_edit_distance < 0:
        return False

    return _levenshtein_distance(token_a, token_b) <= max_edit_distance


def _levenshtein_distance(token_a: str, token_b: str) -> int:
    """Compute the Levenshtein distance between two strings."""

    if token_a == token_b:
        return 0

    if len(token_a) < len(token_b):
        token_a, token_b = token_b, token_a

    previous_row = list(range(len(token_b) + 1))
    for i, char_a in enumerate(token_a, start=1):
        current_row = [i]
        for j, char_b in enumerate(token_b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def _compute_reorder_penalty(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    len_a = len(tokens_a)
    len_b = len(tokens_b)
    if not len_a and not len_b:
        return 0.0
    return abs(len_a - len_b) / max(len_a, len_b, 1)


def classify(divergence: float, config: Config) -> str:
    thresholds = _resolve_thresholds(config)
    if divergence < thresholds["strongly_coupled"]:
        return "STRONGLY_COUPLED"
    if divergence < thresholds["coupled"]:
        return "COUPLED"
    if divergence < thresholds["weakly_coupled"]:
        return "WEAKLY_COUPLED"
    return "DECOUPLED"


def _resolve_thresholds(config: Config) -> Mapping[str, float]:
    if config.thresholds is not None:
        return config.thresholds

    domain = config.domain.lower()
    return DEFAULT_DIVERGENCE_THRESHOLDS.get(domain, DEFAULT_DIVERGENCE_THRESHOLDS["general"])
