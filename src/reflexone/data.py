"""Pinned NLU++ ingestion with exact spans and frozen, leakage-safe text groups."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from typing import Any

SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{SOURCE_REVISION}/nlupp/data/"
)
DOMAINS = ("banking", "hotels")
SPLITS = ("train", "validation", "calibration", "test")
SPLIT_PROPORTIONS = (0.65, 0.15, 0.10, 0.10)
PREPARATION_VERSION = 1

ATTRIBUTION = {
    "dataset": "NLU++",
    "authors": [
        "Inigo Casanueva", "Ivan Vulić", "Georgios Spithourakis", "Paweł Budzianowski"
    ],
    "paper": "https://aclanthology.org/2022.findings-naacl.154/",
    "repository": "https://github.com/PolyAI-LDN/task-specific-datasets",
    "source_revision": SOURCE_REVISION,
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "source_license": (
        "https://github.com/PolyAI-LDN/task-specific-datasets/"
        f"blob/{SOURCE_REVISION}/LICENSE"
    ),
    "modifications": (
        "Original text and character spans retained; conflicting duplicate annotations "
        "quarantined, identical within-domain examples collapsed, normalized calendar "
        "values omitted, custom grouped train/validation/calibration/test split."
    ),
}


@dataclass(frozen=True, slots=True)
class Slot:
    label: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    text: str
    domain: str
    intents: tuple[str, ...]
    slots: tuple[Slot, ...]
    group: str


@dataclass(frozen=True, slots=True)
class Ontology:
    intents: tuple[str, ...]
    slot_types: tuple[str, ...]
    _intent_domains: tuple[tuple[str, ...], ...] = field(repr=False)
    _slot_domains: tuple[tuple[str, ...], ...] = field(repr=False)

    def intent_mask(self, domain: str) -> list[bool]:
        return self._mask(domain, self._intent_domains)

    def slot_mask(self, domain: str) -> list[bool]:
        return self._mask(domain, self._slot_domains)

    @staticmethod
    def _mask(domain: str, domains: tuple[tuple[str, ...], ...]) -> list[bool]:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown support domain {domain!r}; choose banking or hotels")
        return ["general" in allowed or domain in allowed for allowed in domains]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def load_ontology(path: Path | None = None) -> Ontology:
    """Load fixed alphabetical label orders from packaged or prepared ontology JSON."""
    if path is None:
        payload = json.loads(files("reflexone").joinpath("resources/ontology.json").read_text())
    else:
        payload = _read_json(Path(path))
    try:
        intents = tuple(sorted(payload["intents"]))
        slots = tuple(sorted(payload["slots"]))
        if not intents or not slots:
            raise ValueError("ontology must contain intents and slots")
        domain_sets = []
        for kind, labels in (("intents", intents), ("slots", slots)):
            for label in labels:
                allowed = payload[kind][label]["domain"]
                if (
                    not isinstance(label, str)
                    or not isinstance(allowed, list)
                    or not allowed
                    or any(d not in (*DOMAINS, "general") for d in allowed)
                ):
                    raise ValueError(f"invalid domain declaration for {kind} {label!r}")
            domain_sets.append(tuple(tuple(payload[kind][label]["domain"]) for label in labels))
        return Ontology(intents, slots, *domain_sets)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed ontology: {exc}") from exc


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _group_id(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _validate_record(record: Record, ontology: Ontology) -> None:
    context = f"Record {record.id}"
    if not isinstance(record.text, str) or not record.text.strip():
        raise ValueError(f"{context}: text must be a nonempty string")
    if not isinstance(record.id, str) or not record.id:
        raise ValueError("Record id must be a nonempty string")
    valid_intents = {
        label for label, valid in zip(ontology.intents, ontology.intent_mask(record.domain)) if valid
    }
    valid_slots = {
        label for label, valid in zip(ontology.slot_types, ontology.slot_mask(record.domain)) if valid
    }
    for label in record.intents:
        if label not in valid_intents:
            raise ValueError(f"{context}: intent {label!r} is unknown or invalid for domain {record.domain}")
    for slot in record.slots:
        if slot.label not in valid_slots:
            raise ValueError(f"{context}: slot {slot.label!r} is unknown or invalid for domain {record.domain}")
        if (
            type(slot.start) is not int
            or type(slot.end) is not int
            or not 0 <= slot.start < slot.end <= len(record.text)
            or not isinstance(slot.text, str)
            or record.text[slot.start:slot.end] != slot.text
        ):
            raise ValueError(f"{context}: invalid or inexact span for {slot.label!r}: {slot.start, slot.end}")


def load_records(path: Path) -> list[Record]:
    """Read the JSONL dataclass representation, validating domain and copied spans."""
    path = Path(path)
    ontology_path = path.parent / "ontology.json"
    ontology = load_ontology(ontology_path if ontology_path.is_file() else None)
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read records from {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            record = Record(
                id=value["id"], text=value["text"], domain=value["domain"],
                intents=tuple(value["intents"]),
                slots=tuple(Slot(**s) for s in value["slots"]), group=value["group"],
            )
            _validate_record(record, ontology)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed record at {path}:{line_number}: {exc}") from exc
        records.append(record)
    return records


def _download_raw(raw_root: Path, relative_path: str) -> None:
    destination = raw_root / relative_path
    if destination.is_file():
        return
    url = SOURCE_BASE + relative_path
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ReflexOne/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            json.loads(content)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return
        except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Failed to download pinned source {url} after 3 attempts: {last_error}")


def _source_records(raw_root: Path, ontology: Ontology, source_paths: list[str]) -> tuple[
    list[Record], dict[str, Any]
]:
    records = []
    overlaps = []
    repeated_labels = 0
    repeated_records = []
    for name in source_paths:
        if name == "ontology.json":
            continue
        values = _read_json(raw_root / name)
        if not isinstance(values, list):
            raise ValueError(f"Malformed fold {name}: expected a JSON array")
        domain = name.split("/")[0]
        for row, value in enumerate(values):
            source_id = f"{name}:{row}"
            try:
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                text = value["text"]
                raw_intents = value.get("intents", [])
                raw_slots = value.get("slots", {})
                if not isinstance(raw_intents, list) or any(not isinstance(x, str) for x in raw_intents):
                    raise ValueError("intents must be a list of label strings")
                if not isinstance(raw_slots, dict):
                    raise ValueError("slots must be an object")
                slots = []
                for label, slot in raw_slots.items():
                    if not isinstance(slot, dict) or not isinstance(slot.get("span"), list) or len(slot["span"]) != 2:
                        raise ValueError(f"malformed span for {label!r}")
                    slots.append(Slot(label, slot["span"][0], slot["span"][1], slot["text"]))
                record = Record(
                    source_id, text, domain, tuple(sorted(set(raw_intents))),
                    tuple(sorted(slots, key=lambda s: (s.label, s.start, s.end))), _group_id(text),
                )
                _validate_record(record, ontology)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Malformed source record {source_id}: {exc}") from exc
            repeated = len(raw_intents) - len(record.intents)
            repeated_labels += repeated
            if repeated:
                repeated_records.append(source_id)
            for left, right in combinations(record.slots, 2):
                if max(left.start, right.start) < min(left.end, right.end):
                    overlaps.append({"id": record.id, "slots": [asdict(left), asdict(right)]})
            records.append(record)
    return records, {
        "input_records": len(records),
        "input_by_domain": dict(Counter(r.domain for r in records)),
        "input_slot_annotations": sum(len(r.slots) for r in records),
        "invalid_spans": 0,
        "overlap_pairs": len(overlaps),
        "overlap_records": len({x["id"] for x in overlaps}),
        "overlaps": overlaps,
        "empty_intent_records": sum(not r.intents for r in records),
        "empty_slot_records": sum(not r.slots for r in records),
        "repeated_intent_labels_removed": repeated_labels,
        "repeated_intent_records": repeated_records,
    }


def _canonicalize(records: list[Record]) -> tuple[list[Record], dict[str, list[str]], dict[str, Any]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    duplicate_groups = [group for group in grouped.values() if len(group) > 1]
    canonical = []
    provenance = {}
    quarantine = []
    intent_conflicts = slot_conflicts = 0
    for group_id in sorted(grouped):
        group = grouped[group_id]
        conflicting_intents = len({r.intents for r in group}) > 1
        conflicting_slots = len({r.slots for r in group}) > 1
        if conflicting_intents or conflicting_slots:
            intent_conflicts += conflicting_intents
            slot_conflicts += conflicting_slots
            quarantine.append({
                "group": group_id, "normalized_text": _normalize_text(group[0].text),
                "intent_conflict": conflicting_intents, "slot_conflict": conflicting_slots,
                "records": [asdict(r) for r in group],
            })
            continue
        # Source records were read by domain, then numeric fold index, then row.
        for domain in DOMAINS:
            domain_records = [r for r in group if r.domain == domain]
            if domain_records:
                selected = domain_records[0]
                canonical.append(selected)
                provenance[selected.id] = [r.id for r in domain_records]
    canonical.sort(key=lambda r: (r.domain, r.group, r.id))
    return canonical, provenance, {
        "duplicate_groups": len(duplicate_groups),
        "duplicate_records": sum(len(g) for g in duplicate_groups),
        "duplicate_excess": sum(len(g) - 1 for g in duplicate_groups),
        "cross_domain_duplicate_groups": sum(len({r.domain for r in g}) > 1 for g in duplicate_groups),
        "conflicting_groups": len(quarantine),
        "conflicting_intent_groups": intent_conflicts,
        "conflicting_slot_groups": slot_conflicts,
        "quarantined_records": sum(len(g["records"]) for g in quarantine),
        "quarantine": quarantine,
        "canonical_records": len(canonical),
        "canonical_by_domain": dict(Counter(r.domain for r in canonical)),
        "canonical_slot_annotations": sum(len(r.slots) for r in canonical),
        "global_groups": len({r.group for r in canonical}),
    }


def _assign_splits(records: list[Record], seed: int) -> dict[str, str]:
    domains_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        domains_by_group[record.group].add(record.domain)
    buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for group, domains in domains_by_group.items():
        buckets[tuple(sorted(domains))].append(group)
    rng = random.Random(seed)
    assignment = {}
    for bucket in sorted(buckets):
        groups = sorted(buckets[bucket])
        rng.shuffle(groups)
        ideals = [len(groups) * p for p in SPLIT_PROPORTIONS]
        sizes = [math.floor(x) for x in ideals]
        remainders = sorted(range(4), key=lambda i: (-(ideals[i] - sizes[i]), i))
        for index in remainders[:len(groups) - sum(sizes)]:
            sizes[index] += 1
        offset = 0
        for split, size in zip(SPLITS, sizes):
            for group in groups[offset:offset + size]:
                assignment[group] = split
            offset += size
    return assignment


def _supports(records: list[Record], ontology: Ontology) -> dict[str, Any]:
    intents = Counter(label for r in records for label in r.intents)
    slots = Counter(slot.label for r in records for slot in r.slots)
    return {
        "records": len(records), "groups": len({r.group for r in records}),
        "domains": {d: sum(r.domain == d for r in records) for d in DOMAINS},
        "empty_intents": sum(not r.intents for r in records),
        "empty_slots": sum(not r.slots for r in records),
        "intent_positives": {label: intents[label] for label in ontology.intents},
        "slot_spans": {label: slots[label] for label in ontology.slot_types},
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def prepare_dataset(path: Path, seed: int = 17) -> dict[str, Any]:
    """Prepare pinned NLU++ with one frozen grouped split and explicit source provenance.

    Existing ``path/raw`` source files are accepted as local inputs and hashed. A
    completed manifest freezes their bytes and the split seed for later calls.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "manifest.json"
    previous = _read_json(manifest_path) if manifest_path.is_file() else None
    if previous and previous.get("split_seed") != seed:
        raise ValueError("Dataset split is frozen; use its recorded seed or prepare another directory")
    source_paths = ["ontology.json"] + [f"{d}/fold{fold}.json" for d in DOMAINS for fold in range(20)]
    raw_root = path / "raw"
    raw_root.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda name: _download_raw(raw_root, name), source_paths))
    source_files = []
    for name in source_paths:
        content = (raw_root / name).read_bytes()
        source_files.append({
            "path": f"raw/{name}", "url": SOURCE_BASE + name,
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        })
    if previous and previous.get("source_files") != source_files:
        raise ValueError("Dataset source bytes are frozen by the existing manifest; use another directory")
    ontology = load_ontology(raw_root / "ontology.json")
    raw_records, raw_audit = _source_records(raw_root, ontology, source_paths)
    records, provenance, dedup_audit = _canonicalize(raw_records)
    assignment = _assign_splits(records, seed)
    partitions = {name: [r for r in records if assignment[r.group] == name] for name in SPLITS}
    ontology_json = _read_json(raw_root / "ontology.json")
    ontology_json["_attribution"] = ATTRIBUTION
    prepared_files = {"ontology.json": _json_bytes(ontology_json)}
    for name in SPLITS:
        prepared_files[f"{name}.jsonl"] = b"".join(
            (json.dumps(asdict(r), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            for r in partitions[name]
        )
    records_manifest = [{
        "id": r.id, "group": r.group, "domain": r.domain, "split": assignment[r.group],
        "source_ids": provenance[r.id],
    } for r in records]
    fingerprint = hashlib.sha256(_json_bytes({
        "preparation_version": PREPARATION_VERSION, "source_revision": SOURCE_REVISION,
        "source_files": source_files, "split_seed": seed,
        "output_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in prepared_files.items()},
        "records": records_manifest,
    })).hexdigest()
    audit = {
        "dataset": "NLU++", "source_revision": SOURCE_REVISION,
        "preparation_version": PREPARATION_VERSION, "split_seed": seed,
        "fingerprint": fingerprint,
        **raw_audit, **dedup_audit,
        "supports": {name: _supports(partitions[name], ontology) for name in SPLITS},
        "label_order": {"intents": list(ontology.intents), "slots": list(ontology.slot_types)},
        "protocol": (
            "Custom 65/15/10/10 grouped holdout, approximately stratified by known domain; "
            "not the NLU++ paper's cross-validation protocol."
        ),
    }
    prepared_files["audit.json"] = _json_bytes(audit)
    manifest = {
        "dataset": "NLU++", "attribution": ATTRIBUTION,
        "preparation_version": PREPARATION_VERSION,
        "source_revision": SOURCE_REVISION, "source_files": source_files,
        "split_seed": seed, "split_proportions": dict(zip(SPLITS, SPLIT_PROPORTIONS)),
        "group_normalization": "NFKC, casefold, collapse whitespace; original text unchanged",
        "fingerprint": fingerprint, "records": records_manifest,
        "output_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in prepared_files.items()},
    }
    for name, content in prepared_files.items():
        _atomic_write(path / name, content)
    _atomic_write(manifest_path, _json_bytes(manifest))
    return audit
