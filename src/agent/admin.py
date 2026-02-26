from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


ALLOWED_TRUST_LEVELS = {"Guest", "Friend", "OWNER"}


@dataclass
class AdminActionResult:
    ok: bool
    message: str


def _load_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _save_json_object(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalize_trust_level(raw: str) -> str | None:
    val = raw.strip().lower()
    mapping = {
        "guest": "Guest",
        "friend": "Friend",
        "owner": "OWNER",
    }
    return mapping.get(val)


def promote_person(
    name: str,
    target_level: str,
    trust_map_path: str,
    face_db_path: str,
) -> AdminActionResult:
    clean_name = name.strip()
    if not clean_name:
        return AdminActionResult(False, "name cannot be empty")

    normalized = normalize_trust_level(target_level)
    if normalized is None or normalized not in ALLOWED_TRUST_LEVELS:
        return AdminActionResult(False, f"invalid trust level '{target_level}'")

    trust_map_file = Path(trust_map_path)
    face_db_file = Path(face_db_path)

    try:
        face_db = _load_json_object(face_db_file)
    except Exception as e:
        return AdminActionResult(False, f"cannot read face DB: {e}")

    if clean_name not in face_db:
        return AdminActionResult(False, f"{clean_name} is not registered in face DB")

    try:
        trust_map = _load_json_object(trust_map_file)
        trust_map[clean_name] = normalized
        _save_json_object(trust_map_file, trust_map)
    except Exception as e:
        return AdminActionResult(False, f"cannot update trust map: {e}")

    return AdminActionResult(True, f"{clean_name} promoted to {normalized}")
