"""Install the pinned public model release without replacing local models."""

from __future__ import annotations

import hashlib
import http.client
import json
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

MODEL_VERSION = "0.1.0"
MODEL_URL = (
    "https://github.com/PDGamerSG/reflexone/releases/download/v0.1.0/reflexone-models-v0.1.0.tar.gz"
)
MODEL_SHA256 = "c656098809b926b6ca789294d4023e4ee087887e5ca9ea8880ee8b97f1246381"
MODEL_BYTES = 121_863_179
_PROFILES = ("support", "commands", "spam")
_MANIFEST = "reports/artifact-manifest.json"
_NOTICES = {"LICENSE", "THIRD_PARTY.md", "licenses/Apache-2.0.txt"}
_CHUNK_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_EXPANDED_BYTES = 256 * 1024 * 1024


def _download_archive(url: str, destination: Path, expected_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": f"reflexone/{MODEL_VERSION}"})
    for attempt in range(3):
        try:
            deadline = time.monotonic() + 600
            total = 0
            with urllib.request.urlopen(request, timeout=30) as response:
                with destination.open("wb") as output:
                    while True:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Model download exceeded ten minutes")
                        chunk = response.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > expected_bytes:
                            raise OSError("Model download exceeds the pinned release size")
                        output.write(chunk)
            if total != expected_bytes:
                raise OSError(f"Model download size is {total} bytes; expected {expected_bytes}")
            return
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            destination.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"Model download failed after 3 attempts: {exc}") from exc
            time.sleep(attempt + 1)


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _safe_member_name(name: str, *, directory: bool = False) -> str:
    if not isinstance(name, str) or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    if directory:
        name = name.removesuffix("/")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    if parts[0] == "artifacts":
        if len(parts) > 1 and parts[1] not in _PROFILES:
            raise ValueError(f"Unknown model profile in archive: {name!r}")
        if directory or len(parts) >= 3:
            return name
    elif directory and name in {"licenses", "reports"}:
        return name
    elif not directory and name in _NOTICES | {_MANIFEST}:
        return name
    raise ValueError(f"Unexpected archive path: {name!r}")


def _manifest_files(raw: bytes) -> dict:
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid artifact manifest JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != MODEL_VERSION
        or manifest.get("profiles") != list(_PROFILES)
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("Artifact manifest does not describe the expected release profiles")
    files = {}
    for name, identity in manifest["files"].items():
        _safe_member_name(name)
        if (
            name == _MANIFEST
            or not isinstance(identity, dict)
            or type(identity.get("bytes")) is not int
            or identity["bytes"] < 0
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in identity["sha256"])
        ):
            raise ValueError(f"Invalid artifact manifest entry: {name!r}")
        files[name] = {"bytes": identity["bytes"], "sha256": identity["sha256"]}
    for profile in _PROFILES:
        if not any(name.startswith(f"artifacts/{profile}/") for name in files):
            raise ValueError(f"Artifact manifest is missing profile '{profile}'")
    return files


def _extract_verified(archive_path: Path, stage: Path) -> dict:
    seen = set()
    found = {}
    manifest = None
    expanded_bytes = 0
    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if len(seen) >= 512:
                    raise ValueError("Unsafe archive contains too many entries")
                if not member.isdir() and member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    raise ValueError(f"Unsafe archive entry type: {member.name!r}")
                name = _safe_member_name(member.name, directory=member.isdir())
                if name in seen:
                    raise ValueError(f"Duplicate archive path: {name!r}")
                seen.add(name)
                if member.isdir():
                    if member.size:
                        raise ValueError(f"Unsafe archive directory has a payload: {name!r}")
                    continue
                expanded_bytes += member.size
                if member.size < 0 or expanded_bytes > _MAX_EXPANDED_BYTES:
                    raise ValueError("Unsafe archive exceeds the model release size limit")
                with archive.extractfile(member) as source:
                    if name == _MANIFEST:
                        if member.size > _MAX_MANIFEST_BYTES:
                            raise ValueError("Artifact manifest exceeds the size limit")
                        manifest = _manifest_files(source.read(_MAX_MANIFEST_BYTES + 1))
                        continue
                    target = None
                    if name.startswith("artifacts/"):
                        target = stage.joinpath(*PurePosixPath(name).parts[1:])
                        target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    output = target.open("xb") if target else None
                    try:
                        while chunk := source.read(_CHUNK_BYTES):
                            size += len(chunk)
                            digest.update(chunk)
                            if output:
                                output.write(chunk)
                    finally:
                        if output:
                            output.close()
                    found[name] = {"bytes": size, "sha256": digest.hexdigest()}
    except (tarfile.TarError, EOFError) as exc:
        raise ValueError(f"Invalid model archive: {exc}") from exc
    if manifest is None or set(manifest) != set(found):
        raise ValueError("Artifact manifest file list does not match the archive")
    for name, expected in manifest.items():
        if found[name] != expected:
            raise ValueError(f"Artifact manifest checksum or size differs for {name!r}")
    return manifest


def _existing_profile_matches(path: Path, expected: dict) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    conflict = (
        f"Existing profile '{path.name}' conflicts with model release {MODEL_VERSION}; "
        "use an empty artifacts root or move the existing profile aside"
    )
    if path.is_symlink() or not path.is_dir():
        raise ValueError(conflict)
    actual = {}
    for item in path.rglob("*"):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError(conflict)
        if item.is_file():
            actual[item.relative_to(path).as_posix()] = _file_identity(item)
    if actual != expected:
        raise ValueError(conflict)
    return True


def _install_archive(
    archive_path: Path,
    artifacts_root: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict:
    """Verify a trusted archive and publish complete profiles; private local-fixture seam."""
    identity = _file_identity(archive_path)
    if identity["bytes"] != expected_bytes:
        raise ValueError(f"Model archive size differs from the pinned {expected_bytes} bytes")
    if identity["sha256"] != expected_sha256:
        raise ValueError("Model archive SHA256 checksum differs from the pinned release")
    root = Path(artifacts_root).absolute()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"Artifacts root must be a directory, not a symlink or file: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reflexone-models-", dir=root.parent) as temporary:
        stage = Path(temporary)
        manifest = _extract_verified(archive_path, stage)
        root.mkdir(exist_ok=True)
        lock = root / ".reflexone-install.lock"
        try:
            lock.touch(exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(
                f"Another model installer holds {lock}; remove a stale lock only after "
                "confirming no installer is running"
            ) from exc
        installed = []
        unchanged = []
        try:
            for profile in _PROFILES:
                prefix = f"artifacts/{profile}/"
                expected = {
                    name.removeprefix(prefix): value
                    for name, value in manifest.items()
                    if name.startswith(prefix)
                }
                if _existing_profile_matches(root / profile, expected):
                    unchanged.append(profile)
            for profile in _PROFILES:
                if profile not in unchanged:
                    destination = root / profile
                    if destination.exists() or destination.is_symlink():
                        raise ValueError(f"Profile '{profile}' appeared during installation")
                    (stage / profile).rename(destination)
                    installed.append(profile)
        finally:
            lock.unlink(missing_ok=True)
    return {
        "version": MODEL_VERSION,
        "artifacts_root": str(root.resolve()),
        "archive_sha256": identity["sha256"],
        "installed": installed,
        "unchanged": unchanged,
    }


def download_models(artifacts_root: Path | str = "artifacts") -> dict:
    """Download the pinned release and install its three verified model profiles."""
    with tempfile.TemporaryDirectory(prefix="reflexone-download-") as temporary:
        archive = Path(temporary) / "models.tar.gz"
        _download_archive(MODEL_URL, archive, MODEL_BYTES)
        return _install_archive(
            archive,
            Path(artifacts_root),
            expected_sha256=MODEL_SHA256,
            expected_bytes=MODEL_BYTES,
        )
