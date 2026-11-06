import hashlib
import io
import json
import shutil
import tarfile
import urllib.error
from pathlib import Path

import pytest

from reflexone.cli import main

PROFILE_FILES = {
    "artifacts/support/model.safetensors": b"support weights",
    "artifacts/support/tokenizer/tokenizer.json": b'{"version":"1.0"}',
    "artifacts/commands/model.joblib": b"command weights",
    "artifacts/spam/model.joblib": b"spam weights",
}


def _archive(tmp_path, *, extra_members=(), corrupt_manifest=False, omit=()):
    payloads = {**PROFILE_FILES, "LICENSE": b"test license"}
    manifest = {
        "version": "0.1.0",
        "profiles": ["support", "commands", "spam"],
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        },
    }
    if corrupt_manifest:
        manifest["files"]["artifacts/support/model.safetensors"]["sha256"] = "0" * 64
    payloads["reports/artifact-manifest.json"] = json.dumps(manifest).encode()
    archive = tmp_path / "models.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in payloads.items():
            if name in omit:
                continue
            member = tarfile.TarInfo(name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
        for member in extra_members:
            stream.addfile(member, io.BytesIO(b"x" * member.size))
    return archive


def _install(archive, root, *, expected_sha256=None, expected_bytes=None):
    from reflexone import artifacts

    return artifacts._install_archive(
        archive,
        root,
        expected_sha256=expected_sha256 or hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_bytes=archive.stat().st_size if expected_bytes is None else expected_bytes,
    )


def test_valid_install_publishes_only_profiles_and_repeat_preserves_files(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"
    result = _install(archive, root)

    assert result["version"] == "0.1.0"
    assert set(result["installed"]) == {"support", "commands", "spam"}
    assert result["unchanged"] == []
    assert result["artifacts_root"] == str(root.resolve())
    assert {path.name for path in root.iterdir()} == {"support", "commands", "spam"}
    for name, data in PROFILE_FILES.items():
        assert (root / Path(name).relative_to("artifacts")).read_bytes() == data
    file = root / "support/model.safetensors"
    modified = file.stat().st_mtime_ns

    result = _install(archive, root)

    assert result["installed"] == []
    assert set(result["unchanged"]) == {"support", "commands", "spam"}
    assert file.stat().st_mtime_ns == modified
    json.dumps(result, allow_nan=False)


def test_invalid_archive_digest_never_publishes_models(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"

    with pytest.raises(ValueError, match="[Cc]hecksum|SHA256"):
        _install(archive, root, expected_sha256="0" * 64)

    assert not root.exists()


def test_truncated_archive_never_publishes_models(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"

    with pytest.raises(ValueError, match="[Ss]ize|bytes"):
        _install(archive, root, expected_bytes=archive.stat().st_size + 1)

    assert not root.exists()


@pytest.mark.parametrize(
    "name, kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("/tmp/reflexone-escaped", tarfile.REGTYPE),
        ("artifacts/support/../../escaped", tarfile.REGTYPE),
        ("artifacts/support/./model.bin", tarfile.REGTYPE),
        ("artifacts/unknown/model.joblib", tarfile.REGTYPE),
        ("artifacts/support/link", tarfile.SYMTYPE),
        ("artifacts/support/link", tarfile.LNKTYPE),
        ("artifacts/support/model.safetensors", tarfile.REGTYPE),
    ],
)
def test_unsafe_archive_never_publishes_models(tmp_path, name, kind):
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = "../../escaped"
    archive = _archive(tmp_path, extra_members=[member])
    root = tmp_path / "installed"

    with pytest.raises(ValueError, match="[Uu]nsafe|[Uu]nknown|[Dd]uplicate|[Uu]nexpected"):
        _install(archive, root)

    assert not root.exists()
    assert not (tmp_path / "escaped").exists()
    assert not list(tmp_path.glob(".reflexone-models-*"))


def test_manifest_hash_mismatch_never_publishes_models(tmp_path):
    archive = _archive(tmp_path, corrupt_manifest=True)
    root = tmp_path / "installed"

    with pytest.raises(ValueError, match="[Mm]anifest|[Cc]hecksum"):
        _install(archive, root)

    assert not root.exists()


def test_directory_archive_entries_cannot_hide_unverified_payloads(tmp_path):
    member = tarfile.TarInfo("artifacts/support/hidden")
    member.type = tarfile.DIRTYPE
    member.size = 1
    archive = _archive(tmp_path, extra_members=[member])
    root = tmp_path / "installed"

    with pytest.raises(ValueError, match="[Uu]nsafe|[Ii]nvalid"):
        _install(archive, root)

    assert not root.exists()


def test_missing_manifest_file_never_publishes_models(tmp_path):
    archive = _archive(tmp_path, omit=["artifacts/spam/model.joblib"])
    root = tmp_path / "installed"

    with pytest.raises(ValueError, match="[Mm]anifest|[Mm]issing"):
        _install(archive, root)

    assert not root.exists()


def test_partial_existing_profile_conflicts_before_any_profile_is_published(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"
    profile = root / "commands"
    profile.mkdir(parents=True)
    preserved = profile / "local-model.bin"
    preserved.write_bytes(b"keep this model")

    with pytest.raises(ValueError, match="[Cc]onflict.*commands|commands.*[Cc]onflict"):
        _install(archive, root)

    assert {path.name for path in root.iterdir()} == {"commands"}
    assert preserved.read_bytes() == b"keep this model"


def test_modified_existing_profile_conflicts_without_overwriting(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"
    _install(archive, root)
    model = root / "support/model.safetensors"
    model.write_bytes(b"modified local weights")

    with pytest.raises(ValueError, match="[Cc]onflict.*support|support.*[Cc]onflict"):
        _install(archive, root)

    assert model.read_bytes() == b"modified local weights"


def test_existing_profile_symlink_is_not_followed(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "support").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="[Cc]onflict|[Ss]ymlink"):
        _install(archive, root)

    assert not list(outside.iterdir())
    assert {path.name for path in root.iterdir()} == {"support"}


def test_existing_install_lock_prevents_publication_and_is_preserved(tmp_path):
    archive = _archive(tmp_path)
    root = tmp_path / "installed"
    root.mkdir()
    lock = root / ".reflexone-install.lock"
    lock.write_text("another installer")

    with pytest.raises(ValueError, match="installer.*lock|installer.*holds"):
        _install(archive, root)

    assert lock.read_text() == "another installer"
    assert list(root.iterdir()) == [lock]


def test_download_restarts_stream_after_transient_network_error(tmp_path, monkeypatch):
    from reflexone import artifacts

    attempts = 0

    class InterruptedStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("connection lost")
            return super().read(2)

    def open_release(request, *, timeout):
        nonlocal attempts
        assert 0 < timeout <= 60
        attempts += 1
        return InterruptedStream(b"ab") if attempts == 1 else io.BytesIO(b"complete model")

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", open_release)
    monkeypatch.setattr(artifacts.time, "sleep", lambda _: None)
    destination = tmp_path / "download.tar.gz"

    artifacts._download_archive("https://example.test/models.tar.gz", destination, 14)

    assert destination.read_bytes() == b"complete model"
    assert attempts == 2


def test_download_stops_after_three_failures_and_removes_partial_file(tmp_path, monkeypatch):
    from reflexone import artifacts

    attempts = 0

    def unavailable(request, *, timeout):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(artifacts.time, "sleep", lambda _: None)
    destination = tmp_path / "download.tar.gz"

    with pytest.raises(RuntimeError, match="[Dd]ownload.*3|3.*[Dd]ownload"):
        artifacts._download_archive("https://example.test/models.tar.gz", destination, 14)

    assert attempts == 3
    assert not destination.exists()


def test_download_models_cli_installs_profiles_and_emits_json(tmp_path, monkeypatch, capsys):
    from reflexone import artifacts

    archive = _archive(tmp_path)
    monkeypatch.setattr(artifacts, "MODEL_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(artifacts, "MODEL_BYTES", archive.stat().st_size)

    def download(url, destination, expected_bytes):
        shutil.copyfile(archive, destination)

    monkeypatch.setattr(artifacts, "_download_archive", download)
    root = tmp_path / "installed"

    assert main(["download-models", "--artifacts-root", str(root)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert set(result["installed"]) == {"support", "commands", "spam"}
    assert (root / "spam/model.joblib").read_bytes() == b"spam weights"
