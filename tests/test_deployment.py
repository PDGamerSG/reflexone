"""Publication must use committed source and exclude ignored local credentials."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def deploy_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "deploy_space.py"
    spec = importlib.util.spec_from_file_location("reflexone_deploy_space", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    for name, text in {
        "Dockerfile": "FROM python:3.12-slim\n",
        ".dockerignore": "*.env\n",
        ".gitignore": ".env\n",
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "LICENSE": "fixture license\n",
        "THIRD_PARTY.md": "fixture attribution\n",
        "licenses/Apache-2.0.txt": "fixture upstream license\n",
        "src/reflexone/__init__.py": "",
        "src/reflexone/api.py": "# committed source\n",
        "deploy/space/README.md": "---\nsdk: docker\n---\nReflexOne\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def commit(root):
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def test_space_upload_excludes_ignored_credentials_and_uses_committed_bytes(
    source_repo, deploy_module, monkeypatch, tmp_path
):
    revision = commit(source_repo)
    (source_repo / "src/reflexone/.env").write_text("PRIVATE_FIXTURE_TOKEN=never_upload\n")
    (source_repo / "src/reflexone/api.py").write_text("# later working copy\n")
    monkeypatch.setattr(deploy_module, "ROOT", source_repo)
    destination = tmp_path / "stage"
    destination.mkdir()
    files = deploy_module.stage_source(destination, revision)
    assert "src/reflexone/.env" not in files
    assert (destination / "src/reflexone/api.py").read_text() == "# committed source\n"
    assert revision in (destination / "README.md").read_text()


def test_space_upload_rejects_committed_symlinks(source_repo, deploy_module, monkeypatch, tmp_path):
    (source_repo / "src/reflexone/alias.py").symlink_to("api.py")
    revision = commit(source_repo)
    monkeypatch.setattr(deploy_module, "ROOT", source_repo)
    destination = tmp_path / "stage"
    destination.mkdir()
    with pytest.raises(ValueError, match="symlink|regular"):
        deploy_module.stage_source(destination, revision)
