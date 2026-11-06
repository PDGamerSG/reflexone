"""Publish the tested inference source to a free Hugging Face Docker Space."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/PDGamerSG/reflexone"


def stage_source(destination: Path, revision: str) -> list[str]:
    """Upload committed allowlisted files, never ignored files or working-copy changes."""
    required = {
        "Dockerfile",
        ".dockerignore",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "THIRD_PARTY.md",
    }
    data = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=tar",
            revision,
            "--",
            *sorted(required),
            "src/reflexone",
            "licenses",
        ],
        cwd=ROOT,
    )
    copied = set()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"Deployment source must be a regular file, never a symlink: {member.name}"
                )
            if path.parts[:2] == ("src", "reflexone") and not (
                path.suffix == ".py" or path.as_posix() == "src/reflexone/resources/ontology.json"
            ):
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as content:
                target.write_bytes(content.read())
            copied.add(member.name)
    if not required <= copied:
        raise ValueError("The selected commit does not contain the complete deployment package")
    readme = subprocess.check_output(
        ["git", "show", f"{revision}:deploy/space/README.md"], cwd=ROOT, text=True
    )
    readme += f"\nSource revision: [{revision}]({SOURCE_URL}/commit/{revision}).\n"
    (destination / "README.md").write_text(readme)
    return sorted(
        str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()
    )


def deploy(source: Path, revision: str, token: str, space_id: str | None = None) -> dict:
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    space_id = space_id or f"{owner}/reflexone"
    if space_id.split("/")[0] != owner:
        raise ValueError("Deploy to your own namespace; organization deployment is not configured")
    try:
        existing = api.space_info(space_id)
    except RepositoryNotFoundError:
        existing = None
    if existing is not None:
        readme_path = hf_hub_download(
            space_id, "README.md", repo_type="space", revision=existing.sha, token=token
        )
        if (
            SOURCE_URL not in Path(readme_path).read_text()
            or existing.private
            or existing.sdk != "docker"
        ):
            raise ValueError("Refusing to overwrite an unrelated, private, or non-Docker Space")
    else:
        api.create_repo(
            space_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic",
            private=False,
        )
    commit = api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=source,
        commit_message=f"Deploy ReflexOne source {revision[:12]}",
        parent_commit=existing.sha if existing is not None else None,
        delete_patterns=["src/**", "licenses/**"],
    )
    info = api.space_info(space_id)
    return {
        "space_id": space_id,
        "space_url": f"https://huggingface.co/spaces/{space_id}",
        "api_url": info.host,
        "deployment_commit": commit.oid,
        "source_revision": revision,
        "runtime_stage": info.runtime.stage if info.runtime else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", help="Your namespace/space; defaults to your account/reflexone")
    parser.add_argument(
        "--token-file", type=Path, help="Private token file; never committed or uploaded"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Review the upload allowlist without publishing"
    )
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
        if dirty and not args.dry_run:
            raise ValueError("Commit and verify the source before deploying it")
        with tempfile.TemporaryDirectory(prefix="reflexone-space-") as directory:
            source = Path(directory)
            files = stage_source(source, revision)
            if args.dry_run:
                result = {
                    "source_revision": revision,
                    "dirty": dirty,
                    "files": files,
                    "published": False,
                }
            else:
                token = args.token_file.read_text().strip() if args.token_file else get_token()
                if not token:
                    raise ValueError(
                        "Hugging Face write access is required; run hf auth login first"
                    )
                result = deploy(source, revision, token, args.space)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, HfHubHTTPError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
