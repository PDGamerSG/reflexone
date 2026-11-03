"""Package the three primary trained profiles with checksums and attribution."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    profiles = ("support", "commands", "spam")
    paths = []
    for name in profiles:
        directory = ROOT / "artifacts" / name
        if not (directory / "metadata.json").is_file():
            raise SystemExit(f"Missing trained profile: {directory}")
        paths.extend(p for p in directory.rglob("*") if p.is_file())
    paths.extend(ROOT / name for name in ("THIRD_PARTY.md", "LICENSE", "licenses/Apache-2.0.txt"))
    manifest = {
        "version": "0.1.0",
        "profiles": list(profiles),
        "primary_neural_seed": 42,
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(paths)
        },
    }
    manifest_path = ROOT / "reports" / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive_path = output / "reflexone-models-v0.1.0.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=5) as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        archive.add(manifest_path, arcname="reports/artifact-manifest.json", recursive=False)
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (output / "SHA256SUMS").write_text(f"{checksum}  {archive_path.name}\n")
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": checksum,
                "files": len(paths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
