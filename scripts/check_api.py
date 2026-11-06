"""Check a running ReflexOne deployment using its real HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PROFILES = {"support", "commands", "spam"}


def check_api(base_url: str, wait_seconds: float = 120) -> dict:
    base_url = base_url.rstrip("/")

    def request(path: str, payload: dict | None = None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base_url + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ReflexOne-deployment-check",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            ready = request("/ready")
            if ready.get("ready") and set(ready.get("loaded_profiles", [])) == PROFILES:
                break
            raise ValueError("Readiness did not confirm all three profiles")
        except (OSError, ValueError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Deployment did not become ready: {type(exc).__name__}"
                ) from exc
            time.sleep(3)

    profiles = request("/profiles")
    if set(profiles["loaded"]) != PROFILES or profiles["max_batch_size"] != 32:
        raise ValueError("Unexpected deployment profile contract")
    cases = [
        {"profile": "support", "domain": "banking", "text": "My card was stolen last night."},
        {"profile": "support", "domain": "hotels", "text": "I need a room tomorrow"},
        {"profile": "commands", "text": "Please play jazz music"},
        {"profile": "spam", "text": "WIN a FREE cash prize"},
    ]
    single = request("/predict", cases[0])
    batch = request("/predict/batch", {"requests": cases})
    if single.get("profile") != "support" or single.get("domain") != "banking":
        raise ValueError("Single support prediction failed")
    if (
        len(batch) != 4
        or batch[0].get("domain") != "banking"
        or batch[1].get("domain") != "hotels"
        or batch[2].get("label") != "play_music"
        or batch[3].get("label") != "spam"
    ):
        raise ValueError("Mixed prediction smoke check failed")
    try:
        request("/predict", {"profile": "support", "text": "missing domain"})
    except urllib.error.HTTPError as exc:
        if exc.code != 422:
            raise
    else:
        raise ValueError("Invalid support request was accepted")
    return {
        "base_url": base_url,
        "ready": True,
        "loaded_profiles": sorted(PROFILES),
        "mixed_predictions": len(batch),
        "invalid_request_status": 422,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Public HTTPS URL or local HTTP URL")
    parser.add_argument("--wait-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    try:
        print(json.dumps(check_api(args.url, args.wait_seconds), indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
