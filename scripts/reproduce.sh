#!/bin/sh
# Train into a new directory; existing released artifacts stay available for comparison.
set -eu
run_dir="${REFLEXONE_RUN_DIR:-runs/reproduction}"
mkdir -p "$run_dir/reports"
uv sync --frozen --extra dev
uv run reflexone prepare --data data/support > "$run_dir/support-data.json"
uv run reflexone train --data data/support --artifacts "$run_dir/artifacts/support" --output "$run_dir/reports/support.json"
for profile in commands spam; do
  uv run reflexone prepare-task "$profile" --data "data/$profile" --output "$run_dir/$profile-data.json"
  uv run reflexone train-task "$profile" --data "data/$profile" --artifacts "$run_dir/artifacts/$profile" --output "$run_dir/reports/$profile.json"
done
uv run reflexone benchmark --artifacts "$run_dir/artifacts/support" --output "$run_dir/reports/support-benchmark.json"
uv run pytest -q
