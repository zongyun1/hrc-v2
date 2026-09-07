#!/usr/bin/env bash
# Fetch and unpack assets/objects/ from the pre-zipped OneDrive bundle.
#
# assets/objects/ has 13k+ items, past OneDrive/SharePoint's per-download
# item ceiling for its own "Download as Zip" folder feature -- that button
# silently produces a truncated/corrupted archive on this folder. objects.zip
# is a single pre-built archive uploaded alongside the raw folder (see
# sync_assets.sh) specifically to avoid that: downloading one existing file
# never triggers OneDrive's server-side zip generation, so it can't hit the
# same failure mode.
#
# Usage:
#   scripts/fetch_objects_zip.sh              # via rclone (remote "onedrive")
#   scripts/fetch_objects_zip.sh --url URL    # via direct HTTPS, no rclone

set -euo pipefail

REMOTE_ZIP=onedrive:/Shared/HumanRobot/assets/objects.zip
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/assets"
TMP_ZIP="$(mktemp -t objects.XXXXXX.zip)"
trap 'rm -f "$TMP_ZIP"' EXIT

url=""
if [[ "${1:-}" == "--url" ]]; then
    url="${2:?--url requires a value}"
fi

if [[ -n "$url" ]]; then
    echo "=== downloading $url ==="
    curl -fL --progress-bar -o "$TMP_ZIP" "$url"
else
    echo "=== rclone copyto $REMOTE_ZIP -> $TMP_ZIP ==="
    rclone copyto "$REMOTE_ZIP" "$TMP_ZIP" --progress
fi

echo "=== verifying zip integrity ==="
unzip -t "$TMP_ZIP" > /dev/null

echo "=== unpacking into $ASSETS ==="
mkdir -p "$ASSETS"
unzip -q -o "$TMP_ZIP" -d "$ASSETS"

echo "done: $ASSETS/objects populated from objects.zip"
