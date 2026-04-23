#!/usr/bin/env bash
# Generate docs/src/release-notes.md from CHANGELOG.md,
# stripping the [Unreleased] section.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHANGELOG="$REPO_ROOT/CHANGELOG.md"
OUTPUT="$REPO_ROOT/docs/src/release-notes.md"

if [ ! -f "$CHANGELOG" ]; then
    echo "error: $CHANGELOG not found" >&2
    exit 1
fi

{
    # Emit the page title
    echo "# Release Notes"
    echo ""

    # Skip everything until the first versioned release heading (## [x.y.z])
    # This drops the file header and the [Unreleased] section.
    sed -n '/^## \[[0-9]/,$p' "$CHANGELOG"
} > "$OUTPUT"

echo "Generated $OUTPUT"
