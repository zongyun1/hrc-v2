#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")" && pwd)
exec bash "$PROJECT/tools/isaaclab3/setup.sh" "$@"
