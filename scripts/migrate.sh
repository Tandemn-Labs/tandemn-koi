#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

exec dbmate \
    --wait \
    --wait-timeout 60s \
    --migration-lock \
    --migration-lock-timeout 5m \
    --no-dump-schema \
    --migrations-dir "$script_dir/../db/migrations" \
    migrate \
    --strict
