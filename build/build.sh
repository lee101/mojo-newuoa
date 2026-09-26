#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
# contract=off is required for parity, not a performance knob: LLVM's default
# a+b*c -> FMA contraction changes the rounding of the trust-region model, and
# NEWUOA is chaotically sensitive to it on singular objectives.
mojo build --emit shared-lib --fp-mode contract=off \
    "$repo_dir/src/newuoa.mojo" \
    -o "$repo_dir/dist/libmojo-newuoa.so"
