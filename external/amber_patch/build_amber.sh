#!/bin/bash
# external/amber_patch/build_amber.sh [install_dir]
#
# [ksz-pipeline] Clone AMBER at the commit the adapter was validated against,
# apply the field-dump patch, add the HEALPix stub, build with Intel ifort+MKL.
# No admin rights needed. Re-running is safe (idempotent).
#
#   bash external/amber_patch/build_amber.sh            # -> ~/amber/src/amber.x
#   ONEAPI=/other/path bash external/amber_patch/build_amber.sh ~/amber_alt

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:-$HOME/amber}
AMBER_COMMIT=9703f515a70b17e9bac38457d4891dad9abf7f84    # tested 2026-09-10
ONEAPI=${ONEAPI:-/apps/intel/oneapi}

# setvars.sh is not 'set -u'-safe, so source it before enabling strict mode
if [ -f "$ONEAPI/setvars.sh" ]; then
  source "$ONEAPI/setvars.sh" >/dev/null 2>&1 || true
else
  echo "ERROR: $ONEAPI/setvars.sh not found (set ONEAPI=...)"; exit 1
fi
set -euo pipefail
command -v ifort >/dev/null || { echo "ERROR: ifort not on PATH after setvars"; exit 1; }
[ -n "${MKLROOT:-}" ]       || { echo "ERROR: MKLROOT unset -- MKL not found"; exit 1; }
echo "ifort: $(ifort --version 2>&1 | head -1)"
echo "MKLROOT: $MKLROOT"

# --- source at pinned commit + patch -------------------------------------
if [ ! -d "$DEST/.git" ]; then
  git clone -q https://github.com/hytrac/amber.git "$DEST"
fi
cd "$DEST"
git fetch -q origin || true
if grep -q cmb_dumpfields src/cmbreion.f90 2>/dev/null \
   && [ "$(git rev-parse HEAD)" = "$AMBER_COMMIT" ]; then
  echo "AMBER already at $AMBER_COMMIT with patch applied"
else
  git checkout -q -- . && git checkout -q "$AMBER_COMMIT"
  git apply "$HERE/0001-dump-fields-for-ksz-pipeline.patch"
  echo "AMBER checked out at $AMBER_COMMIT, dump patch applied"
fi
cp "$HERE/healpix_stub.f90" "$HERE/Makefile.ksz" src/
cd src

# --- MKL flag: ifort only WARNS on unknown options, so check the message ---
tmp=$(mktemp -d); printf 'program t\nend program t\n' > "$tmp/t.f90"
if ifort -qmkl=parallel "$tmp/t.f90" -o "$tmp/t" 2>&1 | grep -qi "unknown option\|ignoring"; then
  MKLFLAG=-mkl=parallel
else
  MKLFLAG=-qmkl=parallel
fi
rm -rf "$tmp"; echo "MKL flag: $MKLFLAG"

# --- build ----------------------------------------------------------------
make -f Makefile.ksz clean >/dev/null
make -f Makefile.ksz MKLFLAG="$MKLFLAG" 2>&1 | tee build.log | grep -iE "error" && {
  echo "BUILD FAILED -- see $DEST/src/build.log"; exit 1; } || true
[ -x amber.x ] || { echo "BUILD FAILED: no amber.x -- see $DEST/src/build.log"; exit 1; }
echo
echo "OK: $DEST/src/amber.x"
echo "Runtime needs the same environment:  source $ONEAPI/setvars.sh"
