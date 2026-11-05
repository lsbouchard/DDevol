#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build=${1:-"$here/build_rev"}
if [ "$#" -gt 0 ]; then shift; fi

if [ "$(uname -s)" = Darwin ]; then
    # Prefer a complete Xcode installation when its SDK and command-line
    # linker are mismatched. Respect an explicitly selected developer path.
    if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
        DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
        export DEVELOPER_DIR
    fi
    sdk=$(xcrun --sdk macosx --show-sdk-path)
    CC=${CC:-$(xcrun --find clang)}
    OBJC=${OBJC:-$CC}
    export CC OBJC
    cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_OSX_SYSROOT=$sdk" "$@"
else
    cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=Release "$@"
fi
cmake --build "$build" --parallel "${TPGP_BUILD_JOBS:-4}"
ctest --test-dir "$build" --output-on-failure
