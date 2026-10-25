#!/usr/bin/env bash
# LightGBM на macOS ищет libomp из Homebrew. Без brew: берём libomp из колеса scikit-learn
# и перенаправляем ссылку в lib_lightgbm.dylib на копию рядом с ней. Меняет только .venv.
set -euo pipefail
SP=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SP/lightgbm/lib/lib_lightgbm.dylib"
cp "$SP/sklearn/.dylibs/libomp.dylib" "$SP/lightgbm/lib/libomp.dylib"
OLD=$(otool -L "$LIB" | awk '/libomp/ {print $1}')
[ "$OLD" != "@loader_path/libomp.dylib" ] && install_name_tool -change "$OLD" "@loader_path/libomp.dylib" "$LIB"
codesign --force -s - "$LIB" 2>/dev/null || true
echo "lightgbm: libomp -> @loader_path/libomp.dylib"
