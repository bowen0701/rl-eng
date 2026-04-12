#!/usr/bin/env bash
# =============================================================================
# games/tic_tac_toe/build_app.sh
# =============================================================================
# Packages the RL-Eng Tic-Tac-Toe pygame game into a self-contained macOS
# .app bundle using PyInstaller.
#
# Usage (from project root):
#   chmod +x games/tic_tac_toe/build_app.sh
#   ./games/tic_tac_toe/build_app.sh --run_id <run_id>
#
# Requirements (install once):
#   pip3 install pyinstaller pygame
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------------------------

RUN_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "[build_app] Unknown argument: $1" >&2
            echo "Usage: ./games/tic_tac_toe/build_app.sh --run_id <run_id>" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$RUN_ID" ]]; then
    echo "[build_app] Error: --run_id is required." >&2
    exit 1
fi

APP_NAME="TicTacToe"
RUN_DIR="runs/${RUN_ID}"
GAME_DIR="games/tic_tac_toe"

echo "============================================================"
echo " RL-Eng Tic-Tac-Toe -- macOS App Builder"
echo "============================================================"
echo " App name : ${APP_NAME}"
echo " Run ID   : ${RUN_ID}"
echo " Run dir  : ${RUN_DIR}"
echo " Game dir : ${GAME_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 2. Validate run directory
# ---------------------------------------------------------------------------

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[build_app] Error: run directory not found: ${RUN_DIR}" >&2
    exit 1
fi
# ---------------------------------------------------------------------------
# 3. Ensure PyInstaller is installed.
# ---------------------------------------------------------------------------

if ! python3 -c "import PyInstaller" &>/dev/null; then
    echo "[build_app] PyInstaller not found -- installing..."
    pip3 install pyinstaller
else
    echo "[build_app] PyInstaller already installed."
fi

# ---------------------------------------------------------------------------
# 4. Clean up previous build artifacts so we get a fresh build.
# ---------------------------------------------------------------------------

echo "[build_app] Cleaning previous build artifacts..."
rm -rf "${GAME_DIR}/build/" "${GAME_DIR}/dist/" "${GAME_DIR}/${APP_NAME}.spec"

# ---------------------------------------------------------------------------
# 5. Run PyInstaller via python3 -m to ensure it uses the correct environment.
# ---------------------------------------------------------------------------

echo "[build_app] Running PyInstaller..."

# Use absolute paths for source directories to avoid path resolution issues
# when --specpath is used.
ABS_ROOT="$(pwd)"

python3 -m PyInstaller \
    --onedir \
    --windowed \
    --name "${APP_NAME}" \
    --workpath "${GAME_DIR}/build" \
    --distpath "${GAME_DIR}/dist" \
    --specpath "${GAME_DIR}" \
    --add-data "${ABS_ROOT}/rl_eng:rl_eng" \
    --add-data "${ABS_ROOT}/${RUN_DIR}:${RUN_DIR}" \
    --add-data "${ABS_ROOT}/${GAME_DIR}:games/tic_tac_toe" \
    --hidden-import "rl_eng" \
    --hidden-import "rl_eng.tic_tac_toe" \
    --hidden-import "pygame" \
    "${GAME_DIR}/launcher.py"

echo "[build_app] PyInstaller finished."


# ---------------------------------------------------------------------------
# 6. Inject wrapper
# ---------------------------------------------------------------------------

MACOS_DIR="${GAME_DIR}/dist/${APP_NAME}.app/Contents/MacOS"
REAL_BIN="${MACOS_DIR}/${APP_NAME}_bin"
WRAPPER="${MACOS_DIR}/${APP_NAME}"

echo "[build_app] Injecting run_id wrapper script..."
mv "${MACOS_DIR}/${APP_NAME}" "${REAL_BIN}"

cat > "${WRAPPER}" <<WRAPPER_SCRIPT
#!/usr/bin/env bash
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
exec "\${DIR}/${APP_NAME}_bin" --run_id "${RUN_ID}" "\$@"
WRAPPER_SCRIPT

chmod +x "${WRAPPER}"

# ---------------------------------------------------------------------------
# 7. Ad-hoc sign
# ---------------------------------------------------------------------------

echo "[build_app] Signing the app..."
codesign --deep --force --sign "-" "${GAME_DIR}/dist/${APP_NAME}.app" || true

# ---------------------------------------------------------------------------
# 8. Zip
# ---------------------------------------------------------------------------

echo "[build_app] Creating distributable zip..."
cd "${GAME_DIR}/dist"
zip -r "${APP_NAME}.zip" "${APP_NAME}.app"
cd - > /dev/null

echo "============================================================"
echo " Build complete: ${GAME_DIR}/dist/${APP_NAME}.zip"
echo "============================================================"
