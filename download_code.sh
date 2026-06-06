#!/usr/bin/env bash

set -e

echo "============================================================"
echo " Download and refresh ubench"
echo "============================================================"
echo ""

# -----------------------------
# 0. Helpers
# -----------------------------

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*"
}

err() {
  echo "[ERROR] $*" >&2
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# -----------------------------
# 1. Locate script directory
# -----------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

# -----------------------------
# 2. Check dependencies
# -----------------------------

if ! command_exists curl; then
  err "curl is not installed."
  exit 1
fi

if ! command_exists unzip; then
  err "unzip is not installed."
  echo ""
  echo "Please install unzip first, for example:"
  echo "  sudo apt-get install -y unzip"
  echo "  sudo yum install -y unzip"
  echo ""
  exit 1
fi

# -----------------------------
# 3. Clean old files
# -----------------------------

echo ""
echo "------------------------------------------------------------"
echo " Cleaning old ubench files"
echo "------------------------------------------------------------"

rm -rf "$SCRIPT_DIR/ubench"
rm -rf "$SCRIPT_DIR/ubench-main"
rm -rf "$SCRIPT_DIR/ubench-master"
rm -f "$SCRIPT_DIR/ubench.zip"

info "Old ubench files removed."

# -----------------------------
# 4. Download ubench zip
# -----------------------------

echo ""
echo "------------------------------------------------------------"
echo " Downloading ubench"
echo "------------------------------------------------------------"

UBENCH_ZIP="$SCRIPT_DIR/ubench.zip"
UBENCH_URL="https://gh-proxy.com/https://github.com/mkbk-with-circle/ubench/archive/refs/heads/main.zip"

info "Download URL: $UBENCH_URL"

curl -L --connect-timeout 15 --max-time 180 \
  -o "$UBENCH_ZIP" \
  "$UBENCH_URL"

if [ ! -s "$UBENCH_ZIP" ]; then
  err "Download failed or ubench.zip is empty."
  exit 1
fi

info "Downloaded file:"
ls -lh "$UBENCH_ZIP"

# -----------------------------
# 5. Unzip
# -----------------------------

echo ""
echo "------------------------------------------------------------"
echo " Extracting ubench"
echo "------------------------------------------------------------"

unzip -q "$UBENCH_ZIP" -d "$SCRIPT_DIR"

if [ -d "$SCRIPT_DIR/ubench-main" ]; then
  mv "$SCRIPT_DIR/ubench-main" "$SCRIPT_DIR/ubench"
elif [ -d "$SCRIPT_DIR/ubench-master" ]; then
  mv "$SCRIPT_DIR/ubench-master" "$SCRIPT_DIR/ubench"
else
  err "Cannot find extracted directory: ubench-main or ubench-master"
  echo ""
  echo "Extracted files:"
  ls -la "$SCRIPT_DIR"
  exit 1
fi

# -----------------------------
# 6. Verify
# -----------------------------

echo ""
echo "------------------------------------------------------------"
echo " Done"
echo "------------------------------------------------------------"

if [ -d "$SCRIPT_DIR/ubench" ]; then
  info "ubench has been refreshed successfully."
  info "ubench path: $SCRIPT_DIR/ubench"
  echo ""
  echo "Directory contents:"
  ls -la "$SCRIPT_DIR/ubench"
else
  err "ubench directory was not created."
  exit 1
fi

echo ""
echo "============================================================"
echo " Finished"
echo "============================================================"
