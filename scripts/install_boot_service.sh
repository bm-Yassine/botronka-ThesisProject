#!/usr/bin/env bash
set -euo pipefail

# Installs botfriend.service into systemd and enables it at boot.
# Usage:
#   sudo bash scripts/install_boot_service.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SRC="$PROJECT_DIR/scripts/botfriend.service"
SERVICE_DST="/etc/systemd/system/botfriend.service"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/install_boot_service.sh" >&2
  exit 1
fi

if [[ ! -f "$SERVICE_SRC" ]]; then
  echo "Service template not found: $SERVICE_SRC" >&2
  exit 1
fi

chmod +x "$PROJECT_DIR/scripts/botfriend_boot.sh"
cp "$SERVICE_SRC" "$SERVICE_DST"

systemctl daemon-reload
systemctl enable botfriend.service
systemctl restart botfriend.service

echo "Installed and started botfriend.service"
echo
echo "Check status with:"
echo "  systemctl status botfriend.service --no-pager"
echo "  journalctl -u botfriend.service -f"
