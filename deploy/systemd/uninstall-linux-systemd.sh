#!/usr/bin/env bash
# uninstall-linux-systemd.sh — Cleanly stops and removes BeoFold systemd services
# Pass `--purge-data` to also delete `/etc/beofold`, `/var/lib/beofold`, and `/var/lib/beofold-node`.

set -euo pipefail

PURGE_DATA=0
if [[ "${1:-}" == "--purge-data" ]]; then
    PURGE_DATA=1
fi

if [[ "${EUID}" -ne 0 ]]; then
    echo "[ERROR] Please run as root (sudo $0 [--purge-data])." >&2
    exit 1
fi

for unit in beofold-node.service beofold-sync.service; do
    if systemctl list-unit-files | grep -q "^${unit}"; then
        systemctl stop "${unit}" || true
        systemctl disable "${unit}" || true
        rm -f "/etc/systemd/system/${unit}"
        echo "[-] Stopped and removed ${unit}"
    fi
done

systemctl daemon-reload
rm -f /usr/local/bin/fold-node /usr/local/bin/fold-sync

if [[ "${PURGE_DATA}" -eq 1 ]]; then
    rm -rf /etc/beofold /var/lib/beofold-node /var/lib/beofold
    echo "[-] Purged /etc/beofold and /var/lib/beofold* trajectory data"
else
    echo "[=] Preserved /etc/beofold and /var/lib/beofold* vaults (pass --purge-data to delete)"
fi
