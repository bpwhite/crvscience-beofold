#!/usr/bin/env bash
# Copyright (C) 2026 CRV Science Network (https://crvscience.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://gnu.org>.

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
