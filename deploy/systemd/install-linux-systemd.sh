#!/usr/bin/env bash
# install-linux-systemd.sh — Unattended Linux systemd installer for BeoFold (fold-node & fold-sync)
# Usage examples:
#   sudo ./deploy/systemd/install-linux-systemd.sh --role node --cluster-secret "my-lan-key" --open-firewall
#   sudo ./deploy/systemd/install-linux-systemd.sh --role sync --quorum 1 --init-trpcage
#   sudo ./deploy/systemd/install-linux-systemd.sh --role both --cluster-secret "my-lan-key" --sync-url "https://fold.example.org/beofold/api/v1"

set -euo pipefail

ROLE="node"
BIN_SRC_DIR="./target/release"
INSTALL_BIN_DIR="/usr/local/bin"
CONF_DIR="/etc/beofold"
SYSTEMD_DIR="/etc/systemd/system"
CLUSTER_SECRET="beofold-home-cluster"
BIND_ADDR="0.0.0.0:7443"
SYNC_URL=""
CLUSTER_NAME="Home-Beowulf-Linux"
NODE_LABEL="$(hostname -s 2>/dev/null || echo 'linux-beowulf-node')"
EXTRA_NODE_ARGS="--auto-throttle --max-cpu 85"
SYNC_BIND="127.0.0.1:8780"
SYNC_QUORUM="1"
EXTRA_SYNC_ARGS=""
OPEN_FIREWALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)            ROLE="$2"; shift 2 ;;
        --bin-dir)         BIN_SRC_DIR="$2"; shift 2 ;;
        --cluster-secret)  CLUSTER_SECRET="$2"; shift 2 ;;
        --bind)            BIND_ADDR="$2"; shift 2 ;;
        --sync-url)        SYNC_URL="$2"; shift 2 ;;
        --cluster-name)    CLUSTER_NAME="$2"; shift 2 ;;
        --label)           NODE_LABEL="$2"; shift 2 ;;
        --node-extra-args) EXTRA_NODE_ARGS="$2"; shift 2 ;;
        --sync-bind)       SYNC_BIND="$2"; shift 2 ;;
        --quorum)          SYNC_QUORUM="$2"; shift 2 ;;
        --init-trpcage)    EXTRA_SYNC_ARGS="${EXTRA_SYNC_ARGS} --init-trpcage"; EXTRA_NODE_ARGS="${EXTRA_NODE_ARGS} --init-trpcage"; shift ;;
        --open-firewall)   OPEN_FIREWALL=1; shift ;;
        -h|--help)
            echo "Usage: sudo $0 [--role node|sync|both] [--bin-dir ./target/release] [--cluster-secret KEY] [--sync-url URL] [--cluster-name NAME] [--label HOST] [--quorum 1|2] [--init-trpcage] [--open-firewall]"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${EUID}" -ne 0 ]]; then
    echo "[ERROR] Please run this installer as root (sudo $0 ...)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Create dedicated unprivileged system user & group `beofold`
if ! getent group beofold >/dev/null 2>&1; then
    groupadd --system beofold
    echo "[+] Created system group 'beofold'"
fi

if ! id -u beofold >/dev/null 2>&1; then
    useradd --system --gid beofold --home-dir /var/lib/beofold-node --no-create-home --shell /usr/sbin/nologin beofold
    echo "[+] Created system user 'beofold'"
fi

# Grant headless Vulkan / DRM render-node access (`/dev/dri/renderD128`) when groups exist
for gpu_grp in render video; do
    if getent group "${gpu_grp}" >/dev/null 2>&1; then
        usermod -aG "${gpu_grp}" beofold
        echo "[+] Added 'beofold' to GPU device group '${gpu_grp}'"
    fi
done

install -d -m 0750 -o root -g beofold "${CONF_DIR}"

install_node_service() {
    local src_bin="${BIN_SRC_DIR}/fold-node"
    if [[ ! -x "${src_bin}" ]]; then
        echo "[ERROR] Release binary not found at '${src_bin}'. Run 'cargo build --release -p fold-node' first." >&2
        exit 1
    fi

    install -m 0755 "${src_bin}" "${INSTALL_BIN_DIR}/fold-node"
    install -d -m 0750 -o beofold -g beofold /var/lib/beofold-node

    if [[ ! -f "${CONF_DIR}/node.env" ]]; then
        cat > "${CONF_DIR}/node.env" <<EOF
BEOFOLD_BIND_ADDR=${BIND_ADDR}
BEOFOLD_CLUSTER_SECRET=${CLUSTER_SECRET}
BEOFOLD_NODE_LABEL=${NODE_LABEL}
BEOFOLD_SYNC_URL="${SYNC_URL}"
BEOFOLD_CLUSTER_NAME="${CLUSTER_NAME}"
BEOFOLD_NODE_EXTRA_ARGS="${EXTRA_NODE_ARGS}"
EOF
        chmod 0600 "${CONF_DIR}/node.env"
        chown root:root "${CONF_DIR}/node.env"
        echo "[+] Wrote ${CONF_DIR}/node.env (0600)"
    else
        echo "[=] Preserved existing ${CONF_DIR}/node.env"
    fi

    install -m 0644 "${SCRIPT_DIR}/beofold-node.service" "${SYSTEMD_DIR}/beofold-node.service"

    if [[ "${OPEN_FIREWALL}" -eq 1 ]]; then
        if command -v ufw >/dev/null 2>&1; then
            ufw allow 7443/udp comment "BeoFold QUIC TLS 1.3 LAN Mesh" || true
            ufw allow 5353/udp comment "BeoFold mDNS Discovery" || true
            echo "[+] Opened UDP 7443 & 5353 in ufw"
        elif command -v firewall-cmd >/dev/null 2>&1; then
            firewall-cmd --permanent --add-port=7443/udp || true
            firewall-cmd --permanent --add-port=5353/udp || true
            firewall-cmd --reload || true
            echo "[+] Opened UDP 7443 & 5353 in firewalld"
        fi
    fi

    systemctl daemon-reload
    systemctl enable --now beofold-node.service
    echo "[+] Enabled and started beofold-node.service (journalctl -u beofold-node -f)"
}

install_sync_service() {
    local src_bin="${BIN_SRC_DIR}/fold-sync"
    if [[ ! -x "${src_bin}" ]]; then
        echo "[ERROR] Release binary not found at '${src_bin}'. Run 'cargo build --release -p fold-sync' first." >&2
        exit 1
    fi

    install -m 0755 "${src_bin}" "${INSTALL_BIN_DIR}/fold-sync"
    # 0755 on /var/lib/beofold so Apache static Alias `/beofold/blobs/` can read immutable LZ4 blobs
    install -d -m 0755 -o beofold -g beofold /var/lib/beofold
    install -d -m 0755 -o beofold -g beofold /var/lib/beofold/blobs

    if [[ ! -f "${CONF_DIR}/sync.env" ]]; then
        cat > "${CONF_DIR}/sync.env" <<EOF
BEOFOLD_SYNC_BIND=${SYNC_BIND}
BEOFOLD_SYNC_QUORUM=${SYNC_QUORUM}
BEOFOLD_SYNC_EXTRA_ARGS="${EXTRA_SYNC_ARGS}"
EOF
        chmod 0600 "${CONF_DIR}/sync.env"
        chown root:root "${CONF_DIR}/sync.env"
        echo "[+] Wrote ${CONF_DIR}/sync.env (0600)"
    else
        echo "[=] Preserved existing ${CONF_DIR}/sync.env"
    fi

    install -m 0644 "${SCRIPT_DIR}/beofold-sync.service" "${SYSTEMD_DIR}/beofold-sync.service"

    systemctl daemon-reload
    systemctl enable --now beofold-sync.service
    echo "[+] Enabled and started beofold-sync.service (journalctl -u beofold-sync -f)"
}

case "${ROLE}" in
    node) install_node_service ;;
    sync) install_sync_service ;;
    both) install_sync_service; install_node_service ;;
    *)
        echo "[ERROR] Invalid --role '${ROLE}' (expected: node, sync, or both)" >&2
        exit 1
        ;;
esac
