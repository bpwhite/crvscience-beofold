# BeoFold (`v1.4.0`) — User Guide

BeoFold is a pure-Rust distributed molecular dynamics, protein folding, and Markov State Model (`MSM`) kinetic analysis workbench licensed under the **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)**.

---

## 1. Workspace Setup & Compilation

### Prerequisites
- Rust toolchain (`rustc` / `cargo` 1.80+ recommended, edition 2021)
- Optional GPU compute backend: Vulkan (Linux/Windows), DirectX 12 (Windows), or Metal (macOS) via `wgpu 30`. When no GPU adapter is present, BeoFold transparently uses parallel `rayon` CPU evaluation.

### Build & Test Commands
```bash
# Build all 7 crates and binaries in release mode
cargo build --workspace --release

# Run the full workspace unit & integration test suite
cargo test --workspace

# Check all targets with zero warnings
cargo check --workspace --all-targets
```

---

## 2. Desktop Supervisor Workstation (`fold-gui`)

Launch the 4-tab `eframe`/`egui` desktop workstation:

```bash
cargo run --release -p fold-gui -- \
  --lan-peer 127.0.0.1:7443 \
  --cluster-secret "my-home-beowulf-key" \
  --sync-url "https://yourdomain.com/beofold/api/v1" \
  --data-dir ./beofold_data
```

### `fold-gui` CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--lan-peer <IP:PORT>` | `127.0.0.1:7443` | Initial LAN Beowulf QUIC peer socket address. |
| `--cluster-secret <SECRET>` | `beofold-home-cluster` | Shared secret authenticating LAN QUIC TLS 1.3 RPC frames. |
| `--sync-url <URL>` | `""` | Optional Apache HTTP Web Sync API base URL (e.g., `https://yourdomain.com/beofold/api/v1`). |
| `--data-dir <PATH>` | `./beofold_data` | Local `redb` + `blobs/` vault path for loading offline campaigns and MSMs. |
| `--no-gpu` | `false` | Disable `wgpu` WGSL compute shaders in the built-in live folding studio. |
| `--screenshot <OUT_PNG>` | `None` | Render a single headless `1600x1000` RGBA PNG screenshot and exit. |
| `--screenshot-view <1\|2\|3>` (alias `--view`) | `1` | Headless screenshot preset (`1` = 3D Studio, `2` = MSM Surface, `3` = WAN Federation). |

### Global Hotkeys & Mouse Controls

All global keyboard shortcuts are guarded by `!ctx.wants_keyboard_input()`, so typing into any text field (`LAN Peer`, `Secret`, `Apache Sync URL`, custom sequences, or file paths) never triggers shortcuts accidentally.

| Input | Action |
| :--- | :--- |
| `Space` | Play / Pause local 4-rung MREMD simulation |
| `R` | Reset 3D camera (`yaw`, `pitch`, `zoom`, `pan`) in active 3D viewport |
| `S` | Toggle 3D viewport `Auto-Spin` |
| `1` | Switch to Tab 1: **3D Conformation & MREMD Studio** |
| `2` | Switch to Tab 2: **LAN Beowulf Matrix** |
| `3` | Switch to Tab 3: **Apache WAN Federation** |
| `4` | Switch to Tab 4: **MSM Free-Energy & Flux** |
| `F1` | Toggle the **About BeoFold** modal |
| `Esc` | Dismiss the **About BeoFold** modal |
| Primary Mouse Drag | Orbit 3D protein conformation |
| Secondary / Middle Drag | Pan 3D camera |
| Scroll Wheel | Zoom 3D camera (`0.35x` to `4.50x`) |

### Expandable `Cluster Connection` Bar

At the top of the workstation, raw network inputs are tucked inside a compact toggleable **`▸ Cluster Connection`** bar (collapsed by default to maximize 3D and 2D workspace height):
- Displays current peer/WAN summary when collapsed.
- Expands to edit **`LAN Peer`**, **`Secret`**, and **`Apache Sync URL`**, trigger **`Connect / Refresh`**, or run **`Discover LAN`** (`_beofold._udp.local.` mDNS + localhost `7443..7446` sweep).
- Performs live `SocketAddr` validation on **`LAN Peer`**: entering a malformed socket address (missing port, invalid IP, out-of-range port) highlights the field and displays an inline red warning banner both inside the bar and on the collapsed header without panicking or failing silently.

### The 4 Workstation Tabs

1. **Tab 1 — `3D Conformation & MREMD Studio`**:
   - Fold built-in presets (**Trp-cage 1L2Y**, 20 aa; **Villin Headpiece 1YRF**, 35 aa), import `.pdb`/`.cif` crystallographic files via `pdbtbx`, or enter any 1-letter / FASTA amino-acid sequence.
   - Interactive 3D viewport renders a smooth **Catmull-Rom cubic spline backbone ribbon** colored by local secondary structure (**Rose**: α-helix, **Sky**: β-sheet, **Slate**: coil), biophysical sidechain beads (**Gold**: hydrophobic, **Sky**: basic `+K/R`, **Coral**: acidic `-D/E`, **Teal**: polar), a depth-sorted **XYZ orientation triad**, and a dynamic **Ångström (`Å`) scale bar**.
   - Switch styles (`Backbone + Sidechain Beads`, `C-Alpha Trace`, `All Heavy Atoms`), enable **Side-by-Side: Unfolded Start vs. Folded End**, or export the lowest-energy `.pdb`.
   - Right panel renders real-time time-series traces (**Potential Energy**, **`R_g`**, **`C-alpha RMSD`**, **Native Contacts `Q(t)`**) with Y/X axis tick labels, Series/Min legends, and hover readouts.
2. **Tab 2 — `LAN Beowulf Matrix`**:
   - Displays elected Raft term and leader, peer hardware compute tiers (`Tier S`, `Tier A`, `Tier B`), live Villin `ns/day` throughput, active PRCG work units, CPU load/throttling, MREMD swaps, and quarantine counts.
3. **Tab 3 — `Apache WAN Federation`**:
   - Displays global federation throughput, quorum mode (`q = 1 | 2`), canary audit pass counts, global PRCG progress bar, and per-cluster leaderboards.
4. **Tab 4 — `MSM Free-Energy & Flux`**:
   - Configure **`Microstates (K)`**, **`PCCA+ Macrostates (M)`**, **`Lag (Generations)`**, **`Ref Temp (K)`**, **`Cold-Rung Filter`**, and **`Min Flux`** (each with detailed hover tooltips).
   - Click **`Sample 24 MREMD Frames (Burst)`** followed by **`Compute MSM from Live Studio`**, or click **`Build from redb`** to analyze a disk vault.
   - Center canvas draws a **`40x40` Silverman/Scott adaptive-KDE 2D `(RMSD, R_g)` free-energy surface** overlaid with **`1, 2, 3 kcal/mol` iso-energy contour lines**, directed equilibrium Markov flux edges, and clickable microstates colored by folding committor `p_fold`.
   - Right inspector is wrapped in a vertical `ScrollArea` (safe on compact viewports) housing a 3D macrostate/microstate previewer, `.pdb` exporter, **Open in Full 3D Studio** jump button, and implied relaxation timescale bars.

---

## 3. Headless Compute Node & LAN Cluster (`fold-node`)

Global flags accepted by all `fold-node` subcommands:
- `--threads <N>`: Limit `rayon` worker threads (defaults to all logical cores).
- `--no-gpu`: Disable `wgpu` compute shaders and use `rayon` CPU Morton neighbor lists.

### Subcommands

| Command | Key Flags & Defaults | Purpose |
| :--- | :--- | :--- |
| `fold-node bench` | `--steps 1500` | Benchmark 35-residue Villin Headpiece throughput (`ns/day`) and assign compute tier (`TierS`, `TierA`, `TierB`). |
| `fold-node init` | `--data-dir ./beofold_data`, `--project-id 1`, `--pdb <FILE>`, `--runs 4`, `--clones 2`, `--generations 4`, `--steps-per-gen 1000` | Seed a PRCG + MREMD campaign into a `redb` + `blobs/` store (defaults to built-in Trp-cage `1L2Y`). |
| `fold-node run-local` | `--data-dir ./beofold_data`, `--project-id 1`, `--max-units 0`, `--cluster-secret beofold-local-cluster`, `--export-best <PDB>` | Process queued PRCG work units locally in single-process mode (`0` = all queued units). |
| `fold-node mesh` | `--data-dir ./beofold_node`, `--bind 0.0.0.0:7443`, `--advertise-ip <IPv4>`, `--cluster-secret beofold-home-cluster`, `--seed-peer <ADDR>...`, `--no-subnet-sweep`, `--auto-throttle`, `--max-cpu 80.0`, `--sync-url <URL>`, `--cluster-name Home-Beowulf-01`, `--init-trpcage`, `--label beowulf-node` | Join/boot a live Beowulf LAN node with mDNS + `/24` fallback discovery, Ongaro-Ousterhout Raft leader election, QUIC TLS 1.3 RPC, optional CPU thermal throttling, and outbound Apache WAN gateway sync. |
| `fold-node cluster-status` | `--peer 127.0.0.1:7443`, `--cluster-secret beofold-home-cluster` | Query live Raft term, leader, peer list, PRCG progress, and MREMD swaps over QUIC. |
| `fold-node simulate-cluster` | `--data-dir ./beofold_cluster_sim`, `--nodes 3`, `--generations 3`, `--steps-per-gen 300` | Multi-node QUIC + Raft cluster chaos test (kills active Leader mid-run to verify failover). |
| `fold-node simulate-federation` | `--data-dir ./beofold_federation_sim`, `--quorum 1`, `--generations 2`, `--steps-per-gen 250` | Spawns an HTTP sync server + two multi-node LAN clusters to validate WAN federation. |
| `fold-node status` | `--data-dir ./beofold_data`, `--project-id 1` | Print offline `redb` PRCG completion table, MREMD swap stats, and best conformation. |
| `fold-node demo` | `--data-dir ./beofold_demo`, `--out-pdb ./trp_cage_best.pdb`, `--steps-per-gen 600` | Run a turnkey standalone 4-rung Trp-cage MREMD demo and export `.pdb`. |

### Unattended Boot Services (`deploy/`)
- **Linux (`systemd`)**: `sudo ./deploy/systemd/install-linux-systemd.sh --role node --cluster-secret "<secret>" --open-firewall`
- **Windows Service / Scheduled Task**: `.\deploy\windows\Install-BeoFoldNode.ps1 -ClusterSecret "<secret>" -AutoThrottle`

---

## 4. Apache HTTP Federation Server (`fold-sync`)

`fold-sync` provides both a persistent `axum` reverse-proxy server and zero-daemon RFC-3875 CGI execution:

| Command | Key Flags & Defaults | Purpose |
| :--- | :--- | :--- |
| `fold-sync init` | `--data-dir ./beofold_sync_data`, `--project-id 1`, `--pdb <FILE>`, `--runs 4`, `--clones 2`, `--generations 4`, `--steps-per-gen 1000` | Seed a campaign on the web sync vault. |
| `fold-sync serve` | `--bind 127.0.0.1:8780`, `--data-dir ./beofold_sync_data`, `--quorum 1`, `--strict-trust`, `--trust-cluster <HEX>...`, `--init-trpcage` | Run Mode A (`axum` HTTP server behind Apache `mod_proxy_http`). Exposes `/beofold/dashboard`, `/beofold/api/v1`, and `/beofold/api/v1/public/conformation/best.pdb`. |
| `fold-sync cgi` | `--data-dir /var/lib/beofold`, `--quorum 1` | Run Mode B (single-shot Apache `mod_cgid` CGI request; also auto-detected when `GATEWAY_INTERFACE` is set with zero CLI arguments, configured via `BEOFOLD_DATA_DIR` and `BEOFOLD_QUORUM`). |
| `fold-sync status` | `--data-dir ./beofold_sync_data`, `--export-best <PDB>` | Print global federation telemetry JSON and optionally export the lowest-energy `.pdb`. |

---

## 5. Headless Markov State Model CLI (`fold-msm`)

```bash
# End-to-end Trp-cage MREMD + MSM demo
cargo run --release -p fold-msm -- demo-analyze \
  --data-dir ./beofold_msm_demo_vault \
  --out-dir ./msm_demo_out \
  --steps-per-gen 350 \
  --generations 4

# Analyze an existing FoldStore vault
cargo run --release -p fold-msm -- analyze \
  --data-dir ./beofold_data \
  --project-id 1 \
  --microstates 8 \
  --macrostates 3 \
  --lag 1 \
  --ref-temp 300.0 \
  --out-dir ./msm_analysis
```

### `fold-msm analyze` Flags
- `--data-dir <PATH>` (default `./beofold_data`)
- `--project-id <ID>` (default `1`)
- `--microstates <K>` (default `8`)
- `--min-rmsd-separation <ANG>` (optional minimum C-alpha RMSD spacing for regular-space seeding)
- `--macrostates <M>` (default `3`)
- `--lag <GEN>` (default `1`)
- `--ref-temp <KELVIN>` (default `300.0`)
- `--max-temp <KELVIN>` (optional upper replica temperature cutoff; defaults to cold-rung filter `T_ref + 25 K` with automatic fallback)
- `--out-dir <PATH>` (default `./msm_analysis`)

### Exported Bundle Files
Every analysis writes:
1. `msm_report.json` — Eigenvalues, implied timescales, stationary `pi`, `T_ij`, detailed-balance residual, and PCCA+ macrostate metadata.
2. `free_energy_microstates.csv` — Per-microstate population, relative free energy, `p_fold`, `R_g`, `RMSD_best`, and cluster radius.
3. `free_energy_surface_2d.csv` — `40x40` `(RMSD, R_g)` equilibrium PMF grid (`kcal/mol`).
4. `free_energy_landscape.svg` — Publication-ready 2D PMF heatmap with `1, 2, 3 kcal/mol` Marching Squares iso-energy contours, flux network edges, implied timescales, and macrostate summary.
5. `macrostate_<id>_<label>.pdb` — Representative crystallographic `.pdb` snapshots for every PCCA+ macrostate.
