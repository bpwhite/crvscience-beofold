# BeoFold (`v1.4.0`) — Pure-Rust Distributed Beowulf & Multi-Cluster Protein Folding Engine

![BeoFold Headless Workbench](validation/headless_screenshot.png)

BeoFold is a 100% pure-Rust distributed molecular dynamics, protein structure folding, and Markov State Model (`MSM`) kinetic analysis platform built for heterogeneous Windows and Linux home Beowulf clusters, desktop supervisor visualization, and Apache HTTP multi-cluster WAN federation (`GPL-3.0-or-later`).

## Scientific Overview

- **Implicit-Solvent Atomistic / United-Atom Force Field (`fold-physics`)**: Combines harmonic covalent bonds and valence angles, singularity-free Blondel-Karplus torsions, `C^1` force-shifted 12-6 Lennard-Jones, Still (1990) Generalized-Born screened electrostatics using per-atom effective Born radii (`gb_radius_ang`), and Lazaridis-Karplus EEF1 signed Gaussian solvation (favorable hydrophobic burial vs. polar/charged desolvation penalties). Evaluates across parallel `rayon` 3D Morton Z-order CPU neighbor grids and cross-vendor `wgpu` WGSL compute shaders (`nonbonded.wgsl`) with `< 1e-4` CPU/GPU parity.
- **Thermostatted Sampling & MREMD**: Symmetric Leimkuhler-Matthews `BAOAB` Langevin integration paired with Multiplexed-Replica Exchange Molecular Dynamics ([MREMD](https://pmc.ncbi.nlm.nih.gov/articles/PMC1302658/)) featuring non-positive/non-finite temperature guards and Sugita-Okamoto (1999) canonical velocity rescaling (`v' = v * sqrt(T_new / T_old)`) on accepted swaps.
- **Reversible Kinetic MSMs & Free-Energy Landscapes (`fold-msm`, `#![forbid(unsafe_code)]`)**: Parallel Horn/Kabsch `4x4` quaternion C-alpha RMSD `k`-medoids clustering, Tarjan SCC ergodic pruning, Prinz-Noé reversible detailed-balance MLE, Cyclic Jacobi eigendecomposition, Transition Path Theory (`p_fold`), Röblitz-Weber PCCA+ macrostate coarse-graining, and `40x40` Silverman/Scott adaptive-KDE `(RMSD, R_g)` PMF surfaces with `1, 2, 3 kcal/mol` Marching Squares iso-energy contours.

## Workspace Crate Architecture

| Crate | Binary / Library | Responsibility |
| :--- | :--- | :--- |
| `crates/fold-proto` | Library | [Folding@home `PRCG`](https://foldingathome.org/science/folding) (`Project, Run, Clone, Generation`) coordinates, [`rkyv`](https://crates.io/crates/rkyv) zero-copy frames, [`lz4_flex`](https://crates.io/crates/lz4_flex) compression, [`blake3`](https://crates.io/crates/blake3) LAN MACs, and [`ed25519-dalek`](https://crates.io/crates/ed25519-dalek) work-unit & HTTP request signing. |
| `crates/fold-physics` | Library | [`pdbtbx`](https://docs.rs/pdbtbx) PDB/mmCIF parser, harmonic covalent bonds/angles, singularity-free Blondel-Karplus torsions, `C^1` force-shifted 12-6 LJ + Still Generalized-Born electrostatics + signed EEF1 solvation, 3D Morton Z-order [`rayon`](https://crates.io/crates/rayon) CPU neighbor lists, cross-vendor [`wgpu`](https://docs.rs/crate/wgpu/latest) WGSL compute shaders (`nonbonded.wgsl`), symmetric Langevin `BAOAB` integrator, MREMD Metropolis swapper with Sugita-Okamoto rescaling, and 4-gate biophysical sanity verification. |
| `crates/fold-storage` | Library | Crash-safe [`redb`](https://crates.io/crates/redb) ACID MVCC ledger + atomic content-addressed `blobs/<2-hex>/<blake3>.lz4` store, Raft follower replication, `quorum = 1 \| 2` cross-cluster consensus, and deterministic 20-step canary spot-checks. |
| `crates/fold-node` | `fold-node` (CLI) | Headless Windows/Linux compute worker & Beowulf LAN cluster node featuring [`mdns-sd`](https://crates.io/crates/mdns-sd) `_beofold._udp.local.` auto-discovery, `/24` unicast firewall fallback, Ongaro-Ousterhout Raft leader election, [`quinn`](https://crates.io/crates/quinn) + [`rustls`](https://crates.io/crates/rustls) TLS 1.3 transport, `SIGINT`/`SIGTERM` graceful checkpoint flushing, `--auto-throttle`, and [`reqwest`](https://crates.io/crates/reqwest) outbound Apache WAN synchronization. |
| `crates/fold-sync` | `fold-sync` (Web) | Dual-mode web synchronizing server supporting persistent [`axum`](https://crates.io/crates/axum) daemon mode behind [Apache `mod_proxy_http`](https://httpd.apache.org/docs/current/mod/mod_proxy_http.html) and zero-daemon RFC-3875 [Apache `mod_cgid`](https://httpd.apache.org/docs/current/mod/mod_cgid.html) CGI execution (`GATEWAY_INTERFACE=CGI/1.1` auto-detected), plus a read-only public HTML5 Canvas 3D conformation viewer & multi-cluster leaderboard at `/beofold/dashboard`. |
| `crates/fold-gui` | `fold-gui` (GUI) | 4-tab Desktop Supervisor Workstation built on [`eframe`](https://crates.io/crates/eframe) and [`egui`](https://crates.io/crates/egui): (1) Live 3D Conformation (Catmull-Rom secondary-structure ribbons, XYZ triad, `Å` scale bar) & MREMD Studio, (2) LAN Beowulf Cluster Matrix, (3) Global Apache WAN Federation, and (4) Interactive `fold-msm` `40x40` 2D `(RMSD, R_g)` Free-Energy Surface (`1/2/3 kcal/mol` contours) & Markov Flux Network. |
| `crates/fold-msm` | `fold-msm` (CLI + Lib) | `#![forbid(unsafe_code)]` Markov State Model engine implementing parallel Horn/Kabsch C-alpha RMSD `k`-medoids clustering, [Prinz-Noé reversible detailed-balance MLE](https://pubs.aip.org/aip/jcp/article/134/17/174105/699460/Markov-models-of-molecular-kinetics-Generation-and), pure-Rust Cyclic Jacobi eigendecomposition, Transition Path Theory (`p_fold`), [PCCA+ metastable macrostate coarse-graining](https://www.mdpi.com/2079-3197/6/1/20), `40x40` adaptive-KDE `(RMSD, R_g)` landscapes (`SVG`/`CSV`/`JSON`), and representative macrostate `.pdb` export. |

## Workstation Controls & Global Hotkeys (`fold-gui`)

All global keyboard shortcuts are guarded by `!ctx.wants_keyboard_input()` so typing in text fields never triggers actions.

| Control / Key | Scope | Action |
| :--- | :--- | :--- |
| `Space` | Global | Play / Pause local 4-rung MREMD folding runner |
| `R` | Global | Reset 3D camera (`yaw`, `pitch`, `zoom`, `pan`) in active 3D viewport |
| `S` | Global | Toggle 3D viewport `Auto-Spin` |
| `1` – `4` | Global | Switch workspace tabs (`1` Studio, `2` LAN Matrix, `3` WAN Federation, `4` MSM Surface) |
| `Ctrl+K` / `Cmd+K` | Global | Open fuzzy **Command Palette** modal (switch views, compute/export MSM & ITS/CK diagnostics, discover LAN peers, reset camera) |
| `F1` / `Esc` | Global | Toggle / Dismiss the **About BeoFold** dialog |
| Primary Drag | 3D Viewport | Orbit protein conformation |
| Secondary / Middle Drag | 3D Viewport | Pan 3D camera |
| Scroll Wheel | 3D Viewport | Zoom (`0.35x` – `4.50x`) |
| `▸ Cluster Connection` | Top Bar | Expand/collapse LAN QUIC peer (`SocketAddr`-validated), cluster secret, Apache sync URL, and one-click mDNS `Discover LAN` |

## CLI Reference Across Binaries

| Binary | Subcommands / Primary Flags |
| :--- | :--- |
| `fold-node` | Global: `--threads <N>`, `--no-gpu`<br>Subcommands: `bench [--steps 1500]`, `init [--data-dir ./beofold_data] [--project-id 1] [--pdb <PATH>] [--runs 4] [--clones 2] [--generations 4] [--steps-per-gen 1000]`, `run-local [--max-units 0] [--cluster-secret ...] [--export-best <PDB>]`, `mesh [--bind 0.0.0.0:7443] [--advertise-ip <IP>] [--cluster-secret ...] [--seed-peer <ADDR>] [--no-subnet-sweep] [--auto-throttle] [--max-cpu 80.0] [--sync-url <URL>] [--cluster-name ...] [--init-trpcage] [--label ...]`, `cluster-status [--peer 127.0.0.1:7443] [--cluster-secret ...]`, `simulate-cluster [--nodes 3] [--generations 3] [--steps-per-gen 300]`, `simulate-federation [--quorum 1] [--generations 2] [--steps-per-gen 250]`, `status [--data-dir ...] [--project-id 1]`, `demo [--data-dir ./beofold_demo] [--out-pdb ./trp_cage_best.pdb] [--steps-per-gen 600]` |
| `fold-sync` | `init [--data-dir ./beofold_sync_data] [--project-id 1] [--pdb <PATH>] [--runs 4] [--clones 2] [--generations 4] [--steps-per-gen 1000]`, `serve [--bind 127.0.0.1:8780] [--data-dir ./beofold_sync_data] [--quorum 1] [--strict-trust] [--trust-cluster <HEX>...] [--init-trpcage]`, `cgi [--data-dir /var/lib/beofold] [--quorum 1]` (or `GATEWAY_INTERFACE` auto-detect), `status [--data-dir ./beofold_sync_data] [--export-best <PDB>]` |
| `fold-gui` | `--lan-peer 127.0.0.1:7443`, `--cluster-secret beofold-home-cluster`, `--sync-url ""`, `--data-dir ./beofold_data`, `--no-gpu`, `--screenshot <OUT_PNG>`, `--screenshot-view <1\|2\|3>` (`--view`) |
| `fold-msm` | `analyze [--data-dir ./beofold_data] [--project-id 1] [--microstates 8] [--min-rmsd-separation <ANG>] [--macrostates 3] [--lag 1] [--ref-temp 300.0] [--max-temp <KELVIN>] [--out-dir ./msm_analysis]`, `demo-analyze [--data-dir ./beofold_msm_demo_vault] [--out-dir ./beofold_msm_demo_out] [--steps-per-gen 350] [--generations 4]` |

## Quick-Start Commands

### 1. Build & Run Physics / Cryptography / GPU / MSM / Sync Tests
```bash
cargo build --workspace --release
cargo test --workspace --release
```

### 2. Launch `fold-sync` & Open the Public Read-Only Browser Observatory
```bash
cargo run --release -p fold-sync -- serve --bind 127.0.0.1:8780 --init-trpcage
```
Open `http://127.0.0.1:8780/beofold/dashboard` (or `https://yourdomain.com/beofold/dashboard` behind [Apache `mod_proxy_http`](https://httpd.apache.org/docs/current/mod/mod_proxy_http.html) or [Apache `mod_cgid`](https://httpd.apache.org/docs/current/mod/mod_cgid.html)) to inspect live federated Beowulf cluster leaderboards, PRCG/MREMD energetics, the interactive HTML5 Canvas 3D conformation viewer, and one-click crystallographic `.pdb` downloads (`/beofold/api/v1/public/conformation/best.pdb`). All write and lease endpoints remain strictly protected by `Ed25519` request authentication.

### 3. Launch the 4-Tab Supervisor Workstation (`fold-gui`)
```bash
cargo run --release -p fold-gui -- \
  --lan-peer 127.0.0.1:7443 \
  --cluster-secret "my-home-beowulf-key" \
  --sync-url "https://yourdomain.com/beofold/api/v1"
```

### 4. Headless Markov State Model CLI (`fold-msm`)
```bash
cargo run --release -p fold-msm -- demo-analyze --out-dir ./msm_demo_out
```

### 5. Unattended Boot Service Installation (`deploy/systemd/` & `deploy/windows/`)
```bash
sudo ./deploy/systemd/install-linux-systemd.sh --role node --cluster-secret "my-home-beowulf-key" --open-firewall
```
```powershell
.\deploy\windows\Install-BeoFoldNode.ps1 -ClusterSecret "my-home-beowulf-key" -AutoThrottle
```

## Empirical Validation Metrics (`validation/beofold_validation_summary.json`)

Generated deterministically by `cargo test -p fold-msm --test headless_validation` (`TrpCage_1L2Y`, 20 residues / 97 atoms):

| Validation Gate / Metric | Observed Value |
| :--- | :--- |
| `t1` BAOAB Collapse (`1,800` steps @ `300.0 K`, `dt = 2.0 fs`) | `R_g`: `18.8132 Å` -> `18.4367 Å` (`min 18.2203 Å`), `T_kin` mean `325.71 K` |
| `t2` 4-Rung MREMD (`[285, 320, 360, 405 K]`, `64` WUs, `72` frames) | Pair swap acceptance `21/54` (`mean P_acc = 0.4085`), `store_mremd_swaps = 8/32`, `Q in [0.6000, 1.0000]` |
| `t3` Reversible MSM (`lag = 0.4000 ps`, `K = 6` active states, `M = 3` PCCA+) | Detailed-balance error `2.65e-23`, row-stochastic error `0.00e0`, stationary sum `1.000000` |
| `t3` 2D PMF Surface & Contours | `40x40` adaptive-KDE grid with `1, 2, 3 kcal/mol` iso-contours |
| `fold-msm` ITS & Chapman-Kolmogorov | Multi-lag Implied Timescales (`ITS`) & macrostate Chapman-Kolmogorov (`CK`) diagnostic digest exported to `msm_diagnostics.json` & `msm_diagnostics.csv` via CLI (`fold-msm analyze` / `demo-analyze`) and GUI (`Export ITS/CK`) |
| Workspace Release Test Suite | `55 passed; 0 failed` across all 7 workspace crates (`fold-gui` 16, `fold-msm` 7 + 3 validation, `fold-node` 1, `fold-physics` 17, `fold-proto` 6, `fold-storage` 2, `fold-sync` 3) with `0` non-test `.unwrap()` calls |

## Documentation & Guides

- **User Guide**: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- **Scientific & Mathematical Methods**: [`docs/SCIENTIFIC_METHODS.md`](docs/SCIENTIFIC_METHODS.md)
- **Validation Suite & Telemetry**: [`validation/README.md`](validation/README.md) (`cargo test --workspace`)

## Citation & CRV Science Apps Hub

- **CRV Science Apps Hub**: <https://crvscience.com/apps>
- **Companion Preprint (`APP-PREPRINT-02`)**: [BeoFold: Distributed Langevin Molecular Dynamics, Replica-Exchange Sampling, and Markov State Kinetic Modeling](https://crvscience.com/apps/preprints/beofold-distributed-protein-msm)
- **Zenodo Concept DOI**: [10.5281/zenodo.24222370](https://doi.org/10.5281/zenodo.24222370)
- **Machine-Readable Metadata**: [`CITATION.cff`](CITATION.cff)

```bibtex
@article{White2026BeoFold,
  author  = {White, Bryan P.},
  title   = {BeoFold: Simulating How Proteins Fold Using Distributed Computing and State Networks},
  journal = {CRV Science Apps Preprints},
  year    = {2026},
  doi     = {10.5281/zenodo.24222370},
  url     = {https://crvscience.com/apps/preprints/beofold-distributed-protein-msm},
  note    = {Software v1.4.0; Preprint ID: APP-PREPRINT-02}
}
```
