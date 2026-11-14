# BeoFold Headless Validation Suite

This directory holds empirical validation artifacts generated deterministically by running the Rust integration suite in `crates/fold-msm/tests/headless_validation.rs` against the compiled `fold-physics`, `fold-storage`, `fold-proto`, and `fold-msm` workspace crates.

## Running the Validation Suite

```bash
cargo test --workspace
# or target only the headless validation integration suite:
cargo test -p fold-msm --test headless_validation
```

## Test Coverage & Generated Artifacts

1. `t1_trp_cage_baoab_stability_and_hydrophobic_collapse`: Initializes the 20-residue / 97-atom Trp-cage sequence (`NLYIQWLKDGGPSSGRPPPS`) via `ProteinTopology::build_trp_cage_seed()`, energy-minimizes (`potential_kcal`), integrates 1,800 BAOAB Langevin steps (`dt = 2.0 fs`, `300.0 K`, `friction = 1.5 ps^-1`) across 12 work-unit chunks, and verifies bounded temperature stability and hydrophobic radius-of-gyration (`Rg`) collapse.
2. `t2_mremd_four_replica_swap_and_folding_funnel`: Executes a distributed 4-replica (`285 K`, `320 K`, `360 K`, `405 K`) x 2-clone x 8-generation Metropolis Replica-Exchange (`MREMD`) campaign (64 completed work units, 72 trajectory frames) backed by `WorkUnitStore` leases/commits and parallel `rayon` integration, records Metropolis pair swap acceptance (`step_mremd_ladder`), native contact fraction (`Q`, C-alpha cutoff `8.5 A` / soft `8.8 A`), `RMSD`, `Rg`, and `SASA`, and writes [`mremd_msm_landscape.csv`](mremd_msm_landscape.csv).
3. `t3_fold_msm_reversible_matrix_and_pcca_macrostates`: Extracts trajectory frames via `extract_project_frames`, builds a 10-microstate K-means clustering (`k = 10`, lag `0.4000 ps`), estimates a reversible detailed-balance transition matrix (`MsmModel::estimate`), computes implied timescales and folding committors, coarse-grains into 3 PCCA+ macrostates, and merges all empirical metrics into [`beofold_validation_summary.json`](beofold_validation_summary.json).
