#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use fold_msm::{
    cluster_conformations_kmedoids, compute_pairwise_rmsd_matrix, estimate_reversible_msm,
    export_msm_bundle, extract_project_frames,
};
use fold_physics::{
    initialize_clone_checkpoint, minimize_steric_clashes, run_generation_chunk, ChaCha8Rng,
    LangevinConfig, ProteinTopology, SanityThresholds,
};
use fold_proto::{NodeIdentity, ProjectManifest, SignedWorkUnitResult, WorkUnitId};
use fold_storage::FoldStore;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(
    name = "fold-msm",
    version,
    about = "BeoFold pure-Rust Markov State Model (MSM) kinetic & free-energy landscape analyzer"
)]
struct MsmCli {
    #[command(subcommand)]
    command: MsmCommands,
}

#[derive(Subcommand, Debug)]
enum MsmCommands {
    /// Clusters completed PRCG checkpoints in a `FoldStore` (`redb` + `blobs/`) vault by C-alpha RMSD,
    /// estimates the reversible detailed-balance Markov transition matrix, coarse-grains PCCA+
    /// macrostates, and exports free-energy CSV/JSON/SVG landscapes + macrostate `.pdb` structures.
    Analyze {
        #[arg(long, default_value = "./beofold_data")]
        data_dir: PathBuf,

        #[arg(long, default_value_t = 1)]
        project_id: u32,

        /// Target number of C-alpha RMSD `k`-medoids microstates (`K`).
        #[arg(long, default_value_t = 8)]
        microstates: usize,

        /// Optional minimum C-alpha RMSD separation (`Angstroms`) between seeded microstate medoids.
        #[arg(long)]
        min_rmsd_separation: Option<f64>,

        /// Target number of coarse-grained PCCA+ metastable macrostates (`M`).
        #[arg(long, default_value_t = 3)]
        macrostates: usize,

        /// Markov lag time in integer `Generation` steps along each `(Run, Clone)` chain.
        #[arg(long, default_value_t = 1)]
        lag: u32,

        /// Reference thermodynamic temperature (`Kelvin`) for Boltzmann free energies `-k_B * T * ln(pi)`.
        #[arg(long, default_value_t = 300.0)]
        ref_temp: f64,

        /// Optional maximum replica bath temperature (`Kelvin`) filter when loading MREMD frames.
        #[arg(long)]
        max_temp: Option<f64>,

        /// Output directory for `msm_report.json`, CSV free-energy tables, SVG plot, and `.pdb` structures.
        #[arg(long, default_value = "./msm_analysis")]
        out_dir: PathBuf,
    },

    /// Turnkey end-to-end demonstration: folds a 4-rung MREMD Trp-cage (`1L2Y`) ensemble into a
    /// `FoldStore` vault and immediately builds the reversible MSM, free-energy landscape, and PDBs.
    DemoAnalyze {
        #[arg(long, default_value = "./beofold_msm_demo_vault")]
        data_dir: PathBuf,

        #[arg(long, default_value = "./beofold_msm_demo_out")]
        out_dir: PathBuf,

        #[arg(long, default_value_t = 350)]
        steps_per_gen: u64,

        #[arg(long, default_value_t = 4)]
        generations: u32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = MsmCli::parse();

    match cli.command {
        MsmCommands::Analyze {
            data_dir,
            project_id,
            microstates,
            min_rmsd_separation,
            macrostates,
            lag,
            ref_temp,
            max_temp,
            out_dir,
        } => {
            let store = FoldStore::open(&data_dir)?;
            run_analysis_pipeline(
                &store,
                project_id,
                microstates,
                min_rmsd_separation,
                macrostates,
                lag,
                ref_temp,
                max_temp,
                &out_dir,
            )?;
        }

        MsmCommands::DemoAnalyze {
            data_dir,
            out_dir,
            steps_per_gen,
            generations,
        } => {
            println!("=========================================================================");
            println!(" BeoFold fold-msm — End-to-End Trp-Cage MREMD + Reversible MSM Pipeline");
            println!("=========================================================================");
            let store = FoldStore::open(&data_dir)?;
            seed_and_fold_demo_vault(&store, steps_per_gen, generations)?;
            run_analysis_pipeline(&store, 1, 6, None, 3, 1, 300.0, None, &out_dir)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_analysis_pipeline(
    store: &FoldStore,
    project_id: u32,
    microstates: usize,
    min_rmsd_separation: Option<f64>,
    macrostates: usize,
    lag: u32,
    ref_temp: f64,
    max_temp: Option<f64>,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let (manifest, topology, frames) = extract_project_frames(store, project_id, max_temp)?;
    println!(
        "[MSM] Loaded {} trajectory conformations for Project P{} '{}' ({} aa, {} atoms)",
        frames.len(),
        manifest.project_id,
        manifest.name,
        manifest.residue_count,
        topology.atoms.len()
    );

    let rmsd_mat = compute_pairwise_rmsd_matrix(&topology, &frames);
    let clustering =
        cluster_conformations_kmedoids(&frames, &rmsd_mat, microstates, min_rmsd_separation);
    println!(
        "[MSM] Partitioned ensemble into K = {} C-alpha RMSD k-medoids microstates ({:.1} ms)",
        clustering.num_microstates,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Physical lag time in picoseconds (`steps_per_gen * dt_fs * 1e-3 * lag`)
    let lag_ps = (manifest.steps_per_gen as f64) * manifest.dt_fs * 1.0e-3 * (lag.max(1) as f64);
    let msm = estimate_reversible_msm(
        &frames,
        &clustering,
        lag,
        lag_ps,
        ref_temp,
        macrostates,
    )?;

    println!("-------------------------------------------------------------------------");
    println!(
        " Reversible Detailed-Balance MSM Summary (lag = {} gen = {:.3} ps)",
        msm.lag_generations, msm.lag_time_ps
    );
    println!("-------------------------------------------------------------------------");
    println!(
        " Transitions Counted      : {} across (Run, Clone) chains",
        msm.total_transitions
    );
    println!(
        " Detailed-Balance Residual: {:.3e} (max |pi_i T_ij - pi_j T_ji|)",
        msm.detailed_balance_max_error
    );
    let top_evals: Vec<String> = msm
        .eigenvalues
        .iter()
        .take(5)
        .map(|v| format!("{:.5}", v))
        .collect();
    println!(" Dominant Eigenvalues     : [{}]", top_evals.join(", "));
    let top_its: Vec<String> = msm
        .implied_timescales_ps
        .iter()
        .take(4)
        .map(|v| format!("{:.2} ps", v))
        .collect();
    println!(" Implied Timescales (ITS) : [{}]", top_its.join(", "));
    println!("-------------------------------------------------------------------------");
    println!(" PCCA+ Metastable Macrostates ({}):", msm.macrostates.len());
    for mac in &msm.macrostates {
        let rep = &frames[mac.representative_frame_idx];
        println!(
            "   M{} [{:<17}] | pi = {:>6.2}% | dG = {:>5.2} kcal/mol | p_fold = {:.2} | Rg = {:.2} A | Rep = {}",
            mac.macro_id,
            mac.biophysical_label,
            mac.stationary_population * 100.0,
            mac.free_energy_kcal,
            mac.mean_committor,
            mac.mean_rg_angstrom,
            rep.wu_id
        );
    }

    let exported = export_msm_bundle(out_dir, &manifest, &topology, &frames, &clustering, &msm)?;
    println!("-------------------------------------------------------------------------");
    println!(" Exported MSM Landscape & Structural Bundle:");
    println!("   JSON Report      : {}", exported.json_report_path.display());
    println!("   Microstate CSV   : {}", exported.microstate_csv_path.display());
    println!("   2D PMF Grid CSV  : {}", exported.pmf_grid_csv_path.display());
    println!("   Landscape SVG    : {}", exported.svg_landscape_path.display());
    for pdb in &exported.macrostate_pdb_paths {
        println!("   Macrostate PDB   : {}", pdb.display());
    }
    println!("=========================================================================");

    Ok(())
}

fn seed_and_fold_demo_vault(
    store: &FoldStore,
    steps_per_gen: u64,
    generations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if store.list_project_ids()?.is_empty() {
        let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
        minimize_steric_clashes(&topo, &mut coords, 120);
        let ladder = vec![285.0, 320.0, 360.0, 400.0];
        let clones = 2u32;
        let mut gen0 = Vec::new();
        for (r, &temp) in ladder.iter().enumerate() {
            for c in 0..clones {
                gen0.push(initialize_clone_checkpoint(
                    &topo,
                    &coords,
                    WorkUnitId::new(1, r as u32, c, 0),
                    temp,
                    b"msm-demo-seed",
                ));
            }
        }
        let manifest = ProjectManifest {
            project_id: 1,
            name: topo.name.clone(),
            sequence: topo.sequence_one_letter.clone(),
            topology_blake3: [0u8; 32],
            initial_coords_blake3: [0u8; 32],
            atom_count: topo.atoms.len() as u32,
            residue_count: topo.residue_count,
            temp_ladder_kelvin: ladder,
            dt_fs: 1.5,
            friction_per_ps: 1.0,
            steps_per_gen: steps_per_gen.max(50),
            num_clones_per_run: clones,
            max_generations: generations.max(2),
        };
        store.seed_project(manifest, &topo, &gen0)?;
    }

    let (manifest, topo) = store.load_project(1)?;
    let identity = NodeIdentity::from_cluster_secret("msm-demo-secret", "msm-worker");
    let mut mremd_rng = ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 55, 55, 0), b"msm-demo");
    let thresholds = SanityThresholds::default();

    loop {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let Some((lease, input_cp)) =
            store.acquire_next_lease(identity.public_key_bytes(), now, 600)?
        else {
            break;
        };
        let cfg = LangevinConfig {
            dt_fs: manifest.dt_fs,
            friction_per_ps: manifest.friction_per_ps,
            temperature_kelvin: input_cp.temperature_kelvin,
            steps: lease.target_steps,
            sample_stride: (lease.target_steps / 10).max(1),
        };
        let t0 = Instant::now();
        let summary = run_generation_chunk(&topo, &input_cp, lease.wu_id, &cfg);
        let out_frame = summary.checkpoint.to_compressed_frame()?;
        let out_hash = *blake3::hash(&out_frame).as_bytes();

        let mut receipt = SignedWorkUnitResult {
            wu_id: lease.wu_id,
            lease_token: lease.lease_token,
            worker_pubkey: [0u8; 32],
            input_checkpoint_blake3: lease.input_checkpoint_blake3,
            output_checkpoint_blake3: out_hash,
            steps_executed: cfg.steps,
            temperature_kelvin: cfg.temperature_kelvin,
            mean_potential_kcal: summary.mean_potential_kcal,
            min_potential_kcal: summary.min_potential_kcal,
            final_potential_kcal: summary.checkpoint.potential_energy_kcal,
            final_kinetic_kcal: summary.checkpoint.kinetic_energy_kcal,
            final_rg_angstrom: summary.checkpoint.rg_angstrom,
            final_sasa_angstrom2: summary.checkpoint.sasa_angstrom2,
            wall_time_ms: t0.elapsed().as_millis() as u64,
            signature: [0u8; 64],
        };
        identity.sign_result(&mut receipt);
        store.commit_signed_result(
            &manifest,
            &topo,
            &receipt,
            &summary.checkpoint,
            &thresholds,
            &mut mremd_rng,
            now,
        )?;
    }

    Ok(())
}
