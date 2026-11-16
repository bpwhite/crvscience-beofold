// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

mod discovery;
mod mesh;
mod quic_net;
mod raft;
mod throttle;
mod wan_sync;

use clap::{Parser, Subcommand};
use discovery::DEFAULT_BEOWULF_PORT;
use fold_physics::{
    benchmark_local_node_with_gpu, compute_ca_rmsd, initialize_clone_checkpoint,
    minimize_steric_clashes, probe_gpu_adapter, run_generation_chunk_with_gpu, ChaCha8Rng,
    GpuForceEngine, LangevinConfig, ProteinTopology, SanityThresholds,
};
use fold_proto::{
    NodeIdentity, ProjectManifest, SignedWorkUnitResult, WorkUnitId, WorkUnitStatus,
};
use fold_storage::{to_hex, CommitOutcome, FoldStore};
use mesh::{
    query_remote_cluster_status, run_multi_node_failover_simulation, start_mesh_node,
    MeshNodeConfig,
};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;
use wan_sync::run_two_cluster_wan_federation_simulation;

#[derive(Parser, Debug)]
#[command(
    name = "fold-node",
    version,
    about = "BeoFold headless Beowulf LAN compute node & Apache HTTP multi-cluster federation CLI"
)]
struct Cli {
    /// Optional explicit CPU worker thread limit for `rayon` (defaults to all logical cores).
    #[arg(long, global = true)]
    threads: Option<usize>,

    /// Disable `wgpu` GPU compute shaders and force `rayon` CPU Morton-grid evaluation.
    #[arg(long, global = true, default_value_t = false)]
    no_gpu: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Benchmarks local Windows/Linux CPU & `wgpu` GPU throughput on the 35-residue Villin Headpiece reference.
    Bench {
        #[arg(long, default_value_t = 1500)]
        steps: u64,
    },

    /// Initializes a new PRCG + MREMD folding project in `redb` (from a PDB/mmCIF file or built-in Trp-cage).
    Init {
        #[arg(long, default_value = "./beofold_data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        project_id: u32,
        #[arg(long)]
        pdb: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        runs: u32,
        #[arg(long, default_value_t = 2)]
        clones: u32,
        #[arg(long, default_value_t = 4)]
        generations: u32,
        #[arg(long, default_value_t = 1000)]
        steps_per_gen: u64,
    },

    /// Processes queued PRCG Work Units locally in single-process mode (`wgpu` GPU + `rayon` CPU).
    RunLocal {
        #[arg(long, default_value = "./beofold_data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        project_id: u32,
        #[arg(long, default_value_t = 0)]
        max_units: usize,
        #[arg(long, default_value = "beofold-local-cluster")]
        cluster_secret: String,
        #[arg(long)]
        export_best: Option<PathBuf>,
    },

    /// Boots a live Beowulf LAN node (mDNS + `/24` fallback + Raft leader election + QUIC TLS 1.3
    /// + `wgpu` GPU compute + optional `--sync-url` outbound Apache HTTPS multi-cluster synchronization).
    Mesh {
        #[arg(long, default_value = "./beofold_node")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = SocketAddr::from(([0, 0, 0, 0], DEFAULT_BEOWULF_PORT)))]
        bind: SocketAddr,
        #[arg(long)]
        advertise_ip: Option<Ipv4Addr>,
        #[arg(long, default_value = "beofold-home-cluster")]
        cluster_secret: String,
        #[arg(long)]
        seed_peer: Vec<SocketAddr>,
        #[arg(long, default_value_t = false)]
        no_subnet_sweep: bool,
        #[arg(long, default_value_t = false)]
        auto_throttle: bool,
        #[arg(long, default_value_t = 80.0)]
        max_cpu: f32,
        #[arg(long)]
        sync_url: Option<String>,
        #[arg(long, default_value = "Home-Beowulf-01")]
        cluster_name: String,
        #[arg(long, default_value_t = false)]
        init_trpcage: bool,
        #[arg(long, default_value = "beowulf-node")]
        label: String,
    },

    /// Queries live Raft leadership, peer telemetry, and PRCG progress from a running LAN Beowulf node.
    ClusterStatus {
        #[arg(long, default_value_t = SocketAddr::from(([127, 0, 0, 1], DEFAULT_BEOWULF_PORT)))]
        peer: SocketAddr,
        #[arg(long, default_value = "beofold-home-cluster")]
        cluster_secret: String,
    },

    /// Phase 2 harness: spawns 3+ live QUIC + Raft LAN nodes, kills the elected Leader mid-campaign,
    /// and verifies zero-loss Raft failover.
    SimulateCluster {
        #[arg(long, default_value = "./beofold_cluster_sim")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 3)]
        nodes: usize,
        #[arg(long, default_value_t = 3)]
        generations: u32,
        #[arg(long, default_value_t = 300)]
        steps_per_gen: u64,
    },

    /// Phase 3 harness: boots a live `fold-sync` HTTP server + TWO separate multi-node QUIC/Raft
    /// Beowulf LAN clusters (`Cluster-Alpha` & `Cluster-Beta`) and folds end-to-end over HTTP.
    SimulateFederation {
        #[arg(long, default_value = "./beofold_federation_sim")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        quorum: u8,
        #[arg(long, default_value_t = 2)]
        generations: u32,
        #[arg(long, default_value_t = 250)]
        steps_per_gen: u64,
    },

    /// Displays offline `redb` campaign PRCG completion table, MREMD swap rate, and best conformation.
    Status {
        #[arg(long, default_value = "./beofold_data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        project_id: u32,
    },

    /// Runs a turnkey single-node 4-rung MREMD Trp-cage (`1L2Y`) demonstration and exports `.pdb`.
    Demo {
        #[arg(long, default_value = "./beofold_demo")]
        data_dir: PathBuf,
        #[arg(long, default_value = "./trp_cage_best.pdb")]
        out_pdb: PathBuf,
        #[arg(long, default_value_t = 600)]
        steps_per_gen: u64,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    let enable_gpu = !cli.no_gpu;

    if let Some(n_threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads.max(1))
            .build_global()
            .ok();
    }

    match cli.command {
        Commands::Bench { steps } => {
            let id = NodeIdentity::generate_random()?;
            println!(
                "[BeoFold] Running {} Langevin BAOAB steps on 35-residue Villin Headpiece (GPU={})...",
                steps, enable_gpu
            );
            let prof = benchmark_local_node_with_gpu(id.public_key_bytes(), steps, enable_gpu);
            println!("  Host          : {} ({}/{})", prof.hostname, prof.os, prof.arch);
            println!("  Logical Cores : {}", prof.logical_cores);
            println!("  System RAM    : {} MB", prof.total_ram_mb);
            if enable_gpu {
                match probe_gpu_adapter() {
                    Some(g) => println!(
                        "  GPU Adapter   : {} (backend={}, type={})",
                        g.name, g.backend, g.device_type
                    ),
                    None => println!("  GPU Adapter   : none detected (using rayon CPU backend)"),
                }
            } else {
                println!("  GPU Adapter   : disabled via --no-gpu");
            }
            println!("  Throughput    : {:.2} ns/day", prof.villin_ns_per_day);
            println!("  Assigned Tier : {}", prof.tier);
            println!(
                "  Chunk Target  : {} steps/generation",
                prof.tier.recommended_steps()
            );
        }

        Commands::Init {
            data_dir,
            project_id,
            pdb,
            runs,
            clones,
            generations,
            steps_per_gen,
        } => {
            let store = FoldStore::open(&data_dir)?;
            init_campaign(
                &store,
                project_id,
                pdb.as_deref(),
                runs,
                clones,
                generations,
                steps_per_gen,
            )?;
        }

        Commands::RunLocal {
            data_dir,
            project_id,
            max_units,
            cluster_secret,
            export_best,
        } => {
            let store = FoldStore::open(&data_dir)?;
            execute_worker_loop(
                &store,
                project_id,
                max_units,
                &cluster_secret,
                export_best.as_deref(),
                enable_gpu,
            )?;
        }

        Commands::Mesh {
            data_dir,
            bind,
            advertise_ip,
            cluster_secret,
            seed_peer,
            no_subnet_sweep,
            auto_throttle,
            max_cpu,
            sync_url,
            cluster_name,
            init_trpcage,
            label,
        } => {
            // Treat blank strings from systemd EnvironmentFile / PowerShell wrappers as None
            let normalized_sync_url = sync_url
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());

            if init_trpcage && normalized_sync_url.is_none() {
                let store = FoldStore::open(&data_dir)?;
                if store.list_project_ids()?.is_empty() {
                    init_campaign(&store, 1, None, 4, 2, 4, 1000)?;
                }
            }

            let (shutdown_tx, shutdown_rx) = watch::channel(false);
            let cfg = MeshNodeConfig {
                data_dir,
                bind_addr: bind,
                advertise_ip,
                cluster_secret,
                seed_peers: seed_peer,
                enable_mdns: true,
                enable_subnet_sweep: !no_subnet_sweep,
                auto_throttle,
                max_system_cpu_pct: max_cpu,
                bench_steps: 600,
                stop_when_idle_completed: None,
                node_label: label,
                wan_sync_url: normalized_sync_url,
                wan_cluster_name: cluster_name,
                enable_gpu,
            };

            let state = start_mesh_node(cfg, shutdown_rx).await?;
            println!("[Mesh] Beowulf node online. Awaiting SIGINT / SIGTERM for clean shutdown.");
            let sig_name = wait_for_shutdown_signal().await;
            println!(
                "\n[Mesh] Caught {} — flushing checkpoints and closing QUIC, Raft & WAN tasks...",
                sig_name
            );
            let _ = shutdown_tx.send(true);
            state
                .transport
                .endpoint()
                .close(quinn::VarInt::from_u32(0), b"service-stop");
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }

        Commands::ClusterStatus {
            peer,
            cluster_secret,
        } => {
            let snap = query_remote_cluster_status(peer, &cluster_secret).await?;
            println!("=========================================================================");
            println!(
                " BeoFold LAN Beowulf Cluster Status (Raft Term {})",
                snap.raft_term
            );
            println!("=========================================================================");
            match &snap.leader {
                Some(ldr) => println!(
                    " Elected Leader : {} ({}..)",
                    ldr.addr,
                    &to_hex(&ldr.node_pubkey)[0..12]
                ),
                None => println!(" Elected Leader : (election in progress)"),
            }
            if let Some(pname) = &snap.active_project_name {
                println!(
                    " Active Project : P{} — {}",
                    snap.active_project_id.unwrap_or(0),
                    pname
                );
            }
            println!(
                " PRCG Progress  : {} completed / {} queued / {} leased ({} total)",
                snap.completed_workunits,
                snap.queued_workunits,
                snap.leased_workunits,
                snap.total_workunits
            );
            println!(
                " MREMD Swaps    : {} accepted / {} trials | Quarantined: {}",
                snap.mremd_swaps, snap.mremd_trials, snap.quarantined_events
            );
            if let Some((best_wu, best_e)) = snap.lowest_potential_kcal {
                println!(" Best Structure : {} ({:.2} kcal/mol)", best_wu, best_e);
            }
            println!("-------------------------------------------------------------------------");
            println!(" Connected Beowulf LAN Nodes ({}):", snap.peers.len());
            for p in &snap.peers {
                let wu_str = p
                    .active_wu
                    .map(|w| w.to_string())
                    .unwrap_or_else(|| "IDLE".into());
                let throt = if p.throttled { " [THROTTLED]" } else { "" };
                println!(
                    "   {} ({}/{}, {}c, {}, {:.1} ns/d) @ {} | Active: {}{}",
                    p.capability.hostname,
                    p.capability.os,
                    p.capability.arch,
                    p.capability.logical_cores,
                    p.capability.tier,
                    p.capability.villin_ns_per_day,
                    p.addr,
                    wu_str,
                    throt
                );
            }
            println!("=========================================================================");
        }

        Commands::SimulateCluster {
            data_dir,
            nodes,
            generations,
            steps_per_gen,
        } => {
            run_multi_node_failover_simulation(data_dir, nodes, generations, steps_per_gen).await?;
        }

        Commands::SimulateFederation {
            data_dir,
            quorum,
            generations,
            steps_per_gen,
        } => {
            run_two_cluster_wan_federation_simulation(data_dir, quorum, generations, steps_per_gen)
                .await?;
        }

        Commands::Status {
            data_dir,
            project_id,
        } => {
            let store = FoldStore::open(&data_dir)?;
            print_campaign_status(&store, project_id)?;
        }

        Commands::Demo {
            data_dir,
            out_pdb,
            steps_per_gen,
        } => {
            println!("==================================================================");
            println!(" BeoFold — Standalone Pure-Rust PRCG + MREMD Trp-Cage Demo");
            println!("==================================================================");
            let store = FoldStore::open(&data_dir)?;
            init_campaign(&store, 1, None, 4, 2, 3, steps_per_gen)?;
            execute_worker_loop(
                &store,
                1,
                0,
                "beofold-demo-cluster",
                Some(&out_pdb),
                enable_gpu,
            )?;
            print_campaign_status(&store, 1)?;
        }
    }

    Ok(())
}

/// Awaits either `Ctrl+C` (`SIGINT` on Unix/Windows) or `SIGTERM` (`systemctl stop` on Linux).
async fn wait_for_shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => "SIGINT",
                _ = sigterm.recv() => "SIGTERM",
            }
        } else {
            let _ = tokio::signal::ctrl_c().await;
            "SIGINT"
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "CTRL_C_EVENT"
    }
}

fn init_campaign(
    store: &FoldStore,
    project_id: u32,
    pdb_path: Option<&Path>,
    runs: u32,
    clones: u32,
    generations: u32,
    steps_per_gen: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (topo, mut seed_coords) = match pdb_path {
        Some(p) => {
            println!("[Init] Parsing structure file {} via pdbtbx...", p.display());
            ProteinTopology::from_pdb_file(p)?
        }
        None => {
            println!("[Init] Building built-in 20-residue Trp-cage (1L2Y) seed topology...");
            ProteinTopology::build_trp_cage_seed()
        }
    };

    let e_relaxed = minimize_steric_clashes(&topo, &mut seed_coords, 150);
    println!(
        "[Init] Topology '{}' ({} residues, {} atoms, {} bonds, {} angles, {} torsions) relaxed to {:.2} kcal/mol",
        topo.name,
        topo.residue_count,
        topo.atoms.len(),
        topo.bonds.len(),
        topo.angles.len(),
        topo.torsions.len(),
        e_relaxed
    );

    let n_rungs = runs.max(1);
    let mut ladder = Vec::with_capacity(n_rungs as usize);
    for r in 0..n_rungs {
        if n_rungs == 1 {
            ladder.push(300.0);
        } else {
            let frac = (r as f64) / ((n_rungs - 1) as f64);
            let temp = 285.0 * (405.0f64 / 285.0f64).powf(frac);
            ladder.push((temp * 10.0).round() / 10.0);
        }
    }

    let mut gen0_list = Vec::with_capacity((n_rungs * clones.max(1)) as usize);
    let salt = format!("project-{}-{}", project_id, topo.name);
    for r in 0..n_rungs {
        let temp_k = ladder[r as usize];
        for c in 0..clones.max(1) {
            let wu = WorkUnitId::new(project_id, r, c, 0);
            let cp = initialize_clone_checkpoint(&topo, &seed_coords, wu, temp_k, salt.as_bytes());
            gen0_list.push(cp);
        }
    }

    let manifest = ProjectManifest {
        project_id,
        name: topo.name.clone(),
        sequence: topo.sequence_one_letter.clone(),
        topology_blake3: [0u8; 32],
        initial_coords_blake3: [0u8; 32],
        atom_count: topo.atoms.len() as u32,
        residue_count: topo.residue_count,
        temp_ladder_kelvin: ladder.clone(),
        dt_fs: 1.5,
        friction_per_ps: 1.0,
        steps_per_gen: steps_per_gen.max(10),
        num_clones_per_run: clones.max(1),
        max_generations: generations.max(1),
    };

    let saved = store.seed_project(manifest, &topo, &gen0_list)?;
    println!(
        "[Init] Seeded Project P{} in {} (Topology BLAKE3: {}..)",
        saved.project_id,
        store.root_dir().display(),
        &to_hex(&saved.topology_blake3)[0..16]
    );
    println!(
        "[Init] MREMD Ladder: {:?} K | Total Campaign Chunks: {}",
        ladder,
        n_rungs * clones.max(1) * generations.max(1)
    );
    Ok(())
}

fn execute_worker_loop(
    store: &FoldStore,
    project_id: u32,
    max_units: usize,
    cluster_secret: &str,
    export_best: Option<&Path>,
    enable_gpu: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (manifest, topo) = store.load_project(project_id)?;
    let seed_cp = store.get_checkpoint(&manifest.initial_coords_blake3)?;
    let identity = NodeIdentity::load_or_create_keyfile(&store.root_dir().join("node_ed25519.key"))
        .unwrap_or_else(|_| NodeIdentity::from_cluster_secret(cluster_secret, "local-worker"));

    let gpu_engine = if enable_gpu {
        match GpuForceEngine::try_new(&topo) {
            Ok(g) => {
                println!(
                    "[GPU] Initialized wgpu compute pipeline on {} ({})",
                    g.adapter_summary().name,
                    g.adapter_summary().backend
                );
                Some(g)
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let mut mremd_rng =
        ChaCha8Rng::from_prcg_seed(WorkUnitId::new(project_id, 999, 999, 0), b"mremd-coordinator");
    let sanity_limits = SanityThresholds::default();

    let mut processed = 0usize;
    let mut global_min_energy = f64::INFINITY;
    let mut global_min_coords: Option<(WorkUnitId, Vec<[f32; 3]>)> = None;

    while max_units == 0 || processed < max_units {
        let now = unix_now();
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
            sample_stride: (lease.target_steps / 20).max(1),
        };

        let t0 = Instant::now();
        let summary =
            run_generation_chunk_with_gpu(&topo, gpu_engine.as_ref(), &input_cp, lease.wu_id, &cfg);
        let wall_ms = t0.elapsed().as_millis() as u64;

        if summary.min_potential_kcal < global_min_energy {
            global_min_energy = summary.min_potential_kcal;
            global_min_coords = Some((lease.wu_id, summary.min_energy_positions.clone()));
        }

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
            wall_time_ms: wall_ms,
            signature: [0u8; 64],
        };
        identity.sign_result(&mut receipt);

        let rmsd_from_seed = compute_ca_rmsd(
            &topo,
            &seed_cp.positions_angstrom,
            &summary.checkpoint.positions_angstrom,
        );

        let outcome = store.commit_signed_result(
            &manifest,
            &topo,
            &receipt,
            &summary.checkpoint,
            &sanity_limits,
            &mut mremd_rng,
            unix_now(),
        )?;

        match outcome {
            CommitOutcome::Accepted {
                completed_wu,
                spawned_next_gen,
                next_temperature_kelvin,
                mremd_swapped_with,
            } => {
                let swap_note = match mremd_swapped_with {
                    Some(partner) => format!(
                        " | MREMD SWAP <-> {} (next T={:.1}K)",
                        partner, next_temperature_kelvin
                    ),
                    None => String::new(),
                };
                let next_note = match spawned_next_gen {
                    Some(next_wu) => format!(" -> queued {}", next_wu),
                    None => " -> chain complete".to_string(),
                };
                println!(
                    "[Fold] {} | T={:>5.1}K | V_pot={:>8.2} kcal/mol (min {:>8.2}) | Rg={:>5.2} A | RMSD_seed={:>5.2} A | {} ms{}{}",
                    completed_wu,
                    cfg.temperature_kelvin,
                    summary.checkpoint.potential_energy_kcal,
                    summary.min_potential_kcal,
                    summary.checkpoint.rg_angstrom,
                    rmsd_from_seed,
                    wall_ms,
                    swap_note,
                    next_note
                );
            }
            CommitOutcome::AwaitingSecondQuorum { wu_id } => {
                println!("[Quorum] {} staged awaiting second verification", wu_id);
            }
            CommitOutcome::Quarantined { wu_id, reason } => {
                println!("[Quarantine] {} rejected & re-queued: {}", wu_id, reason);
            }
        }

        processed += 1;
    }

    if let Some(pdb_out) = export_best {
        if let Some((best_wu, coords)) = global_min_coords {
            topo.write_pdb_snapshot(
                &coords,
                pdb_out,
                &format!("Lowest energy conformation from {}", best_wu),
                global_min_energy,
            )?;
            println!(
                "[Export] Wrote lowest-energy trajectory structure ({}, {:.2} kcal/mol) to {}",
                best_wu,
                global_min_energy,
                pdb_out.display()
            );
        } else {
            let stats = store.collect_stats(Some(project_id))?;
            if let Some((best_wu, best_e, best_hash)) = stats.lowest_potential_kcal {
                let cp = store.get_checkpoint(&best_hash)?;
                topo.write_pdb_snapshot(
                    &cp.positions_angstrom,
                    pdb_out,
                    &format!("Checkpoint {}", best_wu),
                    best_e,
                )?;
                println!(
                    "[Export] Wrote lowest-energy checkpoint ({}, {:.2} kcal/mol) to {}",
                    best_wu,
                    best_e,
                    pdb_out.display()
                );
            }
        }
    }

    Ok(())
}

fn print_campaign_status(
    store: &FoldStore,
    project_id: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (manifest, _topo) = store.load_project(project_id)?;
    let stats = store.collect_stats(Some(project_id))?;
    let records = store.list_workunits(project_id)?;

    println!("------------------------------------------------------------------");
    println!(
        " Project P{}: {} ({} aa: {})",
        manifest.project_id, manifest.name, manifest.residue_count, manifest.sequence
    );
    println!(
        " Work Units : {} completed / {} queued / {} leased ({} discovered so far)",
        stats.completed, stats.queued, stats.leased, stats.total_workunits
    );
    let swap_pct = if stats.mremd_trials > 0 {
        100.0 * (stats.mremd_swaps as f64) / (stats.mremd_trials as f64)
    } else {
        0.0
    };
    println!(
        " MREMD Swaps: {} accepted / {} trials ({:.1}%)",
        stats.mremd_swaps, stats.mremd_trials, swap_pct
    );
    println!(" Quarantined: {} events", stats.quarantined_events);

    if let Some((best_wu, best_e, best_hash)) = stats.lowest_potential_kcal {
        let cp = store.get_checkpoint(&best_hash)?;
        println!(
            " Best State : {} | V_pot = {:.2} kcal/mol | Rg = {:.2} A | SASA = {:.1} A^2 | BLAKE3 = {}..",
            best_wu,
            best_e,
            cp.rg_angstrom,
            cp.sasa_angstrom2,
            &to_hex(&best_hash)[0..16]
        );
    }

    let preview_count = records.len().min(12);
    if preview_count > 0 {
        println!(
            " Recent PRCG Ledger Entries (showing {} of {}):",
            preview_count,
            records.len()
        );
        for rec in records.iter().rev().take(preview_count).rev() {
            let status_label = match &rec.status {
                WorkUnitStatus::Queued => "QUEUED".to_string(),
                WorkUnitStatus::Leased { .. } => "LEASED".to_string(),
                WorkUnitStatus::Completed { .. } => "COMPLETED".to_string(),
                WorkUnitStatus::Quarantined { failed_attempts, .. } => {
                    format!("QUARANTINED({})", failed_attempts)
                }
            };
            let pot_str = rec
                .final_potential_kcal
                .map(|v| format!("{:.2} kcal/mol", v))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "   {} | T={:>5.1}K | {:<12} | V_pot={}",
                rec.wu_id, rec.temperature_kelvin, status_label, pot_str
            );
        }
    }
    println!("------------------------------------------------------------------");
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
