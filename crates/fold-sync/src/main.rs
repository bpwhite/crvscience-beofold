use clap::{Parser, Subcommand};
use fold_physics::{initialize_clone_checkpoint, minimize_steric_clashes, ProteinTopology};
use fold_proto::{hex_to_bytes_fixed, ProjectManifest, WorkUnitId};
use fold_storage::{to_hex, FoldStore};
use fold_sync::{run_cgi_once, serve_axum, SyncCore, SyncServerPolicy};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Parser, Debug)]
#[command(
    name = "fold-sync",
    version,
    about = "BeoFold Apache HTTP Web Synchronizing Server (axum reverse-proxy daemon + mod_cgid CGI mode + public 3D dashboard)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Seeds a protein folding campaign on the web sync server (`ledger.redb` + `blobs/`).
    Init {
        #[arg(long, default_value = "./beofold_sync_data")]
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

    /// Runs Mode A: Persistent `axum` HTTP daemon behind Apache `mod_proxy_http`.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8780")]
        bind: SocketAddr,
        #[arg(long, default_value = "./beofold_sync_data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        quorum: u8,
        #[arg(long, default_value_t = false)]
        strict_trust: bool,
        #[arg(long)]
        trust_cluster: Vec<String>,
        #[arg(long, default_value_t = false)]
        init_trpcage: bool,
    },

    /// Runs Mode B: Single-shot RFC-3875 Apache `mod_cgid` request handler.
    Cgi {
        #[arg(long, default_value = "/var/lib/beofold")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        quorum: u8,
    },

    /// Prints connected remote Beowulf clusters, global PRCG progress, MREMD swaps, and best structure.
    Status {
        #[arg(long, default_value = "./beofold_sync_data")]
        data_dir: PathBuf,
        #[arg(long)]
        export_best: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Automatic RFC-3875 CGI detection when invoked directly by Apache `ScriptAlias` with zero CLI args
    if std::env::var("GATEWAY_INTERFACE").is_ok() && std::env::args().len() <= 1 {
        let data_dir = std::env::var("BEOFOLD_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/var/lib/beofold"));
        let quorum = std::env::var("BEOFOLD_QUORUM")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(1);
        let policy = SyncServerPolicy {
            default_quorum: quorum.max(1),
            trust_all_registered: quorum <= 1,
            ..Default::default()
        };
        let core = SyncCore::open(&data_dir, policy)?;
        run_cgi_once(&core)?;
        return Ok(());
    }

    let cli = Cli::parse();
    match cli.command {
        Commands::Init {
            data_dir,
            project_id,
            pdb,
            runs,
            clones,
            generations,
            steps_per_gen,
        } => {
            seed_sync_campaign(
                &data_dir,
                project_id,
                pdb.as_deref(),
                runs,
                clones,
                generations,
                steps_per_gen,
            )?;
        }

        Commands::Serve {
            bind,
            data_dir,
            quorum,
            strict_trust,
            trust_cluster,
            init_trpcage,
        } => {
            if init_trpcage {
                let store = FoldStore::open(&data_dir)?;
                if store.list_project_ids()?.is_empty() {
                    drop(store);
                    seed_sync_campaign(&data_dir, 1, None, 4, 2, 4, 1000)?;
                }
            }

            let mut trusted_set = HashSet::new();
            for hex_pk in &trust_cluster {
                trusted_set.insert(hex_to_bytes_fixed::<32>(hex_pk)?);
            }

            let policy = SyncServerPolicy {
                default_quorum: quorum.max(1),
                trusted_cluster_pubkeys: trusted_set,
                trust_all_registered: !strict_trust && quorum <= 1,
                ..Default::default()
            };

            let core = Arc::new(SyncCore::open(&data_dir, policy)?);
            let (tx, rx) = watch::channel(false);
            let bound = serve_axum(core, bind, rx).await?;
            println!("=========================================================================");
            println!(" BeoFold Apache Web Sync Server listening on http://{}", bound);
            println!("   Dashboard  : http://{}/beofold/dashboard", bound);
            println!("   API Base   : http://{}/beofold/api/v1", bound);
            println!("   Telemetry  : http://{}/beofold/api/v1/telemetry/global", bound);
            println!("   Best PDB   : http://{}/beofold/api/v1/public/conformation/best.pdb", bound);
            println!("   Data Vault : {}", data_dir.display());
            println!("=========================================================================");
            let sig = wait_for_shutdown_signal().await;
            println!("\n[SyncServer] Caught {} — closing redb ledger cleanly...", sig);
            let _ = tx.send(true);
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }

        Commands::Cgi { data_dir, quorum } => {
            let policy = SyncServerPolicy {
                default_quorum: quorum.max(1),
                trust_all_registered: quorum <= 1,
                ..Default::default()
            };
            let core = SyncCore::open(&data_dir, policy)?;
            run_cgi_once(&core)?;
        }

        Commands::Status {
            data_dir,
            export_best,
        } => {
            let store = FoldStore::open(&data_dir)?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let snap = store.build_global_telemetry(1, now)?;
            println!("{}", snap.to_pretty_json());

            if let Some(pdb_path) = export_best {
                let stats = store.collect_stats(None)?;
                if let Some((best_wu, best_e, best_hash)) = stats.lowest_potential_kcal {
                    let (_, topo) = store.load_project(best_wu.project)?;
                    let cp = store.get_checkpoint(&best_hash)?;
                    topo.write_pdb_snapshot(
                        &cp.positions_angstrom,
                        &pdb_path,
                        &format!("Global Federation Best Structure {}", best_wu),
                        best_e,
                    )?;
                    println!(
                        "[Sync] Exported global lowest-energy conformation ({}, {:.2} kcal/mol) to {}",
                        best_wu,
                        best_e,
                        pdb_path.display()
                    );
                }
            }
        }
    }

    Ok(())
}

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

fn seed_sync_campaign(
    data_dir: &Path,
    project_id: u32,
    pdb_path: Option<&Path>,
    runs: u32,
    clones: u32,
    generations: u32,
    steps_per_gen: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let store = FoldStore::open(data_dir)?;
    let (topo, mut coords) = match pdb_path {
        Some(p) => ProteinTopology::from_pdb_file(p)?,
        None => ProteinTopology::build_trp_cage_seed(),
    };
    minimize_steric_clashes(&topo, &mut coords, 150);

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

    let mut gen0 = Vec::new();
    for (r, &t) in ladder.iter().enumerate() {
        for c in 0..clones.max(1) {
            gen0.push(initialize_clone_checkpoint(
                &topo,
                &coords,
                WorkUnitId::new(project_id, r as u32, c, 0),
                t,
                b"apache-sync-seed",
            ));
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
        steps_per_gen: steps_per_gen.max(20),
        num_clones_per_run: clones.max(1),
        max_generations: generations.max(1),
    };

    let saved = store.seed_project(manifest, &topo, &gen0)?;
    println!(
        "[SyncInit] Seeded Project P{} '{}' ({} aa, {} chunks) | Topology BLAKE3: {}..",
        saved.project_id,
        saved.name,
        saved.residue_count,
        n_rungs * clones.max(1) * generations.max(1),
        &to_hex(&saved.topology_blake3)[0..16]
    );
    Ok(())
}
