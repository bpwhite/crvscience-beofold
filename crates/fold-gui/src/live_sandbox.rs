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

use fold_msm::ConformationalFrame;
use fold_physics::{
    compute_ca_rmsd, initialize_clone_checkpoint, minimize_steric_clashes, probe_gpu_adapter,
    run_generation_chunk_with_gpu, try_mremd_swap, ChaCha8Rng, GpuAdapterSummary, GpuForceEngine,
    LangevinConfig, ProteinTopology,
};
use fold_proto::{MremdEndpoint, TrajectoryCheckpoint, WorkUnitId};
use fold_storage::FoldStore;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_PLOT_HISTORY: usize = 240;
const MAX_MSM_ENSEMBLE_FRAMES: usize = 360;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPreset {
    TrpCage1L2Y,
    VillinHP35,
}

impl TargetPreset {
    pub fn label(self) -> &'static str {
        match self {
            TargetPreset::TrpCage1L2Y => "Trp-cage 1L2Y (20 aa)",
            TargetPreset::VillinHP35 => "Villin Headpiece 1YRF (35 aa)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RungSnapshot {
    pub rung_idx: usize,
    pub temperature_kelvin: f64,
    pub generation: u32,
    pub cumulative_steps: u64,
    pub potential_kcal: f64,
    pub rg_angstrom: f64,
    pub rmsd_seed_ang: f64,
}

/// Thread-safe live folding state displayed in the 3D viewport, thermodynamic charts, and MSM studio.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LiveFoldingState {
    pub topology: Arc<ProteinTopology>,
    pub seed_positions: Vec<[f32; 3]>,
    pub current_positions: Vec<[f32; 3]>,
    pub best_positions: Vec<[f32; 3]>,
    pub best_potential_kcal: f64,
    pub best_rg_angstrom: f64,
    pub best_sasa_angstrom2: f64,
    pub best_wu: WorkUnitId,

    pub running: bool,
    pub use_gpu: bool,
    pub steps_per_slice_active: u64,
    pub active_backend_label: String,
    pub gpu_adapter: Option<GpuAdapterSummary>,
    pub instantaneous_ns_per_day: f64,
    pub total_generations_done: u64,
    pub rungs: Vec<RungSnapshot>,
    /// Adjacent rung pair `(k, k+1)` `(accepted_swaps, total_trials)`.
    pub swap_pair_stats: Vec<(u64, u64)>,

    /// Rolling trajectory conformation ensemble across all 4 MREMD chains for live `fold-msm` analysis.
    pub msm_frames: VecDeque<ConformationalFrame>,

    pub potential_history: VecDeque<f64>,
    pub min_potential_history: VecDeque<f64>,
    pub rg_history: VecDeque<f64>,
    pub rmsd_history: VecDeque<f64>,
    pub status_banner: String,
}

pub enum SandboxCommand {
    SetRunning(bool),
    ResetTarget {
        preset: TargetPreset,
        use_gpu: bool,
        steps_per_slice: u64,
    },
    LoadFromVault {
        data_dir: std::path::PathBuf,
        project_id: u32,
    },
    /// Loads a clicked PCCA+ macrostate or microstate conformation from the MSM tab into the 3D Studio.
    LoadMsmConformation {
        topology: Arc<ProteinTopology>,
        positions: Vec<[f32; 3]>,
        wu_id: WorkUnitId,
        potential_kcal: f64,
        rg_angstrom: f64,
        label: String,
    },
    /// Runs `rounds` immediate MREMD slices (`4 * rounds` conformations) so Tab 4 can build a
    /// live MSM on a cold launch with one click.
    RunBurstRounds {
        rounds: u32,
    },
    SetStepsPerSlice(u64),
    SetUseGpu(bool),
    Shutdown,
}

impl LiveFoldingState {
    pub fn new_preset(preset: TargetPreset, use_gpu: bool) -> Self {
        let (topo, mut seed_coords) = match preset {
            TargetPreset::TrpCage1L2Y => ProteinTopology::build_trp_cage_seed(),
            TargetPreset::VillinHP35 => ProteinTopology::build_villin_benchmark_seed(),
        };
        let e0 = minimize_steric_clashes(&topo, &mut seed_coords, 120);
        let gpu_adapter = probe_gpu_adapter();

        let ladder = [285.0, 318.0, 356.0, 398.0];
        let mut rungs = Vec::with_capacity(ladder.len());
        let mut msm_frames = VecDeque::with_capacity(MAX_MSM_ENSEMBLE_FRAMES);

        for (idx, &temp) in ladder.iter().enumerate() {
            let wu0 = WorkUnitId::new(1, idx as u32, 0, 0);
            let cp = initialize_clone_checkpoint(&topo, &seed_coords, wu0, temp, b"gui-live-studio");
            rungs.push(RungSnapshot {
                rung_idx: idx,
                temperature_kelvin: temp,
                generation: 0,
                cumulative_steps: 0,
                potential_kcal: cp.potential_energy_kcal,
                rg_angstrom: cp.rg_angstrom,
                rmsd_seed_ang: 0.0,
            });
            msm_frames.push_back(ConformationalFrame {
                frame_idx: idx,
                wu_id: wu0,
                chain_step: 0,
                checkpoint_blake3: [0u8; 32],
                temperature_kelvin: temp,
                potential_energy_kcal: cp.potential_energy_kcal,
                rg_angstrom: cp.rg_angstrom,
                sasa_angstrom2: cp.sasa_angstrom2,
                rmsd_to_seed_ang: 0.0,
                rmsd_to_lowest_energy_ang: 0.0,
                positions_angstrom: cp.positions_angstrom,
            });
        }

        let rg0 = rungs.first().map(|r| r.rg_angstrom).unwrap_or(10.0);
        let backend_label = if use_gpu {
            gpu_adapter
                .as_ref()
                .map(|g| format!("wgpu {} ({})", g.name, g.backend))
                .unwrap_or_else(|| "rayon CPU (no GPU ICD)".to_string())
        } else {
            "rayon Multi-Core CPU".to_string()
        };

        let mut pot_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
        pot_hist.push_back(e0);
        let mut min_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
        min_hist.push_back(e0);
        let mut rg_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
        rg_hist.push_back(rg0);
        let mut rmsd_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
        rmsd_hist.push_back(0.0);

        Self {
            topology: Arc::new(topo),
            seed_positions: seed_coords.clone(),
            current_positions: seed_coords.clone(),
            best_positions: seed_coords,
            best_potential_kcal: e0,
            best_rg_angstrom: rg0,
            best_sasa_angstrom2: 0.0,
            best_wu: WorkUnitId::new(1, 0, 0, 0),
            running: false,
            use_gpu,
            steps_per_slice_active: 150,
            active_backend_label: backend_label,
            gpu_adapter,
            instantaneous_ns_per_day: 0.0,
            total_generations_done: 0,
            rungs,
            swap_pair_stats: vec![(0, 0); 3],
            msm_frames,
            potential_history: pot_hist,
            min_potential_history: min_hist,
            rg_history: rg_hist,
            rmsd_history: rmsd_hist,
            status_banner: format!(
                "Minimized {} starting seed ({:.2} kcal/mol). Click 'Start Live MREMD' to fold.",
                preset.label(),
                e0
            ),
        }
    }

    pub fn export_best_pdb(&self, out_path: &Path) -> Result<(), String> {
        self.topology
            .write_pdb_snapshot(
                &self.best_positions,
                out_path,
                &format!("Supervisor Export {}", self.best_wu),
                self.best_potential_kcal,
            )
            .map_err(|e| e.to_string())
    }
}

pub fn spawn_live_folding_worker(
    shared: Arc<Mutex<LiveFoldingState>>,
    initial_preset: TargetPreset,
    initial_use_gpu: bool,
) -> mpsc::Sender<SandboxCommand> {
    let (tx, rx) = mpsc::channel::<SandboxCommand>();

    std::thread::Builder::new()
        .name("beofold-gui-mremd-worker".into())
        .spawn(move || {
            let mut preset = initial_preset;
            let mut use_gpu = initial_use_gpu;
            let mut steps_per_slice = 150u64;
            let mut running = false;
            let mut pending_burst_rounds = 0u32;

            let (mut topo, mut checkpoints, mut gpu_engine, mut seed_coords) =
                init_worker_replicas(preset, use_gpu);
            let mut swap_rng =
                ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 42, 42, 0), b"gui-swap");
            let mut even_phase = true;

            loop {
                while let Ok(cmd) = rx.try_recv() {
                    match cmd {
                        SandboxCommand::SetRunning(flag) => {
                            running = flag;
                            if let Ok(mut g) = shared.lock() {
                                g.running = running;
                                g.status_banner = if running {
                                    format!("Folding {} across 4 MREMD rungs...", topo.name)
                                } else {
                                    "Paused local MREMD runner.".into()
                                };
                            }
                        }
                        SandboxCommand::RunBurstRounds { rounds } => {
                            pending_burst_rounds = pending_burst_rounds.saturating_add(rounds.max(1));
                        }
                        SandboxCommand::SetStepsPerSlice(s) => {
                            steps_per_slice = s.clamp(25, 5_000);
                            if let Ok(mut g) = shared.lock() {
                                g.steps_per_slice_active = steps_per_slice;
                            }
                        }
                        SandboxCommand::SetUseGpu(flag) => {
                            use_gpu = flag;
                            gpu_engine = if use_gpu {
                                GpuForceEngine::try_new(&topo).ok()
                            } else {
                                None
                            };
                            if let Ok(mut g) = shared.lock() {
                                g.use_gpu = use_gpu;
                                g.active_backend_label = match &gpu_engine {
                                    Some(eng) => format!(
                                        "wgpu {} ({})",
                                        eng.adapter_summary().name,
                                        eng.adapter_summary().backend
                                    ),
                                    None => "rayon Multi-Core CPU".into(),
                                };
                            }
                        }
                        SandboxCommand::ResetTarget {
                            preset: new_p,
                            use_gpu: new_gpu,
                            steps_per_slice: s,
                        } => {
                            preset = new_p;
                            use_gpu = new_gpu;
                            steps_per_slice = s.clamp(25, 5_000);
                            running = false;
                            pending_burst_rounds = 0;
                            let mut fresh_state = LiveFoldingState::new_preset(preset, use_gpu);
                            fresh_state.steps_per_slice_active = steps_per_slice;
                            let rebuilt = init_worker_replicas(preset, use_gpu);
                            topo = rebuilt.0;
                            checkpoints = rebuilt.1;
                            gpu_engine = rebuilt.2;
                            seed_coords = rebuilt.3;
                            if let Ok(mut g) = shared.lock() {
                                *g = fresh_state;
                            }
                        }
                        SandboxCommand::LoadFromVault {
                            data_dir,
                            project_id,
                        } => {
                            running = false;
                            match load_vault_best_conformation(&data_dir, project_id) {
                                Ok((v_topo, best_wu, cp)) => {
                                    if let Ok(mut g) = shared.lock() {
                                        g.running = false;
                                        g.topology = Arc::new(v_topo.clone());
                                        g.current_positions = cp.positions_angstrom.clone();
                                        g.best_positions = cp.positions_angstrom.clone();
                                        g.best_potential_kcal = cp.potential_energy_kcal;
                                        g.best_rg_angstrom = cp.rg_angstrom;
                                        g.best_sasa_angstrom2 = cp.sasa_angstrom2;
                                        g.best_wu = best_wu;
                                        g.status_banner = format!(
                                            "Loaded best vault checkpoint {} ({:.2} kcal/mol, Rg={:.2} A) from {}",
                                            best_wu,
                                            cp.potential_energy_kcal,
                                            cp.rg_angstrom,
                                            data_dir.display()
                                        );
                                    }
                                    topo = v_topo;
                                    gpu_engine = if use_gpu {
                                        GpuForceEngine::try_new(&topo).ok()
                                    } else {
                                        None
                                    };
                                }
                                Err(err) => {
                                    if let Ok(mut g) = shared.lock() {
                                        g.status_banner = format!("Vault load error: {}", err);
                                    }
                                }
                            }
                        }
                        SandboxCommand::LoadMsmConformation {
                            topology: msm_topo,
                            positions,
                            wu_id,
                            potential_kcal,
                            rg_angstrom,
                            label,
                        } => {
                            running = false;
                            topo = (*msm_topo).clone();
                            gpu_engine = if use_gpu {
                                GpuForceEngine::try_new(&topo).ok()
                            } else {
                                None
                            };
                            if let Ok(mut g) = shared.lock() {
                                g.running = false;
                                g.topology = msm_topo;
                                g.current_positions = positions.clone();
                                g.best_positions = positions;
                                g.best_potential_kcal = potential_kcal;
                                g.best_rg_angstrom = rg_angstrom;
                                g.best_wu = wu_id;
                                g.status_banner = format!(
                                    "Inspecting MSM Conformation {} ({}) | V_pot={:.2} kcal/mol | Rg={:.2} A",
                                    label, wu_id, potential_kcal, rg_angstrom
                                );
                            }
                        }
                        SandboxCommand::Shutdown => return,
                    }
                }

                if !running && pending_burst_rounds == 0 {
                    std::thread::sleep(Duration::from_millis(40));
                    continue;
                }
                if pending_burst_rounds > 0 {
                    pending_burst_rounds -= 1;
                }

                let t0 = Instant::now();
                let mut total_steps_batch = 0u64;
                let mut newly_generated_frames: Vec<ConformationalFrame> =
                    Vec::with_capacity(checkpoints.len());

                for (r_idx, cp) in checkpoints.iter_mut().enumerate() {
                    let next_gen = cp.wu_id.generation + 1;
                    let next_wu = WorkUnitId::new(1, r_idx as u32, 0, next_gen);
                    let cfg = LangevinConfig {
                        dt_fs: 1.5,
                        friction_per_ps: 1.0,
                        temperature_kelvin: cp.temperature_kelvin,
                        steps: steps_per_slice,
                        sample_stride: (steps_per_slice / 5).max(1),
                    };
                    let summary = run_generation_chunk_with_gpu(
                        &topo,
                        gpu_engine.as_ref(),
                        cp,
                        next_wu,
                        &cfg,
                    );
                    total_steps_batch += steps_per_slice;

                    let rmsd_seed =
                        compute_ca_rmsd(&topo, &seed_coords, &summary.checkpoint.positions_angstrom);
                    newly_generated_frames.push(ConformationalFrame {
                        frame_idx: 0,
                        wu_id: next_wu,
                        chain_step: next_gen,
                        checkpoint_blake3: [0u8; 32],
                        temperature_kelvin: summary.checkpoint.temperature_kelvin,
                        potential_energy_kcal: summary.checkpoint.potential_energy_kcal,
                        rg_angstrom: summary.checkpoint.rg_angstrom,
                        sasa_angstrom2: summary.checkpoint.sasa_angstrom2,
                        rmsd_to_seed_ang: rmsd_seed,
                        rmsd_to_lowest_energy_ang: 0.0,
                        positions_angstrom: summary.checkpoint.positions_angstrom.clone(),
                    });

                    if let Ok(mut g) = shared.lock() {
                        if summary.min_potential_kcal < g.best_potential_kcal {
                            g.best_potential_kcal = summary.min_potential_kcal;
                            g.best_positions = summary.min_energy_positions.clone();
                            g.best_rg_angstrom = summary.checkpoint.rg_angstrom;
                            g.best_sasa_angstrom2 = summary.checkpoint.sasa_angstrom2;
                            g.best_wu = next_wu;
                        }
                    }

                    *cp = summary.checkpoint;
                }

                let mut swap_updates: Vec<(usize, bool)> = Vec::new();
                let start_pair = if even_phase { 0usize } else { 1usize };
                even_phase = !even_phase;

                let mut k = start_pair;
                while k + 1 < checkpoints.len() {
                    let ep_a = MremdEndpoint {
                        wu_id: checkpoints[k].wu_id,
                        rung_index: k as u32,
                        temperature_kelvin: checkpoints[k].temperature_kelvin,
                        potential_energy_kcal: checkpoints[k].potential_energy_kcal,
                        output_checkpoint_blake3: [0u8; 32],
                    };
                    let ep_b = MremdEndpoint {
                        wu_id: checkpoints[k + 1].wu_id,
                        rung_index: (k + 1) as u32,
                        temperature_kelvin: checkpoints[k + 1].temperature_kelvin,
                        potential_energy_kcal: checkpoints[k + 1].potential_energy_kcal,
                        output_checkpoint_blake3: [0u8; 32],
                    };
                    let (accepted, new_ta, new_tb) = try_mremd_swap(&ep_a, &ep_b, &mut swap_rng);
                    if accepted {
                        checkpoints[k].temperature_kelvin = new_ta;
                        checkpoints[k + 1].temperature_kelvin = new_tb;
                    }
                    swap_updates.push((k, accepted));
                    k += 2;
                }

                let elapsed_s = t0.elapsed().as_secs_f64().max(1e-6);
                let sim_ns = (total_steps_batch as f64) * 1.5e-6;
                let ns_per_day = sim_ns * (86_400.0 / elapsed_s);

                let mut coldest_idx = 0usize;
                for i in 1..checkpoints.len() {
                    if checkpoints[i].temperature_kelvin < checkpoints[coldest_idx].temperature_kelvin
                    {
                        coldest_idx = i;
                    }
                }

                if let Ok(mut g) = shared.lock() {
                    g.instantaneous_ns_per_day = ns_per_day;
                    g.total_generations_done += checkpoints.len() as u64;
                    g.current_positions = checkpoints[coldest_idx].positions_angstrom.clone();

                    for nf in newly_generated_frames {
                        if g.msm_frames.len() >= MAX_MSM_ENSEMBLE_FRAMES {
                            g.msm_frames.pop_front();
                        }
                        g.msm_frames.push_back(nf);
                    }

                    for (pair_idx, acc) in swap_updates {
                        if let Some(slot) = g.swap_pair_stats.get_mut(pair_idx) {
                            slot.1 += 1;
                            if acc {
                                slot.0 += 1;
                            }
                        }
                    }

                    for (idx, cp) in checkpoints.iter().enumerate() {
                        let rmsd = compute_ca_rmsd(&topo, &seed_coords, &cp.positions_angstrom);
                        if let Some(slot) = g.rungs.get_mut(idx) {
                            slot.temperature_kelvin = cp.temperature_kelvin;
                            slot.generation = cp.wu_id.generation;
                            slot.cumulative_steps = cp.cumulative_steps;
                            slot.potential_kcal = cp.potential_energy_kcal;
                            slot.rg_angstrom = cp.rg_angstrom;
                            slot.rmsd_seed_ang = rmsd;
                        }
                    }

                    let cold_pot = checkpoints[coldest_idx].potential_energy_kcal;
                    let cold_rg = checkpoints[coldest_idx].rg_angstrom;
                    let cold_rmsd = compute_ca_rmsd(
                        &topo,
                        &seed_coords,
                        &checkpoints[coldest_idx].positions_angstrom,
                    );
                    let best_pot = g.best_potential_kcal;

                    push_bounded(&mut g.potential_history, cold_pot);
                    push_bounded(&mut g.min_potential_history, best_pot);
                    push_bounded(&mut g.rg_history, cold_rg);
                    push_bounded(&mut g.rmsd_history, cold_rmsd);
                }
            }
        })
        .ok();

    tx
}

fn push_bounded(buf: &mut VecDeque<f64>, val: f64) {
    if buf.len() >= MAX_PLOT_HISTORY {
        buf.pop_front();
    }
    buf.push_back(val);
}

fn init_worker_replicas(
    preset: TargetPreset,
    use_gpu: bool,
) -> (
    ProteinTopology,
    Vec<TrajectoryCheckpoint>,
    Option<GpuForceEngine>,
    Vec<[f32; 3]>,
) {
    let (topo, mut seed_coords) = match preset {
        TargetPreset::TrpCage1L2Y => ProteinTopology::build_trp_cage_seed(),
        TargetPreset::VillinHP35 => ProteinTopology::build_villin_benchmark_seed(),
    };
    minimize_steric_clashes(&topo, &mut seed_coords, 120);

    let ladder = [285.0, 318.0, 356.0, 398.0];
    let mut cps = Vec::with_capacity(ladder.len());
    for (idx, &t) in ladder.iter().enumerate() {
        cps.push(initialize_clone_checkpoint(
            &topo,
            &seed_coords,
            WorkUnitId::new(1, idx as u32, 0, 0),
            t,
            b"gui-live-studio",
        ));
    }

    let gpu_engine = if use_gpu {
        GpuForceEngine::try_new(&topo).ok()
    } else {
        None
    };

    (topo, cps, gpu_engine, seed_coords)
}

fn load_vault_best_conformation(
    data_dir: &Path,
    project_id: u32,
) -> Result<(ProteinTopology, WorkUnitId, TrajectoryCheckpoint), String> {
    let store = FoldStore::open(data_dir).map_err(|e| e.to_string())?;
    let (manifest, topo) = store.load_project(project_id).map_err(|e| e.to_string())?;
    let stats = store
        .collect_stats(Some(project_id))
        .map_err(|e| e.to_string())?;

    if let Some((best_wu, _best_e, best_hash)) = stats.lowest_potential_kcal {
        let cp = store.get_checkpoint(&best_hash).map_err(|e| e.to_string())?;
        Ok((topo, best_wu, cp))
    } else {
        let cp = store
            .get_checkpoint(&manifest.initial_coords_blake3)
            .map_err(|e| e.to_string())?;
        Ok((topo, cp.wu_id, cp))
    }
}
