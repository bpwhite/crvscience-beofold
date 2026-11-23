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
    compute_ca_rmsd, fraction_native_contacts, initialize_clone_checkpoint,
    minimize_steric_clashes, native_ca_contact_pairs, probe_gpu_adapter,
    run_generation_chunk_with_gpu, try_mremd_swap_checkpoints, ChaCha8Rng, GpuAdapterSummary,
    GpuForceEngine, LangevinConfig, ProteinTopology,
};
use fold_proto::{TrajectoryCheckpoint, WorkUnitId};
use fold_storage::FoldStore;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_PLOT_HISTORY: usize = 240;
const MAX_MSM_ENSEMBLE_FRAMES: usize = 360;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeReferenceSource {
    PdbCrystal,
    LowestEnergyFold,
}

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
    pub target_display_name: String,
    pub native_source: NativeReferenceSource,
    pub native_pairs: Vec<(usize, usize)>,
    pub seed_positions: Vec<[f32; 3]>,
    pub seed_potential_kcal: f64,
    pub seed_rg_angstrom: f64,
    pub current_positions: Vec<[f32; 3]>,
    pub current_wu: WorkUnitId,
    pub current_potential_kcal: f64,
    pub inspected_label: Option<String>,
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
    pub q_history: VecDeque<f64>,
    pub status_banner: String,
}

pub enum SandboxCommand {
    SetRunning(bool),
    ResetTarget {
        preset: TargetPreset,
        use_gpu: bool,
        steps_per_slice: u64,
    },
    LoadCustomTarget {
        name: String,
        topology: ProteinTopology,
        positions: Vec<[f32; 3]>,
        from_pdb: bool,
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
        let (topo, seed_coords) = match preset {
            TargetPreset::TrpCage1L2Y => ProteinTopology::build_trp_cage_seed(),
            TargetPreset::VillinHP35 => ProteinTopology::build_villin_benchmark_seed(),
        };
        Self::new_from_topology(preset.label().to_string(), topo, seed_coords, false, use_gpu)
    }

    pub fn new_from_topology(
        display_name: String,
        topo: ProteinTopology,
        initial_coords: Vec<[f32; 3]>,
        from_pdb: bool,
        use_gpu: bool,
    ) -> Self {
        let native_source = if from_pdb {
            NativeReferenceSource::PdbCrystal
        } else {
            NativeReferenceSource::LowestEnergyFold
        };
        let native_pairs = if from_pdb {
            native_ca_contact_pairs(&topo, &initial_coords, 8.5)
        } else {
            Vec::new()
        };
        let mut seed_coords = initial_coords;
        let min_steps = if from_pdb { 60 } else { 120 };
        let e0 = minimize_steric_clashes(&topo, &mut seed_coords, min_steps);
        let native_pairs = if from_pdb {
            native_pairs
        } else {
            native_ca_contact_pairs(&topo, &seed_coords, 8.5)
        };
        let q0 = fraction_native_contacts(&native_pairs, &seed_coords, 8.8);
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
        let mut q_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
        q_hist.push_back(q0);

        Self {
            topology: Arc::new(topo),
            target_display_name: display_name.clone(),
            native_source,
            native_pairs,
            seed_positions: seed_coords.clone(),
            seed_potential_kcal: e0,
            seed_rg_angstrom: rg0,
            current_positions: seed_coords.clone(),
            current_wu: WorkUnitId::new(1, 0, 0, 0),
            current_potential_kcal: e0,
            inspected_label: None,
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
            q_history: q_hist,
            status_banner: format!(
                "Loaded {} ({:.2} kcal/mol, Q={:.2}). Click 'Start Live MREMD' to fold.",
                display_name, e0, q0
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

pub(crate) struct WorkerSimState {
    pub(crate) use_gpu: bool,
    pub(crate) steps_per_slice: u64,
    pub(crate) running: bool,
    pub(crate) pending_burst_rounds: u32,
    pub(crate) topo: ProteinTopology,
    pub(crate) checkpoints: Vec<TrajectoryCheckpoint>,
    pub(crate) gpu_engine: Option<GpuForceEngine>,
    pub(crate) seed_coords: Vec<[f32; 3]>,
    pub(crate) cold_positions_history: VecDeque<Vec<[f32; 3]>>,
    pub(crate) swap_rng: ChaCha8Rng,
    pub(crate) even_phase: bool,
}

impl WorkerSimState {
    pub(crate) fn new(preset: TargetPreset, use_gpu: bool) -> Self {
        let (topo, checkpoints, gpu_engine, seed_coords) = init_worker_replicas(preset, use_gpu);
        let mut cold_positions_history = VecDeque::with_capacity(MAX_PLOT_HISTORY);
        cold_positions_history.push_back(seed_coords.clone());
        Self {
            use_gpu,
            steps_per_slice: 150,
            running: false,
            pending_burst_rounds: 0,
            topo,
            checkpoints,
            gpu_engine,
            seed_coords,
            cold_positions_history,
            swap_rng: ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 42, 42, 0), b"gui-swap"),
            even_phase: true,
        }
    }

    /// Processes a single `SandboxCommand`. Returns `true` if the worker should shut down.
    pub(crate) fn handle_command(
        &mut self,
        cmd: SandboxCommand,
        shared: &Arc<Mutex<LiveFoldingState>>,
    ) -> bool {
        match cmd {
            SandboxCommand::SetRunning(flag) => {
                self.running = flag;
                if let Ok(mut g) = shared.lock() {
                    g.running = self.running;
                    g.status_banner = if self.running {
                        format!("Folding {} across 4 MREMD rungs...", self.topo.name)
                    } else {
                        "Paused local MREMD runner.".into()
                    };
                }
            }
            SandboxCommand::RunBurstRounds { rounds } => {
                self.pending_burst_rounds = self.pending_burst_rounds.saturating_add(rounds.max(1));
            }
            SandboxCommand::SetStepsPerSlice(s) => {
                self.steps_per_slice = s.clamp(25, 5_000);
                if let Ok(mut g) = shared.lock() {
                    g.steps_per_slice_active = self.steps_per_slice;
                }
            }
            SandboxCommand::SetUseGpu(flag) => {
                self.use_gpu = flag;
                self.gpu_engine = if self.use_gpu {
                    GpuForceEngine::try_new(&self.topo).ok()
                } else {
                    None
                };
                if let Ok(mut g) = shared.lock() {
                    g.use_gpu = self.use_gpu;
                    g.active_backend_label = backend_label_for(self.use_gpu, &self.gpu_engine);
                }
            }
            SandboxCommand::ResetTarget {
                preset: new_p,
                use_gpu: new_gpu,
                steps_per_slice: s,
            } => {
                self.use_gpu = new_gpu;
                self.steps_per_slice = s.clamp(25, 5_000);
                self.running = false;
                self.pending_burst_rounds = 0;
                let mut fresh_state = LiveFoldingState::new_preset(new_p, self.use_gpu);
                fresh_state.steps_per_slice_active = self.steps_per_slice;
                let rebuilt = init_worker_replicas(new_p, self.use_gpu);
                self.topo = rebuilt.0;
                self.checkpoints = rebuilt.1;
                self.gpu_engine = rebuilt.2;
                self.seed_coords = rebuilt.3;
                self.cold_positions_history.clear();
                self.cold_positions_history.push_back(self.seed_coords.clone());
                self.swap_rng =
                    ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 42, 42, 0), b"gui-swap");
                self.even_phase = true;
                if let Ok(mut g) = shared.lock() {
                    *g = fresh_state;
                }
            }
            SandboxCommand::LoadCustomTarget {
                name,
                topology: custom_topo,
                positions: custom_pos,
                from_pdb,
                use_gpu: new_gpu,
                steps_per_slice: s,
            } => {
                self.use_gpu = new_gpu;
                self.steps_per_slice = s.clamp(25, 5_000);
                self.running = false;
                self.pending_burst_rounds = 0;
                let mut fresh_state = LiveFoldingState::new_from_topology(
                    name,
                    custom_topo.clone(),
                    custom_pos,
                    from_pdb,
                    self.use_gpu,
                );
                fresh_state.steps_per_slice_active = self.steps_per_slice;
                let rebuilt = init_replicas_from_topology_and_seed(
                    custom_topo,
                    fresh_state.seed_positions.clone(),
                    self.use_gpu,
                );
                self.topo = rebuilt.0;
                self.checkpoints = rebuilt.1;
                self.gpu_engine = rebuilt.2;
                self.seed_coords = rebuilt.3;
                self.cold_positions_history.clear();
                self.cold_positions_history.push_back(self.seed_coords.clone());
                self.swap_rng =
                    ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 42, 42, 0), b"gui-swap");
                self.even_phase = true;
                if let Ok(mut g) = shared.lock() {
                    *g = fresh_state;
                }
            }
            SandboxCommand::LoadFromVault {
                data_dir,
                project_id,
            } => {
                self.running = false;
                self.pending_burst_rounds = 0;
                match load_vault_best_conformation(&data_dir, project_id) {
                    Ok((mut v_topo, best_wu, mut best_cp, init_cp_opt)) => {
                        let expected_atoms = v_topo.atoms.len();
                        let valid_best_pos = match ensure_matching_positions(
                            &mut v_topo,
                            best_cp.positions_angstrom.clone(),
                        ) {
                            Ok(p) => p,
                            Err(err) => {
                                if let Ok(mut g) = shared.lock() {
                                    g.running = false;
                                    g.status_banner = format!("Vault load error: {}", err);
                                }
                                return false;
                            }
                        };
                        if valid_best_pos.len() != best_cp.positions_angstrom.len()
                            || v_topo.atoms.len() != expected_atoms
                        {
                            best_cp = initialize_clone_checkpoint(
                                &v_topo,
                                &valid_best_pos,
                                best_wu,
                                best_cp.temperature_kelvin,
                                b"vault-reseed",
                            );
                        }

                        let (seed_pos, seed_pot, seed_rg) = match init_cp_opt {
                            Some(icp) if icp.positions_angstrom.len() == v_topo.atoms.len() => (
                                icp.positions_angstrom,
                                icp.potential_energy_kcal,
                                icp.rg_angstrom,
                            ),
                            _ => (
                                best_cp.positions_angstrom.clone(),
                                best_cp.potential_energy_kcal,
                                best_cp.rg_angstrom,
                            ),
                        };

                        let rebuilt = init_replicas_from_topology_and_seed(
                            v_topo,
                            best_cp.positions_angstrom.clone(),
                            self.use_gpu,
                        );
                        self.topo = rebuilt.0;
                        self.checkpoints = rebuilt.1;
                        self.gpu_engine = rebuilt.2;
                        self.seed_coords = seed_pos;
                        self.cold_positions_history.clear();
                        self.cold_positions_history
                            .push_back(best_cp.positions_angstrom.clone());
                        self.swap_rng =
                            ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 42, 42, 0), b"gui-swap");
                        self.even_phase = true;

                        let banner = format!(
                            "Loaded best vault checkpoint {} ({:.2} kcal/mol, Rg={:.2} A) from {}",
                            best_wu,
                            best_cp.potential_energy_kcal,
                            best_cp.rg_angstrom,
                            data_dir.display()
                        );
                        let fresh_state = build_loaded_state(
                            format!("{} ({} aa, Vault)", self.topo.name, self.topo.residue_count),
                            &self.topo,
                            &self.seed_coords,
                            seed_pot,
                            seed_rg,
                            &best_cp.positions_angstrom,
                            best_wu,
                            best_cp.potential_energy_kcal,
                            best_cp.rg_angstrom,
                            best_cp.sasa_angstrom2,
                            None,
                            &self.checkpoints,
                            self.use_gpu,
                            self.steps_per_slice,
                            backend_label_for(self.use_gpu, &self.gpu_engine),
                            banner,
                        );
                        if let Ok(mut g) = shared.lock() {
                            *g = fresh_state;
                        }
                    }
                    Err(err) => {
                        if let Ok(mut g) = shared.lock() {
                            g.running = false;
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
                self.running = false;
                self.pending_burst_rounds = 0;

                let topo_changed = msm_topo.atoms.len() != self.topo.atoms.len()
                    || msm_topo.sequence_one_letter != self.topo.sequence_one_letter;

                if topo_changed {
                    let mut target_topo = (*msm_topo).clone();
                    let valid_pos = match ensure_matching_positions(&mut target_topo, positions) {
                        Ok(p) => p,
                        Err(err) => {
                            if let Ok(mut g) = shared.lock() {
                                g.running = false;
                                g.status_banner = format!("MSM conformation load error: {}", err);
                            }
                            return false;
                        }
                    };
                    let rebuilt = init_replicas_from_topology_and_seed(
                        target_topo,
                        valid_pos.clone(),
                        self.use_gpu,
                    );
                    self.topo = rebuilt.0;
                    self.checkpoints = rebuilt.1;
                    self.gpu_engine = rebuilt.2;
                    self.seed_coords = rebuilt.3;
                    self.cold_positions_history.clear();
                    self.cold_positions_history.push_back(valid_pos.clone());
                    self.swap_rng =
                        ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 42, 42, 0), b"gui-swap");
                    self.even_phase = true;

                    let banner = format!(
                        "Inspecting MSM Conformation {} ({}) | V_pot={:.2} kcal/mol | Rg={:.2} A",
                        label, wu_id, potential_kcal, rg_angstrom
                    );
                    let fresh_state = build_loaded_state(
                        format!("{} ({} aa, MSM)", self.topo.name, self.topo.residue_count),
                        &self.topo,
                        &self.seed_coords,
                        potential_kcal,
                        rg_angstrom,
                        &valid_pos,
                        wu_id,
                        potential_kcal,
                        rg_angstrom,
                        0.0,
                        Some(label),
                        &self.checkpoints,
                        self.use_gpu,
                        self.steps_per_slice,
                        backend_label_for(self.use_gpu, &self.gpu_engine),
                        banner,
                    );
                    if let Ok(mut g) = shared.lock() {
                        *g = fresh_state;
                    }
                } else {
                    if positions.len() != self.topo.atoms.len() {
                        if let Ok(mut g) = shared.lock() {
                            g.running = false;
                            g.status_banner = format!(
                                "MSM conformation atom count {} mismatches active topology ({} atoms)",
                                positions.len(),
                                self.topo.atoms.len()
                            );
                        }
                        return false;
                    }
                    if let Ok(mut g) = shared.lock() {
                        g.running = false;
                        g.current_positions = positions.clone();
                        g.current_wu = wu_id;
                        g.current_potential_kcal = potential_kcal;
                        g.inspected_label = Some(label.clone());
                        if potential_kcal < g.best_potential_kcal {
                            g.best_positions = positions;
                            g.best_potential_kcal = potential_kcal;
                            g.best_rg_angstrom = rg_angstrom;
                            g.best_wu = wu_id;
                            if g.native_source == NativeReferenceSource::LowestEnergyFold {
                                g.native_pairs =
                                    native_ca_contact_pairs(&self.topo, &g.best_positions, 8.5);
                            }
                        }
                        g.status_banner = format!(
                            "Inspecting MSM Conformation {} ({}) | V_pot={:.2} kcal/mol | Rg={:.2} A",
                            label, wu_id, potential_kcal, rg_angstrom
                        );
                    }
                }
            }
            SandboxCommand::Shutdown => return true,
        }
        false
    }

    pub(crate) fn step_once(&mut self, shared: &Arc<Mutex<LiveFoldingState>>) {
        let t0 = Instant::now();
        let rung_outputs = step_mremd_rungs_batch(
            &self.topo,
            self.gpu_engine.as_ref(),
            &self.seed_coords,
            &mut self.checkpoints,
            self.steps_per_slice,
        );
        let total_steps_batch = (self.checkpoints.len() as u64) * self.steps_per_slice;

        let mut swap_updates: Vec<(usize, bool)> = Vec::new();
        let start_pair = if self.even_phase { 0usize } else { 1usize };
        self.even_phase = !self.even_phase;

        let mut k = start_pair;
        while k + 1 < self.checkpoints.len() {
            let (left, right) = self.checkpoints.split_at_mut(k + 1);
            let accepted =
                try_mremd_swap_checkpoints(&mut left[k], &mut right[0], &mut self.swap_rng);
            swap_updates.push((k, accepted));
            k += 2;
        }

        let elapsed_s = t0.elapsed().as_secs_f64().max(1e-6);
        let sim_ns = (total_steps_batch as f64) * 1.5e-6;
        let ns_per_day = sim_ns * (86_400.0 / elapsed_s);

        let mut coldest_idx = 0usize;
        for i in 1..self.checkpoints.len() {
            if self.checkpoints[i].temperature_kelvin
                < self.checkpoints[coldest_idx].temperature_kelvin
            {
                coldest_idx = i;
            }
        }
        if self.cold_positions_history.len() >= MAX_PLOT_HISTORY {
            self.cold_positions_history.pop_front();
        }
        self.cold_positions_history
            .push_back(self.checkpoints[coldest_idx].positions_angstrom.clone());

        if let Ok(mut g) = shared.lock() {
            g.instantaneous_ns_per_day = ns_per_day;
            g.total_generations_done += self.checkpoints.len() as u64;
            g.current_positions = self.checkpoints[coldest_idx].positions_angstrom.clone();
            g.current_wu = self.checkpoints[coldest_idx].wu_id;
            g.current_potential_kcal = self.checkpoints[coldest_idx].potential_energy_kcal;
            g.inspected_label = None;

            let mut best_updated = false;
            for out in rung_outputs {
                if out.min_potential_kcal < g.best_potential_kcal {
                    g.best_potential_kcal = out.min_potential_kcal;
                    g.best_positions = out.min_energy_positions;
                    g.best_rg_angstrom = out.frame.rg_angstrom;
                    g.best_sasa_angstrom2 = out.frame.sasa_angstrom2;
                    g.best_wu = out.frame.wu_id;
                    best_updated = true;
                }
                if g.msm_frames.len() >= MAX_MSM_ENSEMBLE_FRAMES {
                    g.msm_frames.pop_front();
                }
                g.msm_frames.push_back(out.frame);
            }

            if best_updated && g.native_source == NativeReferenceSource::LowestEnergyFold {
                let pairs = native_ca_contact_pairs(&self.topo, &g.best_positions, 8.5);
                g.native_pairs = pairs;
                g.q_history.clear();
                let limit = self.cold_positions_history.len().saturating_sub(1);
                for pos in self.cold_positions_history.iter().take(limit) {
                    let q_val = fraction_native_contacts(&g.native_pairs, pos, 8.8);
                    push_bounded(&mut g.q_history, q_val);
                }
            }

            for (pair_idx, acc) in swap_updates {
                if let Some(slot) = g.swap_pair_stats.get_mut(pair_idx) {
                    slot.1 += 1;
                    if acc {
                        slot.0 += 1;
                    }
                }
            }

            for (idx, cp) in self.checkpoints.iter().enumerate() {
                let rmsd = compute_ca_rmsd(&self.topo, &self.seed_coords, &cp.positions_angstrom);
                if let Some(slot) = g.rungs.get_mut(idx) {
                    slot.temperature_kelvin = cp.temperature_kelvin;
                    slot.generation = cp.wu_id.generation;
                    slot.cumulative_steps = cp.cumulative_steps;
                    slot.potential_kcal = cp.potential_energy_kcal;
                    slot.rg_angstrom = cp.rg_angstrom;
                    slot.rmsd_seed_ang = rmsd;
                }
            }

            let cold_pot = self.checkpoints[coldest_idx].potential_energy_kcal;
            let cold_rg = self.checkpoints[coldest_idx].rg_angstrom;
            let cold_rmsd = compute_ca_rmsd(
                &self.topo,
                &self.seed_coords,
                &self.checkpoints[coldest_idx].positions_angstrom,
            );
            let cold_q = fraction_native_contacts(
                &g.native_pairs,
                &self.checkpoints[coldest_idx].positions_angstrom,
                8.8,
            );
            let best_pot = g.best_potential_kcal;

            push_bounded(&mut g.potential_history, cold_pot);
            push_bounded(&mut g.min_potential_history, best_pot);
            push_bounded(&mut g.rg_history, cold_rg);
            push_bounded(&mut g.rmsd_history, cold_rmsd);
            push_bounded(&mut g.q_history, cold_q);
        }
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
            let mut worker = WorkerSimState::new(initial_preset, initial_use_gpu);

            loop {
                while let Ok(cmd) = rx.try_recv() {
                    if worker.handle_command(cmd, &shared) {
                        return;
                    }
                }

                if !worker.running && worker.pending_burst_rounds == 0 {
                    std::thread::sleep(Duration::from_millis(40));
                    continue;
                }
                if worker.pending_burst_rounds > 0 {
                    worker.pending_burst_rounds -= 1;
                }

                worker.step_once(&shared);
            }
        })
        .ok();

    tx
}

#[derive(Debug, Clone)]
pub(crate) struct RungStepOutput {
    pub frame: ConformationalFrame,
    pub min_potential_kcal: f64,
    pub min_energy_positions: Vec<[f32; 3]>,
}

/// Advances all MREMD temperature rungs for one slice.
/// Parallelizes across rungs via `rayon::par_iter_mut()` ONLY when `gpu_engine.is_none()`
/// because `GpuForceEngine` holds single shared staging buffers; GPU mode executes sequentially.
pub(crate) fn step_mremd_rungs_batch(
    topo: &ProteinTopology,
    gpu_engine: Option<&GpuForceEngine>,
    seed_coords: &[[f32; 3]],
    checkpoints: &mut [TrajectoryCheckpoint],
    steps_per_slice: u64,
) -> Vec<RungStepOutput> {
    let run_one = |r_idx: usize, cp: &mut TrajectoryCheckpoint, gpu: Option<&GpuForceEngine>| {
        let next_gen = cp.wu_id.generation + 1;
        let next_wu = WorkUnitId::new(1, r_idx as u32, 0, next_gen);
        let cfg = LangevinConfig {
            dt_fs: 1.5,
            friction_per_ps: 1.0,
            temperature_kelvin: cp.temperature_kelvin,
            steps: steps_per_slice,
            sample_stride: (steps_per_slice / 5).max(1),
        };
        let summary = run_generation_chunk_with_gpu(topo, gpu, cp, next_wu, &cfg);
        let rmsd_seed =
            compute_ca_rmsd(topo, seed_coords, &summary.checkpoint.positions_angstrom);
        let frame = ConformationalFrame {
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
        };
        *cp = summary.checkpoint;
        RungStepOutput {
            frame,
            min_potential_kcal: summary.min_potential_kcal,
            min_energy_positions: summary.min_energy_positions,
        }
    };

    if let Some(eng) = gpu_engine {
        checkpoints
            .iter_mut()
            .enumerate()
            .map(|(r_idx, cp)| run_one(r_idx, cp, Some(eng)))
            .collect()
    } else {
        checkpoints
            .par_iter_mut()
            .enumerate()
            .map(|(r_idx, cp)| run_one(r_idx, cp, None))
            .collect()
    }
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
    init_replicas_from_topology_and_seed(topo, seed_coords, use_gpu)
}

fn init_replicas_from_topology_and_seed(
    topo: ProteinTopology,
    seed_coords: Vec<[f32; 3]>,
    use_gpu: bool,
) -> (
    ProteinTopology,
    Vec<TrajectoryCheckpoint>,
    Option<GpuForceEngine>,
    Vec<[f32; 3]>,
) {
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

fn backend_label_for(use_gpu: bool, gpu_engine: &Option<GpuForceEngine>) -> String {
    match gpu_engine {
        Some(eng) => format!(
            "wgpu {} ({})",
            eng.adapter_summary().name,
            eng.adapter_summary().backend
        ),
        None if use_gpu => "rayon CPU (no GPU ICD)".to_string(),
        None => "rayon Multi-Core CPU".to_string(),
    }
}

fn ensure_matching_positions(
    topo: &mut ProteinTopology,
    positions: Vec<[f32; 3]>,
) -> Result<Vec<[f32; 3]>, String> {
    if !topo.atoms.is_empty() && positions.len() == topo.atoms.len() {
        return Ok(positions);
    }
    if topo.sequence_one_letter.trim().is_empty() {
        return Err(format!(
            "Coordinate count {} mismatches topology {} ({} atoms) and sequence is empty",
            positions.len(),
            topo.name,
            topo.atoms.len()
        ));
    }
    let (regen_topo, mut regen_coords) =
        ProteinTopology::build_polypeptide_from_sequence(&topo.name, &topo.sequence_one_letter);
    if regen_topo.atoms.is_empty() {
        return Err(format!(
            "Failed to regenerate coordinates for topology {}",
            topo.name
        ));
    }
    minimize_steric_clashes(&regen_topo, &mut regen_coords, 60);
    if regen_coords.len() != topo.atoms.len() {
        *topo = regen_topo;
    }
    Ok(regen_coords)
}

#[allow(clippy::too_many_arguments)]
fn build_loaded_state(
    display_name: String,
    topo: &ProteinTopology,
    seed_coords: &[[f32; 3]],
    seed_pot: f64,
    seed_rg: f64,
    active_positions: &[[f32; 3]],
    active_wu: WorkUnitId,
    active_pot: f64,
    active_rg: f64,
    active_sasa: f64,
    inspected_label: Option<String>,
    checkpoints: &[TrajectoryCheckpoint],
    use_gpu: bool,
    steps_per_slice: u64,
    active_backend_label: String,
    status_banner: String,
) -> LiveFoldingState {
    let native_pairs = native_ca_contact_pairs(topo, active_positions, 8.5);
    let q0 = fraction_native_contacts(&native_pairs, active_positions, 8.8);
    let rmsd0 = compute_ca_rmsd(topo, seed_coords, active_positions);

    let mut rungs = Vec::with_capacity(checkpoints.len());
    let mut msm_frames = VecDeque::with_capacity(MAX_MSM_ENSEMBLE_FRAMES);
    for (idx, cp) in checkpoints.iter().enumerate() {
        let rmsd_seed = compute_ca_rmsd(topo, seed_coords, &cp.positions_angstrom);
        rungs.push(RungSnapshot {
            rung_idx: idx,
            temperature_kelvin: cp.temperature_kelvin,
            generation: cp.wu_id.generation,
            cumulative_steps: cp.cumulative_steps,
            potential_kcal: cp.potential_energy_kcal,
            rg_angstrom: cp.rg_angstrom,
            rmsd_seed_ang: rmsd_seed,
        });
        msm_frames.push_back(ConformationalFrame {
            frame_idx: idx,
            wu_id: cp.wu_id,
            chain_step: cp.wu_id.generation,
            checkpoint_blake3: [0u8; 32],
            temperature_kelvin: cp.temperature_kelvin,
            potential_energy_kcal: cp.potential_energy_kcal,
            rg_angstrom: cp.rg_angstrom,
            sasa_angstrom2: cp.sasa_angstrom2,
            rmsd_to_seed_ang: rmsd_seed,
            rmsd_to_lowest_energy_ang: 0.0,
            positions_angstrom: cp.positions_angstrom.clone(),
        });
    }

    let mut pot_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
    pot_hist.push_back(active_pot);
    let mut min_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
    min_hist.push_back(active_pot);
    let mut rg_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
    rg_hist.push_back(active_rg);
    let mut rmsd_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
    rmsd_hist.push_back(rmsd0);
    let mut q_hist = VecDeque::with_capacity(MAX_PLOT_HISTORY);
    q_hist.push_back(q0);

    LiveFoldingState {
        topology: Arc::new(topo.clone()),
        target_display_name: display_name,
        native_source: NativeReferenceSource::LowestEnergyFold,
        native_pairs,
        seed_positions: seed_coords.to_vec(),
        seed_potential_kcal: seed_pot,
        seed_rg_angstrom: seed_rg,
        current_positions: active_positions.to_vec(),
        current_wu: active_wu,
        current_potential_kcal: active_pot,
        inspected_label,
        best_positions: active_positions.to_vec(),
        best_potential_kcal: active_pot,
        best_rg_angstrom: active_rg,
        best_sasa_angstrom2: active_sasa,
        best_wu: active_wu,
        running: false,
        use_gpu,
        steps_per_slice_active: steps_per_slice,
        active_backend_label,
        gpu_adapter: probe_gpu_adapter(),
        instantaneous_ns_per_day: 0.0,
        total_generations_done: 0,
        rungs,
        swap_pair_stats: vec![(0, 0); checkpoints.len().saturating_sub(1)],
        msm_frames,
        potential_history: pot_hist,
        min_potential_history: min_hist,
        rg_history: rg_hist,
        rmsd_history: rmsd_hist,
        q_history: q_hist,
        status_banner,
    }
}

fn load_vault_best_conformation(
    data_dir: &Path,
    project_id: u32,
) -> Result<(ProteinTopology, WorkUnitId, TrajectoryCheckpoint, Option<TrajectoryCheckpoint>), String> {
    let store = FoldStore::open(data_dir).map_err(|e| e.to_string())?;
    let (manifest, topo) = store.load_project(project_id).map_err(|e| e.to_string())?;
    let stats = store
        .collect_stats(Some(project_id))
        .map_err(|e| e.to_string())?;

    let init_cp = if manifest.initial_coords_blake3 != [0u8; 32] {
        store.get_checkpoint(&manifest.initial_coords_blake3).ok()
    } else {
        None
    };

    if let Some((best_wu, _best_e, best_hash)) = stats.lowest_potential_kcal {
        let cp = store.get_checkpoint(&best_hash).map_err(|e| e.to_string())?;
        Ok((topo, best_wu, cp, init_cp))
    } else {
        let cp = store
            .get_checkpoint(&manifest.initial_coords_blake3)
            .map_err(|e| e.to_string())?;
        Ok((topo, cp.wu_id, cp, init_cp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fold_proto::ProjectManifest;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDirGuard(std::path::PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn make_temp_vault_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "beofold_sandbox_{}_{}_{}",
            tag,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault dir");
        dir
    }

    #[test]
    fn cpu_mremd_par_iter_mut_is_deterministic_across_runs() {
        let (topo_a, mut cps_a, _, seed_a) = init_worker_replicas(TargetPreset::TrpCage1L2Y, false);
        let (topo_b, mut cps_b, _, seed_b) = init_worker_replicas(TargetPreset::TrpCage1L2Y, false);

        let out_a = step_mremd_rungs_batch(&topo_a, None, &seed_a, &mut cps_a, 50);
        let out_b = step_mremd_rungs_batch(&topo_b, None, &seed_b, &mut cps_b, 50);

        assert_eq!(out_a.len(), 4);
        assert_eq!(out_b.len(), 4);
        for (ra, rb) in out_a.iter().zip(out_b.iter()) {
            assert_eq!(ra.frame.wu_id, rb.frame.wu_id);
            assert!(
                (ra.frame.potential_energy_kcal - rb.frame.potential_energy_kcal).abs() < 1e-12
            );
            assert!((ra.min_potential_kcal - rb.min_potential_kcal).abs() < 1e-12);
        }
    }

    #[test]
    fn live_state_tracks_pdb_crystal_vs_lowest_energy_native_q_reference() {
        let (topo, coords) = ProteinTopology::build_trp_cage_seed();
        let pdb_state = LiveFoldingState::new_from_topology(
            "1L2Y.pdb".into(),
            topo.clone(),
            coords.clone(),
            true,
            false,
        );
        assert_eq!(pdb_state.native_source, NativeReferenceSource::PdbCrystal);
        assert!(!pdb_state.native_pairs.is_empty());
        assert_eq!(pdb_state.q_history.len(), 1);
        assert!(pdb_state.q_history[0] > 0.5 && pdb_state.q_history[0] <= 1.0);

        let seq_state = LiveFoldingState::new_from_topology(
            "CustomSeq".into(),
            topo,
            coords,
            false,
            false,
        );
        assert_eq!(seq_state.native_source, NativeReferenceSource::LowestEnergyFold);
        assert_eq!(seq_state.q_history.len(), 1);
        assert!((seq_state.q_history[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn load_from_vault_resizes_replicas_and_runs_mremd_without_panic() {
        let shared = Arc::new(Mutex::new(LiveFoldingState::new_preset(
            TargetPreset::TrpCage1L2Y,
            false,
        )));
        let mut worker = WorkerSimState::new(TargetPreset::TrpCage1L2Y, false);
        worker.steps_per_slice = 25;
        let trp_atoms = worker.topo.atoms.len();
        worker.step_once(&shared);

        // Seed a vault with Villin_HP35 (different atom count from TrpCage_1L2Y)
        let vault_dir = make_temp_vault_dir("vault_switch");
        let _guard = TempDirGuard(vault_dir.clone());
        let (villin_topo, mut villin_seed) = ProteinTopology::build_villin_benchmark_seed();
        assert_ne!(villin_topo.atoms.len(), trp_atoms);
        minimize_steric_clashes(&villin_topo, &mut villin_seed, 60);
        let cp0 = initialize_clone_checkpoint(
            &villin_topo,
            &villin_seed,
            WorkUnitId::new(1, 0, 0, 0),
            285.0,
            b"test-vault",
        );
        let manifest = ProjectManifest {
            project_id: 1,
            name: villin_topo.name.clone(),
            sequence: villin_topo.sequence_one_letter.clone(),
            topology_blake3: [0u8; 32],
            initial_coords_blake3: [0u8; 32],
            atom_count: villin_topo.atoms.len() as u32,
            residue_count: villin_topo.residue_count,
            temp_ladder_kelvin: vec![285.0, 318.0, 356.0, 398.0],
            dt_fs: 1.5,
            friction_per_ps: 1.0,
            steps_per_gen: 50,
            num_clones_per_run: 1,
            max_generations: 1,
        };
        let store = FoldStore::open(&vault_dir).expect("open temp FoldStore");
        store
            .seed_project(manifest, &villin_topo, &[cp0])
            .expect("seed Villin project");
        drop(store);

        let shutdown = worker.handle_command(
            SandboxCommand::LoadFromVault {
                data_dir: vault_dir,
                project_id: 1,
            },
            &shared,
        );
        assert!(!shutdown);

        let villin_atoms = villin_topo.atoms.len();
        assert_eq!(worker.topo.atoms.len(), villin_atoms);
        assert_eq!(worker.seed_coords.len(), villin_atoms);
        assert!(!worker.checkpoints.is_empty());
        assert!(worker
            .checkpoints
            .iter()
            .all(|c| c.positions_angstrom.len() == villin_atoms));
        {
            let g = shared.lock().unwrap();
            assert_eq!(g.topology.atoms.len(), villin_atoms);
            assert_eq!(g.seed_positions.len(), villin_atoms);
            assert_eq!(g.best_positions.len(), villin_atoms);
            assert_eq!(g.current_positions.len(), villin_atoms);
            assert_eq!(g.q_history.len(), 1);
            assert_eq!(g.msm_frames.len(), worker.checkpoints.len());
        }

        // Subsequent MREMD stepping must succeed cleanly with zero panic
        let batch_out = step_mremd_rungs_batch(
            &worker.topo,
            worker.gpu_engine.as_ref(),
            &worker.seed_coords,
            &mut worker.checkpoints,
            25,
        );
        assert_eq!(batch_out.len(), 4);
        worker.step_once(&shared);
        {
            let g = shared.lock().unwrap();
            assert_eq!(g.q_history.len(), 2);
            assert_eq!(g.total_generations_done, 4);
        }
    }

    #[test]
    fn load_msm_conformation_handles_topology_switch_and_preserves_lowest_energy_best() {
        let shared = Arc::new(Mutex::new(LiveFoldingState::new_preset(
            TargetPreset::TrpCage1L2Y,
            false,
        )));
        let mut worker = WorkerSimState::new(TargetPreset::TrpCage1L2Y, false);
        worker.steps_per_slice = 25;

        // Establish a low-energy best conformation on TrpCage
        let original_best_pos;
        let original_best_pot;
        {
            let mut g = shared.lock().unwrap();
            g.best_potential_kcal = -125.0;
            original_best_pot = g.best_potential_kcal;
            original_best_pos = g.best_positions.clone();
        }

        // 1. Same topology, higher-energy MSM conformation: must inspect in current_positions
        // without overwriting best_potential_kcal / best_positions.
        let mut stretched = original_best_pos.clone();
        for (idx, p) in stretched.iter_mut().enumerate() {
            p[0] += (idx as f32) * 0.5;
        }
        let trp_topo = Arc::new(worker.topo.clone());
        worker.handle_command(
            SandboxCommand::LoadMsmConformation {
                topology: trp_topo,
                positions: stretched.clone(),
                wu_id: WorkUnitId::new(1, 2, 0, 9),
                potential_kcal: 48.5,
                rg_angstrom: 14.2,
                label: "M2 (Unfolded)".into(),
            },
            &shared,
        );
        {
            let g = shared.lock().unwrap();
            assert_eq!(g.best_potential_kcal, original_best_pot);
            assert_eq!(g.best_positions, original_best_pos);
            assert_eq!(g.current_positions, stretched);
            assert_eq!(g.inspected_label.as_deref(), Some("M2 (Unfolded)"));
        }

        // 2. Different topology (Villin_HP35): must cleanly re-initialize replicas & state so
        // MREMD executes without panicking on atom-count mismatch.
        let (villin_topo, mut villin_pos) = ProteinTopology::build_villin_benchmark_seed();
        minimize_steric_clashes(&villin_topo, &mut villin_pos, 60);
        let villin_atoms = villin_topo.atoms.len();
        worker.handle_command(
            SandboxCommand::LoadMsmConformation {
                topology: Arc::new(villin_topo),
                positions: villin_pos.clone(),
                wu_id: WorkUnitId::new(2, 0, 0, 3),
                potential_kcal: -32.0,
                rg_angstrom: 11.8,
                label: "M0 (Villin)".into(),
            },
            &shared,
        );
        assert_eq!(worker.topo.atoms.len(), villin_atoms);
        assert_eq!(worker.seed_coords.len(), villin_atoms);
        assert_eq!(worker.checkpoints.len(), 4);
        assert!(worker
            .checkpoints
            .iter()
            .all(|c| c.positions_angstrom.len() == villin_atoms));
        {
            let g = shared.lock().unwrap();
            assert_eq!(g.topology.atoms.len(), villin_atoms);
            assert_eq!(g.current_positions, villin_pos);
            assert_eq!(g.msm_frames.len(), 4);
        }

        let batch_out = step_mremd_rungs_batch(
            &worker.topo,
            worker.gpu_engine.as_ref(),
            &worker.seed_coords,
            &mut worker.checkpoints,
            25,
        );
        assert_eq!(batch_out.len(), 4);
        worker.step_once(&shared);
        assert!(shared.lock().unwrap().inspected_label.is_none());
    }

    #[test]
    fn ensure_matching_positions_regenerates_on_length_mismatch() {
        let (mut topo, _coords) = ProteinTopology::build_villin_benchmark_seed();
        let expected = topo.atoms.len();
        let bad_coords = vec![[0.0f32; 3]; 5];
        let fixed = ensure_matching_positions(&mut topo, bad_coords)
            .expect("should regenerate coordinates for valid sequence");
        assert_eq!(fixed.len(), expected);
        assert_eq!(topo.atoms.len(), expected);
    }
}
