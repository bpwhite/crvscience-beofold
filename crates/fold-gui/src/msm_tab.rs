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

use crate::live_sandbox::{LiveFoldingState, SandboxCommand};
use crate::viewport3d::ConformationViewport;
use egui::{
    Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Ui,
    Vec2,
};
use fold_msm::{
    cluster_conformations_kmedoids, compute_2d_pmf_grid, compute_pairwise_rmsd_matrix,
    estimate_reversible_msm, export_msm_bundle, extract_project_frames, ConformationalFrame,
    MicrostateClustering, PmfGrid2D, ReversibleMsmModel,
};
use fold_physics::{compute_ca_rmsd, ProteinTopology};
use fold_proto::ProjectManifest;
use fold_storage::FoldStore;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};

/// Immutable snapshot produced asynchronously by a background MSM estimation job.
#[derive(Debug, Clone)]
pub struct ComputedMsmBundle {
    pub source_label: String,
    pub manifest: ProjectManifest,
    pub topology: Arc<ProteinTopology>,
    pub frames: Vec<ConformationalFrame>,
    pub clustering: MicrostateClustering,
    pub msm: ReversibleMsmModel,
    pub pmf_grid: PmfGrid2D,
}

#[derive(Debug, Default)]
struct AsyncMsmJobSlot {
    computing: bool,
    last_error: Option<String>,
    latest_bundle: Option<ComputedMsmBundle>,
    revision: u64,
}

/// Interactive 4th Supervisor Tab: 2D `(C-alpha RMSD, R_g)` Free-Energy Surface, Reversible Markov
/// Flux Network, Implied Timescales, and Click-to-Load 3D Macrostate/Microstate Inspector.
pub struct MsmLandscapeTab {
    pub num_microstates: usize,
    pub num_macrostates: usize,
    pub lag_generations: u32,
    pub ref_temp_kelvin: f64,
    pub cold_rung_filter_only: bool,
    pub project_id_input: u32,
    pub show_flux_edges: bool,
    pub min_flux_threshold: f32,
    pub export_bundle_dir: String,
    pub single_pdb_out: String,
    pub status_note: Option<String>,

    job_slot: Arc<Mutex<AsyncMsmJobSlot>>,
    seen_revision: u64,
    active_bundle: Option<ComputedMsmBundle>,
    selected_macrostate: Option<usize>,
    selected_microstate: Option<usize>,
    selected_frame_idx: Option<usize>,
    preview_viewport: ConformationViewport,
}

impl Default for MsmLandscapeTab {
    fn default() -> Self {
        let mut vp = ConformationViewport::default();
        vp.zoom = 1.12;
        Self {
            num_microstates: 8,
            num_macrostates: 3,
            lag_generations: 1,
            ref_temp_kelvin: 300.0,
            cold_rung_filter_only: true,
            project_id_input: 1,
            show_flux_edges: true,
            min_flux_threshold: 0.005,
            export_bundle_dir: "./beofold_msm_gui_export".to_string(),
            single_pdb_out: "./selected_macrostate.pdb".to_string(),
            status_note: None,
            job_slot: Arc::new(Mutex::new(AsyncMsmJobSlot::default())),
            seen_revision: 0,
            active_bundle: None,
            selected_macrostate: None,
            selected_microstate: None,
            selected_frame_idx: None,
            preview_viewport: vp,
        }
    }
}

impl MsmLandscapeTab {
    pub fn is_animating(&self) -> bool {
        let busy = self
            .job_slot
            .lock()
            .map(|g| g.computing)
            .unwrap_or(false);
        busy || (self.active_bundle.is_some() && self.preview_viewport.auto_spin)
    }

    fn spawn_live_ensemble_msm(&mut self, live: &LiveFoldingState) {
        let raw_frames: Vec<ConformationalFrame> = live.msm_frames.iter().cloned().collect();
        if raw_frames.len() < 8 {
            self.status_note = Some(format!(
                "Live studio currently has {} conformations (need >= 8). Click 'Sample 24 Frames + Build MSM' or run Live MREMD for a few seconds.",
                raw_frames.len()
            ));
            return;
        }

        let topo = live.topology.clone();
        let steps_per_gen = live.steps_per_slice_active.max(50);
        let k_req = self.num_microstates;
        let m_req = self.num_macrostates;
        let lag = self.lag_generations.max(1);
        let ref_t = self.ref_temp_kelvin;
        let cold_only = self.cold_rung_filter_only;
        let slot = self.job_slot.clone();
        let err_slot = slot.clone();

        if let Ok(mut g) = slot.lock() {
            g.computing = true;
            g.last_error = None;
        }

        if let Err(e) = std::thread::Builder::new()
            .name("beofold-gui-msm-live".into())
            .spawn(move || {
                let res = build_msm_from_live_frames(
                    topo,
                    raw_frames,
                    steps_per_gen,
                    k_req,
                    m_req,
                    lag,
                    ref_t,
                    cold_only,
                );
                if let Ok(mut g) = slot.lock() {
                    g.computing = false;
                    match res {
                        Ok(bundle) => {
                            g.latest_bundle = Some(bundle);
                            g.revision += 1;
                        }
                        Err(err) => {
                            g.last_error = Some(err);
                        }
                    }
                }
            })
        {
            if let Ok(mut g) = err_slot.lock() {
                g.computing = false;
                g.last_error = Some(format!("Failed to spawn MSM worker: {e}"));
            }
        }
    }

    pub(crate) fn compute_live_msm_blocking(
        &mut self,
        live: &LiveFoldingState,
        sandbox_tx: &mpsc::Sender<SandboxCommand>,
    ) -> Result<(), String> {
        let raw_frames: Vec<ConformationalFrame> = live.msm_frames.iter().cloned().collect();
        if raw_frames.len() < 8 {
            return Err(format!(
                "Need >= 8 conformations for live MSM, found {}",
                raw_frames.len()
            ));
        }
        let topo = live.topology.clone();
        let steps_per_gen = live.steps_per_slice_active.max(50);
        let bundle = build_msm_from_live_frames(
            topo,
            raw_frames,
            steps_per_gen,
            self.num_microstates,
            self.num_macrostates,
            self.lag_generations.max(1),
            self.ref_temp_kelvin,
            self.cold_rung_filter_only,
        )?;
        self.active_bundle = Some(bundle);
        self.select_macrostate_and_sync(0, sandbox_tx);
        Ok(())
    }

    fn spawn_vault_msm(&mut self, vault_dir: PathBuf) {
        let pid = self.project_id_input;
        let k_req = self.num_microstates;
        let m_req = self.num_macrostates;
        let lag = self.lag_generations.max(1);
        let ref_t = self.ref_temp_kelvin;
        let cold_only = self.cold_rung_filter_only;
        let slot = self.job_slot.clone();
        let err_slot = slot.clone();

        if let Ok(mut g) = slot.lock() {
            g.computing = true;
            g.last_error = None;
        }

        if let Err(e) = std::thread::Builder::new()
            .name("beofold-gui-msm-vault".into())
            .spawn(move || {
                let res = build_msm_from_redb_vault(
                    &vault_dir,
                    pid,
                    k_req,
                    m_req,
                    lag,
                    ref_t,
                    cold_only,
                );
                if let Ok(mut g) = slot.lock() {
                    g.computing = false;
                    match res {
                        Ok(bundle) => {
                            g.latest_bundle = Some(bundle);
                            g.revision += 1;
                        }
                        Err(err) => {
                            g.last_error = Some(err);
                        }
                    }
                }
            })
        {
            if let Ok(mut g) = err_slot.lock() {
                g.computing = false;
                g.last_error = Some(format!("Failed to spawn vault MSM worker: {e}"));
            }
        }
    }

    fn select_macrostate_and_sync(
        &mut self,
        macro_idx: usize,
        sandbox_tx: &mpsc::Sender<SandboxCommand>,
    ) {
        let Some(ref bundle) = self.active_bundle else {
            return;
        };
        let Some(mac) = bundle.msm.macrostates.get(macro_idx) else {
            return;
        };
        let rep_idx = mac.representative_frame_idx;
        let rep_frame = &bundle.frames[rep_idx];
        let orig_micro = bundle.clustering.frame_assignments[rep_idx];
        let active_micro = bundle
            .msm
            .orig_to_active
            .get(orig_micro)
            .copied()
            .flatten()
            .or_else(|| mac.member_microstates.first().copied());

        self.selected_macrostate = Some(macro_idx);
        self.selected_microstate = active_micro;
        self.selected_frame_idx = Some(rep_idx);
        self.single_pdb_out = format!("./macrostate_{}_{}.pdb", mac.macro_id, mac.biophysical_label);

        let _ = sandbox_tx.send(SandboxCommand::LoadMsmConformation {
            topology: bundle.topology.clone(),
            positions: rep_frame.positions_angstrom.clone(),
            wu_id: rep_frame.wu_id,
            potential_kcal: rep_frame.potential_energy_kcal,
            rg_angstrom: rep_frame.rg_angstrom,
            label: format!("M{} ({})", mac.macro_id, mac.biophysical_label),
        });
    }

    fn select_microstate_and_sync(
        &mut self,
        micro_idx: usize,
        sandbox_tx: &mpsc::Sender<SandboxCommand>,
    ) {
        let Some(ref bundle) = self.active_bundle else {
            return;
        };
        if micro_idx >= bundle.msm.num_microstates {
            return;
        }
        let orig_micro = bundle.msm.active_microstates[micro_idx];
        let medoid_frame_idx = bundle.clustering.medoid_frame_indices[orig_micro];
        let med_frame = &bundle.frames[medoid_frame_idx];
        let macro_id = bundle.msm.microstate_to_macrostate[micro_idx];

        self.selected_microstate = Some(micro_idx);
        self.selected_macrostate = Some(macro_id);
        self.selected_frame_idx = Some(medoid_frame_idx);
        self.single_pdb_out = format!("./microstate_S{}_M{}.pdb", micro_idx, macro_id);

        let _ = sandbox_tx.send(SandboxCommand::LoadMsmConformation {
            topology: bundle.topology.clone(),
            positions: med_frame.positions_angstrom.clone(),
            wu_id: med_frame.wu_id,
            potential_kcal: med_frame.potential_energy_kcal,
            rg_angstrom: med_frame.rg_angstrom,
            label: format!("Microstate S{} (M{})", micro_idx, macro_id),
        });
    }

    /// Renders the 4th Supervisor Tab. Returns `true` if the user clicked `"Open in Full 3D Studio"`
    /// to jump directly to Tab 1 with the selected macrostate/microstate conformation loaded.
    pub fn render(
        &mut self,
        ctx: &egui::Context,
        live: &LiveFoldingState,
        vault_dir_input: &mut String,
        sandbox_tx: &mpsc::Sender<SandboxCommand>,
    ) -> bool {
        let mut jump_to_3d_studio = false;

        // Poll background MSM job completion
        let mut newly_arrived = false;
        let (is_computing, job_err) = if let Ok(guard) = self.job_slot.lock() {
            if guard.revision > self.seen_revision {
                self.seen_revision = guard.revision;
                self.active_bundle = guard.latest_bundle.clone();
                newly_arrived = true;
            }
            (guard.computing, guard.last_error.clone())
        } else {
            (false, None)
        };

        if newly_arrived {
            self.select_macrostate_and_sync(0, sandbox_tx);
            self.status_note = Some("Reversible detailed-balance MSM & 2D PMF surface ready. Click any macrostate or S_i node to inspect its 3D fold.".into());
        }

        // 1. Left Control & PCCA+ Macrostate Selector Panel
        egui::SidePanel::left("msm_left_controls")
            .resizable(true)
            .default_width(355.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Markov State Model Builder");
                    ui.label(
                        RichText::new(format!(
                            "Live Studio Buffer: {} conformations",
                            live.msm_frames.len()
                        ))
                        .monospace()
                        .size(11.5)
                        .color(Color32::from_rgb(56, 189, 248)),
                    );
                    ui.add_space(4.0);

                    ui.add(
                        egui::Slider::new(&mut self.num_microstates, 4..=18)
                            .text("Microstates (K)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.num_macrostates, 2..=5)
                            .text("PCCA+ Macrostates (M)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.lag_generations, 1..=4)
                            .text("Lag (Generations)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.ref_temp_kelvin, 275.0..=380.0)
                            .text("Ref Temp (K)"),
                    );
                    ui.checkbox(
                        &mut self.cold_rung_filter_only,
                        "Cold-Rung Filter (T <= T_ref + 25 K)",
                    );

                    ui.add_space(6.0);
                    ui.add_enabled_ui(!is_computing, |ui| {
                        if ui
                            .button(RichText::new("Compute MSM from Live Studio").strong())
                            .clicked()
                        {
                            self.spawn_live_ensemble_msm(live);
                        }
                        if ui
                            .button("Sample 24 MREMD Frames (Burst)")
                            .on_hover_text(
                                "Runs 6 quick 4-rung MREMD slices in the background so you can build a rich MSM immediately.",
                            )
                            .clicked()
                        {
                            let _ = sandbox_tx.send(SandboxCommand::RunBurstRounds { rounds: 6 });
                            self.status_note = Some(
                                "Sampling 24 MREMD conformations across 4 rungs — click 'Compute MSM from Live Studio' in ~2s.".into(),
                            );
                        }

                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(vault_dir_input).desired_width(145.0),
                            );
                            if ui.button("Browse...").clicked() {
                                if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                    *vault_dir_input = folder.display().to_string();
                                }
                            }
                            if ui.button("Build from redb").clicked() {
                                self.spawn_vault_msm(PathBuf::from(vault_dir_input.trim()));
                            }
                        });
                    });

                    if is_computing {
                        ui.label(
                            RichText::new("Computing Kabsch RMSD matrix, Prinz-Noé MLE & Jacobi spectrum...")
                                .monospace()
                                .size(11.0)
                                .color(Color32::from_rgb(251, 191, 36)),
                        );
                    }
                    if let Some(err) = job_err {
                        ui.label(
                            RichText::new(format!("MSM Error: {}", err))
                                .monospace()
                                .size(11.0)
                                .color(Color32::from_rgb(248, 113, 113)),
                        );
                    }

                    ui.separator();
                    ui.checkbox(&mut self.show_flux_edges, "Overlay Equilibrium Flux Network");
                    if self.show_flux_edges {
                        ui.add(
                            egui::Slider::new(&mut self.min_flux_threshold, 0.001..=0.05)
                                .text("Min Flux (pi_i * T_ij)"),
                        );
                    }

                    if let Some(bundle_clone) = self.active_bundle.clone() {
                        ui.separator();
                        ui.heading("PCCA+ Metastable Macrostates");
                        ui.label(
                            RichText::new("Click any macrostate to load its 3D conformation:")
                                .size(11.5)
                                .color(Color32::from_rgb(148, 163, 184)),
                        );
                        ui.add_space(4.0);

                        let mut clicked_macro: Option<usize> = None;
                        for mac in &bundle_clone.msm.macrostates {
                            let is_sel = self.selected_macrostate == Some(mac.macro_id);
                            let dg = if mac.free_energy_kcal.abs() < 5e-4 {
                                0.0
                            } else {
                                mac.free_energy_kcal
                            };
                            let header = format!(
                                "M{} [{}]  pi={:.1}%  dG={:.2} kcal/mol",
                                mac.macro_id,
                                mac.biophysical_label,
                                mac.stationary_population * 100.0,
                                dg
                            );
                            let sub = format!(
                                "   p_fold={:.2} | Rg={:.2} A | RMSD_best={:.2} A | {} microstates",
                                mac.mean_committor,
                                mac.mean_rg_angstrom,
                                mac.mean_rmsd_best_ang,
                                mac.member_microstates.len()
                            );
                            let resp = ui.selectable_label(
                                is_sel,
                                RichText::new(format!("{}\n{}", header, sub))
                                    .monospace()
                                    .size(11.0),
                            );
                            if resp.clicked() {
                                clicked_macro = Some(mac.macro_id);
                            }
                            ui.add_space(3.0);
                        }
                        if let Some(m_id) = clicked_macro {
                            self.select_macrostate_and_sync(m_id, sandbox_tx);
                        }

                        ui.separator();
                        ui.heading("Export Landscape & PDB Bundle");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.export_bundle_dir)
                                    .desired_width(155.0),
                            );
                            if ui.button("Browse...").clicked() {
                                if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                    self.export_bundle_dir = folder.display().to_string();
                                }
                            }
                            if ui.button("Export All").clicked() {
                                let out = PathBuf::from(self.export_bundle_dir.trim());
                                match export_msm_bundle(
                                    &out,
                                    &bundle_clone.manifest,
                                    &bundle_clone.topology,
                                    &bundle_clone.frames,
                                    &bundle_clone.clustering,
                                    &bundle_clone.msm,
                                ) {
                                    Ok(art) => {
                                        self.status_note = Some(format!(
                                            "Exported SVG, JSON, CSVs & {} macrostate PDBs to {}",
                                            art.macrostate_pdb_paths.len(),
                                            out.display()
                                        ));
                                    }
                                    Err(e) => {
                                        self.status_note = Some(format!("Export error: {}", e));
                                    }
                                }
                            }
                        });
                    }

                    if let Some(ref note) = self.status_note {
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(note)
                                .monospace()
                                .size(11.0)
                                .color(Color32::from_rgb(52, 211, 153)),
                        );
                    }
                });
            });

        // 2. Right Panel: Interactive 3D Conformation Preview + Implied Relaxation Timescales
        egui::SidePanel::right("msm_right_inspector")
            .resizable(true)
            .default_width(410.0)
            .show(ctx, |ui| {
                if let Some(ref bundle) = self.active_bundle {
                    let frame_idx = self
                        .selected_frame_idx
                        .unwrap_or(0)
                        .min(bundle.frames.len().saturating_sub(1));
                    let sel_frame = &bundle.frames[frame_idx];
                    let orig_micro_id = bundle.clustering.frame_assignments[frame_idx];
                    let active_micro_opt = bundle
                        .msm
                        .orig_to_active
                        .get(orig_micro_id)
                        .copied()
                        .flatten();
                    let active_micro_label = active_micro_opt
                        .map(|a| format!("S{}", a))
                        .unwrap_or_else(|| format!("S_orig{}", orig_micro_id));
                    let macro_id = active_micro_opt
                        .map(|a| bundle.msm.microstate_to_macrostate[a])
                        .unwrap_or(0);
                    let p_fold_val = active_micro_opt
                        .map(|a| bundle.msm.folding_committor[a])
                        .unwrap_or(0.0);

                    ui.horizontal(|ui| {
                        ui.heading(format!(
                            "3D Inspector — M{} / {}",
                            macro_id, active_micro_label
                        ));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button(RichText::new("Open in Full 3D Studio").strong())
                                .clicked()
                            {
                                jump_to_3d_studio = true;
                            }
                        });
                    });

                    ui.label(
                        RichText::new(format!(
                            "WU {} | V_pot={:.2} kcal/mol | Rg={:.2} A | p_fold={:.2}",
                            sel_frame.wu_id,
                            sel_frame.potential_energy_kcal,
                            sel_frame.rg_angstrom,
                            p_fold_val
                        ))
                        .monospace()
                        .size(11.0)
                        .color(Color32::from_rgb(56, 189, 248)),
                    );

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.preview_viewport.auto_spin, "Auto-Spin");
                        if ui.button("Save .pdb...").clicked() {
                            let default_name = PathBuf::from(self.single_pdb_out.trim())
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("selected_state.pdb")
                                .to_string();
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("PDB Structure", &["pdb"])
                                .set_file_name(&default_name)
                                .save_file()
                            {
                                self.single_pdb_out = p.display().to_string();
                                match bundle.topology.write_pdb_snapshot(
                                    &sel_frame.positions_angstrom,
                                    &p,
                                    &format!("MSM Selected M{} {}", macro_id, active_micro_label),
                                    sel_frame.potential_energy_kcal,
                                ) {
                                    Ok(()) => {
                                        self.status_note = Some(format!(
                                            "Wrote 3D conformation to {}",
                                            p.display()
                                        ));
                                    }
                                    Err(e) => {
                                        self.status_note = Some(format!("PDB save error: {}", e));
                                    }
                                }
                            }
                        }
                    });

                    // Allocate top ~58% height to the interactive 3D Conformation Viewport
                    let top_h = (ui.available_height() * 0.56).max(220.0);
                    ui.allocate_ui(Vec2::new(ui.available_width(), top_h), |ui| {
                        self.preview_viewport.ui(
                            ui,
                            &bundle.topology,
                            &sel_frame.positions_angstrom,
                            &format!("M{} / {}", macro_id, active_micro_label),
                        );
                    });

                    ui.separator();
                    ui.heading("Implied Timescales & Reversibility");
                    ui.label(
                        RichText::new(format!(
                            "Transitions: {} | Detailed-balance err: {:.2e}",
                            bundle.msm.total_transitions, bundle.msm.detailed_balance_max_error
                        ))
                        .monospace()
                        .size(11.0),
                    );
                    ui.add_space(4.0);

                    let max_its = bundle
                        .msm
                        .implied_timescales_ps
                        .iter()
                        .take(5)
                        .cloned()
                        .fold(1.0f64, f64::max);
                    for (idx, &its) in bundle.msm.implied_timescales_ps.iter().take(5).enumerate() {
                        let lam = bundle
                            .msm
                            .eigenvalues
                            .get(idx + 1)
                            .copied()
                            .unwrap_or(0.0);
                        let frac = ((its / max_its) as f32).clamp(0.02, 1.0);
                        ui.add(egui::ProgressBar::new(frac).text(format!(
                            "t_{} = {:.2} ps  (lambda_{} = {:.4})",
                            idx + 2,
                            its,
                            idx + 2,
                            lam
                        )));
                    }
                } else {
                    ui.heading("3D Macrostate Conformation Inspector");
                    ui.add_space(8.0);
                    ui.label(
                        "Build an MSM using the left panel ('Compute MSM from Live Studio' or 'Build from redb') to inspect the 2D (RMSD, Rg) free-energy landscape and click metastable macrostates.",
                    );
                }
            });

        // 3. Center Panel: Interactive 2D (C-alpha RMSD, R_g) Free-Energy Surface + Markov Flux Network
        let mut clicked_micro_on_canvas: Option<usize> = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(ref bundle) = self.active_bundle else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new(
                            "No Markov State Model computed yet.\nClick 'Sample 24 MREMD Frames (Burst)' then 'Compute MSM from Live Studio' (or load a redb vault) on the left.",
                        )
                        .monospace()
                        .size(13.0)
                        .color(Color32::from_rgb(148, 163, 184)),
                    );
                });
                return;
            };

            clicked_micro_on_canvas = draw_interactive_pmf_and_flux_canvas(
                ui,
                bundle,
                self.show_flux_edges,
                self.min_flux_threshold as f64,
                self.selected_macrostate,
                self.selected_microstate,
            );
        });

        if let Some(m_idx) = clicked_micro_on_canvas {
            self.select_microstate_and_sync(m_idx, sandbox_tx);
        }

        jump_to_3d_studio
    }
}

fn draw_interactive_pmf_and_flux_canvas(
    ui: &mut Ui,
    bundle: &ComputedMsmBundle,
    show_flux: bool,
    min_flux: f64,
    selected_macro: Option<usize>,
    selected_micro: Option<usize>,
) -> Option<usize> {
    let avail = ui.available_size();
    let canvas_size = Vec2::new(avail.x.max(300.0), avail.y.max(280.0));
    let (response, painter) = ui.allocate_painter(canvas_size, Sense::click());
    let rect = response.rect;

    painter.rect_filled(rect, CornerRadius::same(8), Color32::from_rgb(12, 17, 25));
    painter.rect_stroke(
        rect,
        CornerRadius::same(8),
        Stroke::new(1.0_f32, Color32::from_rgb(40, 52, 70)),
        StrokeKind::Inside,
    );

    let pad_left = 64.0f32;
    let pad_right = 76.0f32;
    let pad_top = 58.0f32;
    let pad_bottom = 52.0f32;
    let plot_rect = Rect::from_min_max(
        Pos2::new(rect.left() + pad_left, rect.top() + pad_top),
        Pos2::new(
            (rect.right() - pad_right).max(rect.left() + pad_left + 120.0),
            (rect.bottom() - pad_bottom).max(rect.top() + pad_top + 120.0),
        ),
    );
    let circle_pad = (plot_rect.width().min(plot_rect.height()) * 0.10).clamp(22.0, 26.0);
    let data_rect = plot_rect.shrink(circle_pad);

    // Fill outer margin inside `plot_rect` with dark obsidian baseline so edge circles never clip
    painter.rect_filled(plot_rect, CornerRadius::same(0), dark_pmf_colormap(1.0));

    let grid = &bundle.pmf_grid;
    let cell_w = data_rect.width() / (grid.nx.max(1) as f32);
    let cell_h = data_rect.height() / (grid.ny.max(1) as f32);

    // 1. Render 2D Boltzmann Potential of Mean Force heatmap (`0 .. max_capped_kcal`) inside `data_rect`
    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            let fe = grid.free_energy_kcal[ix][iy];
            let norm = (fe / grid.max_capped_kcal.max(0.1)).clamp(0.0, 1.0) as f32;
            let fill = dark_pmf_colormap(norm);
            let min_pt = Pos2::new(
                data_rect.left() + (ix as f32) * cell_w,
                data_rect.bottom() - ((iy + 1) as f32) * cell_h,
            );
            let cell_rect =
                Rect::from_min_size(min_pt, Vec2::new(cell_w + 0.6, cell_h + 0.6));
            painter.rect_filled(cell_rect, CornerRadius::same(0), fill);
        }
    }

    painter.rect_stroke(
        plot_rect,
        CornerRadius::same(0),
        Stroke::new(1.2_f32, Color32::from_rgb(90, 108, 135)),
        StrokeKind::Inside,
    );

    // Colorbar legend on right margin
    let cb_x0 = plot_rect.right() + 14.0;
    let cb_w = 12.0;
    let cb_steps = 24usize;
    let step_h = plot_rect.height() / (cb_steps as f32);
    for s in 0..cb_steps {
        let frac = 1.0 - ((s as f32 + 0.5) / (cb_steps as f32));
        let col = dark_pmf_colormap(frac);
        let r = Rect::from_min_size(
            Pos2::new(cb_x0, plot_rect.top() + (s as f32) * step_h),
            Vec2::new(cb_w, step_h + 0.8),
        );
        painter.rect_filled(r, CornerRadius::same(0), col);
    }
    painter.text(
        Pos2::new(cb_x0 + cb_w + 4.0, plot_rect.top()),
        Align2::LEFT_TOP,
        format!("{:.1}", grid.max_capped_kcal),
        FontId::monospace(10.0),
        Color32::from_rgb(175, 190, 210),
    );
    painter.text(
        Pos2::new(cb_x0 + cb_w + 4.0, plot_rect.bottom() - 10.0),
        Align2::LEFT_TOP,
        "0.0",
        FontId::monospace(10.0),
        Color32::from_rgb(56, 189, 248),
    );
    painter.text(
        Pos2::new(cb_x0 - 2.0, plot_rect.top() - 16.0),
        Align2::LEFT_TOP,
        "kcal/mol",
        FontId::monospace(10.0),
        Color32::from_rgb(175, 190, 210),
    );

    let map_xy = |rmsd: f64, rg: f64| -> Pos2 {
        let fx = ((rmsd - grid.rmsd_min) / (grid.rmsd_max - grid.rmsd_min).max(1e-6))
            .clamp(0.0, 1.0) as f32;
        let fy = ((rg - grid.rg_min) / (grid.rg_max - grid.rg_min).max(1e-6))
            .clamp(0.0, 1.0) as f32;
        Pos2::new(
            data_rect.left() + fx * data_rect.width(),
            data_rect.bottom() - fy * data_rect.height(),
        )
    };

    let plot_painter = painter.with_clip_rect(plot_rect);

    // 2. Equilibrium Markov Flux Network (`pi_i * T_ij` directed from lower -> higher committor `p_fold`)
    if show_flux {
        for i in 0..bundle.msm.num_microstates {
            let orig_i = bundle.msm.active_microstates[i];
            for j in (i + 1)..bundle.msm.num_microstates {
                let flux = bundle.msm.stationary_pi[i] * bundle.msm.transition_matrix[i][j];
                if flux < min_flux {
                    continue;
                }
                let orig_j = bundle.msm.active_microstates[j];
                let fi = &bundle.frames[bundle.clustering.medoid_frame_indices[orig_i]];
                let fj = &bundle.frames[bundle.clustering.medoid_frame_indices[orig_j]];
                let pi_scr = map_xy(fi.rmsd_to_lowest_energy_ang, fi.rg_angstrom);
                let pj_scr = map_xy(fj.rmsd_to_lowest_energy_ang, fj.rg_angstrom);

                let (src, dst) =
                    if bundle.msm.folding_committor[i] <= bundle.msm.folding_committor[j] {
                        (pi_scr, pj_scr)
                    } else {
                        (pj_scr, pi_scr)
                    };

                let width = (1.0 + 22.0 * (flux as f32)).clamp(1.0, 4.8);
                let edge_col = Color32::from_rgba_unmultiplied(226, 232, 240, 130);
                plot_painter.line_segment([src, dst], Stroke::new(width, edge_col));

                // Midpoint directional chevron toward higher `p_fold`
                let dir = dst - src;
                let len = dir.length();
                if len > 24.0 {
                    let u = dir / len;
                    let perp = Vec2::new(-u.y, u.x);
                    let mid = src + dir * 0.55;
                    let wing1 = mid - u * 6.5 + perp * 4.0;
                    let wing2 = mid - u * 6.5 - perp * 4.0;
                    plot_painter.line_segment([wing1, mid], Stroke::new(1.5_f32, edge_col));
                    plot_painter.line_segment([wing2, mid], Stroke::new(1.5_f32, edge_col));
                }
            }
        }
    }

    // 3. Microstate circles (`S0 .. SK-1`) colored by committor `p_fold`, clickable to inspect 3D
    let hover_pos = response.hover_pos();
    let click_pos = if response.clicked() {
        response.interact_pointer_pos()
    } else {
        None
    };
    let mut hovered_micro: Option<usize> = None;
    let mut clicked_micro: Option<usize> = None;

    for i in 0..bundle.msm.num_microstates {
        let orig_i = bundle.msm.active_microstates[i];
        let med = &bundle.frames[bundle.clustering.medoid_frame_indices[orig_i]];
        let center = map_xy(med.rmsd_to_lowest_energy_ang, med.rg_angstrom);
        let radius = (6.5 + 20.0 * (bundle.msm.stationary_pi[i].sqrt() as f32)).clamp(6.5, 18.0);
        let fill = committor_node_color(bundle.msm.folding_committor[i] as f32);

        let in_selected_macro =
            selected_macro == Some(bundle.msm.microstate_to_macrostate[i]);
        let is_exact_micro = selected_micro == Some(i);

        if in_selected_macro {
            plot_painter.circle_stroke(
                center,
                radius + 3.2,
                Stroke::new(1.8_f32, Color32::from_rgb(56, 189, 248)),
            );
        }
        if is_exact_micro {
            plot_painter.circle_stroke(
                center,
                radius + 6.0,
                Stroke::new(2.2_f32, Color32::WHITE),
            );
        }

        plot_painter.circle_filled(center, radius, fill);
        plot_painter.circle_stroke(
            center,
            radius,
            Stroke::new(1.3_f32, Color32::from_rgb(10, 15, 24)),
        );
        plot_painter.text(
            center,
            Align2::CENTER_CENTER,
            format!("S{}", i),
            FontId::monospace(10.0),
            Color32::from_rgb(10, 15, 24),
        );

        if let Some(hp) = hover_pos {
            if hp.distance(center) <= radius + 4.0 {
                hovered_micro = Some(i);
            }
        }
        if let Some(cp) = click_pos {
            if cp.distance(center) <= radius + 5.0 {
                clicked_micro = Some(i);
            }
        }
    }

    // 4. Axis ticks & titles (aligned to `data_rect` projections)
    for t in 0..=4 {
        let f = (t as f32) / 4.0;
        let xv = grid.rmsd_min + (f as f64) * (grid.rmsd_max - grid.rmsd_min);
        let xp = (data_rect.left() + f * data_rect.width())
            .clamp(plot_rect.left() + 14.0, plot_rect.right() - 14.0);
        painter.line_segment(
            [
                Pos2::new(xp, plot_rect.bottom()),
                Pos2::new(xp, plot_rect.bottom() + 4.0),
            ],
            Stroke::new(1.0_f32, Color32::from_rgb(90, 108, 135)),
        );
        painter.text(
            Pos2::new(xp, plot_rect.bottom() + 6.0),
            Align2::CENTER_TOP,
            format!("{:.2}", xv),
            FontId::monospace(10.5),
            Color32::from_rgb(160, 175, 195),
        );

        let yv = grid.rg_min + (f as f64) * (grid.rg_max - grid.rg_min);
        let yp = (data_rect.bottom() - f * data_rect.height())
            .clamp(plot_rect.top() + 8.0, plot_rect.bottom() - 8.0);
        painter.line_segment(
            [
                Pos2::new(plot_rect.left() - 4.0, yp),
                Pos2::new(plot_rect.left(), yp),
            ],
            Stroke::new(1.0_f32, Color32::from_rgb(90, 108, 135)),
        );
        painter.text(
            Pos2::new(plot_rect.left() - 6.0, yp),
            Align2::RIGHT_CENTER,
            format!("{:.2}", yv),
            FontId::monospace(10.5),
            Color32::from_rgb(160, 175, 195),
        );
    }

    painter.text(
        Pos2::new(rect.left() + 14.0, rect.top() + 8.0),
        Align2::LEFT_TOP,
        format!(
            "2D Free-Energy Surface W(RMSD, Rg) & Markov Flux — {} ({})",
            bundle.manifest.name, bundle.source_label
        ),
        FontId::monospace(12.5),
        Color32::from_rgb(226, 232, 240),
    );
    painter.text(
        Pos2::new(plot_rect.center().x, rect.bottom() - 20.0),
        Align2::CENTER_TOP,
        "C-Alpha RMSD to Lowest-Energy State (Angstroms)   [Amber: p_fold=0 Unfolded -> Emerald: p_fold=1 Folded]",
        FontId::monospace(10.5),
        Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        Pos2::new(plot_rect.left(), plot_rect.top() - 16.0),
        Align2::LEFT_TOP,
        "Rg (A)",
        FontId::monospace(10.5),
        Color32::from_rgb(148, 163, 184),
    );

    // 5. Hover readout bar (positioned above `Rg (A)` with zero overlap)
    if let Some(m_idx) = hovered_micro {
        let orig_m = bundle.msm.active_microstates[m_idx];
        let med = &bundle.frames[bundle.clustering.medoid_frame_indices[orig_m]];
        let mac_id = bundle.msm.microstate_to_macrostate[m_idx];
        let micro_dg = if bundle.msm.microstate_free_energy_kcal[m_idx].abs() < 5e-4 {
            0.0
        } else {
            bundle.msm.microstate_free_energy_kcal[m_idx]
        };
        painter.text(
            Pos2::new(rect.left() + 14.0, rect.top() + 24.0),
            Align2::LEFT_TOP,
            format!(
                "Hover S{} (Macrostate M{}): pi={:.2}% | dG={:.2} kcal/mol | p_fold={:.2} | Medoid={} — Click to load 3D",
                m_idx,
                mac_id,
                bundle.msm.stationary_pi[m_idx] * 100.0,
                micro_dg,
                bundle.msm.folding_committor[m_idx],
                med.wu_id
            ),
            FontId::monospace(11.0),
            Color32::from_rgb(52, 211, 153),
        );
    } else if let Some(hp) = hover_pos {
        if plot_rect.contains(hp) {
            let fx = ((hp.x - data_rect.left()) / data_rect.width().max(1.0)).clamp(0.0, 0.9999);
            let fy =
                ((data_rect.bottom() - hp.y) / data_rect.height().max(1.0)).clamp(0.0, 0.9999);
            let ix = ((fx * (grid.nx as f32)) as usize).min(grid.nx.saturating_sub(1));
            let iy = ((fy * (grid.ny as f32)) as usize).min(grid.ny.saturating_sub(1));
            let rmsd_v = grid.rmsd_min + (fx as f64) * (grid.rmsd_max - grid.rmsd_min);
            let rg_v = grid.rg_min + (fy as f64) * (grid.rg_max - grid.rg_min);
            let pmf_v = grid.free_energy_kcal[ix][iy];
            painter.text(
                Pos2::new(rect.left() + 14.0, rect.top() + 24.0),
                Align2::LEFT_TOP,
                format!(
                    "Surface Cursor: RMSD = {:.2} A | Rg = {:.2} A | PMF = {:.2} kcal/mol",
                    rmsd_v, rg_v, pmf_v
                ),
                FontId::monospace(11.0),
                Color32::from_rgb(56, 189, 248),
            );
        }
    }

    clicked_micro
}

fn dark_pmf_colormap(norm: f32) -> Color32 {
    // 0.0 (deep free-energy basin): luminous cyan-cobalt (28, 128, 212)
    // 0.45 (intermediate valley): slate-indigo (32, 58, 102)
    // 1.0 (high barrier >= 6 kcal/mol): dark obsidian background (16, 22, 33)
    let t = norm.clamp(0.0, 1.0);
    if t < 0.45 {
        let u = t / 0.45;
        Color32::from_rgb(
            (28.0 + u * (32.0 - 28.0)) as u8,
            (128.0 + u * (58.0 - 128.0)) as u8,
            (212.0 + u * (102.0 - 212.0)) as u8,
        )
    } else {
        let u = (t - 0.45) / 0.55;
        Color32::from_rgb(
            (32.0 + u * (16.0 - 32.0)) as u8,
            (58.0 + u * (22.0 - 58.0)) as u8,
            (102.0 + u * (33.0 - 102.0)) as u8,
        )
    }
}

fn committor_node_color(p_fold: f32) -> Color32 {
    let p = p_fold.clamp(0.0, 1.0);
    if p < 0.5 {
        let u = p * 2.0;
        Color32::from_rgb(
            (245.0 + u * (192.0 - 245.0)) as u8,
            (178.0 + u * (188.0 - 178.0)) as u8,
            (56.0 + u * (215.0 - 56.0)) as u8,
        )
    } else {
        let u = (p - 0.5) * 2.0;
        Color32::from_rgb(
            (192.0 + u * (52.0 - 192.0)) as u8,
            (188.0 + u * (211.0 - 188.0)) as u8,
            (215.0 + u * (153.0 - 215.0)) as u8,
        )
    }
}

fn filter_cold_rung_frames(
    all_frames: Vec<ConformationalFrame>,
    ref_temp: f64,
    cold_only: bool,
) -> Vec<ConformationalFrame> {
    if !cold_only {
        return all_frames;
    }
    let cutoff = ref_temp + 25.0;
    let cold: Vec<ConformationalFrame> = all_frames
        .iter()
        .filter(|f| f.temperature_kelvin <= cutoff)
        .cloned()
        .collect();
    if cold.len() >= 8 {
        cold
    } else {
        all_frames
    }
}

#[allow(clippy::too_many_arguments)]
fn build_msm_from_live_frames(
    topology: Arc<ProteinTopology>,
    raw_frames: Vec<ConformationalFrame>,
    steps_per_gen: u64,
    k_req: usize,
    m_req: usize,
    lag: u32,
    ref_temp: f64,
    cold_only: bool,
) -> Result<ComputedMsmBundle, String> {
    let total_raw = raw_frames.len();
    let mut frames = filter_cold_rung_frames(raw_frames, ref_temp, cold_only);
    frames.sort_by_key(|f| (f.wu_id.run, f.wu_id.clone, f.chain_step));
    for (idx, f) in frames.iter_mut().enumerate() {
        f.frame_idx = idx;
    }

    let mut best_idx = 0usize;
    for i in 1..frames.len() {
        if frames[i].potential_energy_kcal < frames[best_idx].potential_energy_kcal {
            best_idx = i;
        }
    }
    let best_pos = frames[best_idx].positions_angstrom.clone();
    for f in &mut frames {
        f.rmsd_to_lowest_energy_ang = compute_ca_rmsd(&topology, &best_pos, &f.positions_angstrom);
    }

    let rmsd_mat = compute_pairwise_rmsd_matrix(&topology, &frames);
    let actual_k = k_req.min(frames.len() / 2).max(2);
    let clustering = cluster_conformations_kmedoids(&frames, &rmsd_mat, actual_k, None);
    let actual_m = m_req.min(clustering.num_microstates).max(2);
    let lag_ps = (steps_per_gen as f64) * 1.5e-3 * (lag.max(1) as f64);

    let msm = estimate_reversible_msm(
        &frames,
        &clustering,
        lag,
        lag_ps,
        ref_temp,
        actual_m,
    )
    .map_err(|e| e.to_string())?;

    let pmf_grid = compute_2d_pmf_grid(&frames, &clustering, &msm, 28, 28);
    let manifest = ProjectManifest {
        project_id: 1,
        name: topology.name.clone(),
        sequence: topology.sequence_one_letter.clone(),
        topology_blake3: [0u8; 32],
        initial_coords_blake3: [0u8; 32],
        atom_count: topology.atoms.len() as u32,
        residue_count: topology.residue_count,
        temp_ladder_kelvin: vec![285.0, 318.0, 356.0, 398.0],
        dt_fs: 1.5,
        friction_per_ps: 1.0,
        steps_per_gen,
        num_clones_per_run: 1,
        max_generations: 1,
    };

    Ok(ComputedMsmBundle {
        source_label: if cold_only && frames.len() < total_raw {
            format!("Live Cold-Rung ({}/{total_raw} frames)", frames.len())
        } else {
            format!("Live MREMD Studio ({} frames)", frames.len())
        },
        manifest,
        topology,
        frames,
        clustering,
        msm,
        pmf_grid,
    })
}

fn build_msm_from_redb_vault(
    vault_dir: &PathBuf,
    project_id: u32,
    k_req: usize,
    m_req: usize,
    lag: u32,
    ref_temp: f64,
    cold_only: bool,
) -> Result<ComputedMsmBundle, String> {
    let store = FoldStore::open(vault_dir).map_err(|e| e.to_string())?;
    let max_t = if cold_only {
        Some(ref_temp + 25.0)
    } else {
        None
    };
    let (manifest, topo, frames) = match extract_project_frames(&store, project_id, max_t) {
        Ok(res) if res.2.len() >= 8 => res,
        _ => extract_project_frames(&store, project_id, None).map_err(|e| e.to_string())?,
    };
    let topo_arc = Arc::new(topo);

    let rmsd_mat = compute_pairwise_rmsd_matrix(&topo_arc, &frames);
    let actual_k = k_req.min(frames.len() / 2).max(2);
    let clustering = cluster_conformations_kmedoids(&frames, &rmsd_mat, actual_k, None);
    let actual_m = m_req.min(clustering.num_microstates).max(2);
    let lag_ps = (manifest.steps_per_gen as f64) * manifest.dt_fs * 1.0e-3 * (lag.max(1) as f64);

    let msm = estimate_reversible_msm(
        &frames,
        &clustering,
        lag,
        lag_ps,
        ref_temp,
        actual_m,
    )
    .map_err(|e| e.to_string())?;

    let pmf_grid = compute_2d_pmf_grid(&frames, &clustering, &msm, 28, 28);

    Ok(ComputedMsmBundle {
        source_label: format!("Vault {} ({} frames)", vault_dir.display(), frames.len()),
        manifest,
        topology: topo_arc,
        frames,
        clustering,
        msm,
        pmf_grid,
    })
}
