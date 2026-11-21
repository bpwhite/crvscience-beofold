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

use crate::lan_client::{
    spawn_network_telemetry_thread, spawn_one_click_mdns_scan, NetworkTelemetryState,
    PollerTargets,
};
use crate::live_sandbox::{
    spawn_live_folding_worker, LiveFoldingState, NativeReferenceSource, SandboxCommand,
    TargetPreset,
};
use crate::msm_tab::MsmLandscapeTab;
use crate::plots::draw_scientific_trace;
use crate::viewport3d::{ConformationViewport, RenderRepresentation};
use egui::{Color32, RichText};
use fold_physics::{sanitize_amino_acid_sequence, ProteinTopology};
use fold_proto::{
    ClusterStateSnapshot, ComputeTier, GlobalFederationTelemetry, LivePeerStatus,
    NodeCapabilityProfile, PeerEndpoint, RegisteredClusterRecord, WorkUnitId,
};
use fold_storage::to_hex;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkspaceTab {
    FoldingStudio3D,
    LanBeowulfMatrix,
    GlobalWanFederation,
    MsmFreeEnergyFlux,
}

pub(crate) fn classify_status_color(msg: &str, default_color: Color32) -> Color32 {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("error") || lower.contains("failed") {
        Color32::from_rgb(248, 113, 113)
    } else if lower.contains("need >=") {
        Color32::from_rgb(251, 191, 36)
    } else {
        default_color
    }
}

pub(crate) fn non_empty_trimmed_path(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Err("Export error: output path cannot be empty.".to_string())
    } else {
        Ok(PathBuf::from(trimmed))
    }
}

pub struct GuiLaunchConfig {
    pub lan_peer: String,
    pub cluster_secret: String,
    pub sync_url: String,
    pub vault_dir: String,
    pub use_gpu: bool,
}

pub struct BeoFoldGuiApp {
    active_tab: WorkspaceTab,

    // Connection bar editable fields
    lan_peer_input: String,
    cluster_secret_input: String,
    wan_sync_url_input: String,
    vault_dir_input: String,
    export_pdb_path: String,

    // Local studio controls
    selected_preset: TargetPreset,
    steps_per_slice: u64,
    use_gpu: bool,
    custom_seq_name: String,
    custom_seq_input: String,
    last_import_msg: Option<String>,

    // Shared worker channels & telemetry snapshots
    net_state: Arc<Mutex<NetworkTelemetryState>>,
    net_targets_tx: watch::Sender<PollerTargets>,
    live_state: Arc<Mutex<LiveFoldingState>>,
    sandbox_tx: mpsc::Sender<SandboxCommand>,
    viewport: ConformationViewport,
    compare_start_vs_folded: bool,
    msm_tab: MsmLandscapeTab,
    last_export_msg: Option<String>,
    show_about: bool,
}

impl BeoFoldGuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>, cfg: GuiLaunchConfig) -> Self {
        Self::new_with_ctx(&cc.egui_ctx, cfg)
    }

    pub fn new_with_ctx(ctx: &egui::Context, cfg: GuiLaunchConfig) -> Self {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Color32::from_rgb(15, 20, 28);
        visuals.window_fill = Color32::from_rgb(18, 24, 34);
        ctx.set_visuals(visuals);

        let initial_targets = PollerTargets {
            lan_peer: cfg.lan_peer.trim().parse::<SocketAddr>().ok(),
            cluster_secret: cfg.cluster_secret.clone(),
            wan_sync_url: if cfg.sync_url.trim().is_empty() {
                None
            } else {
                Some(cfg.sync_url.trim().to_string())
            },
        };

        let net_state = Arc::new(Mutex::new(NetworkTelemetryState {
            lan_target: cfg.lan_peer.clone(),
            lan_secret: cfg.cluster_secret.clone(),
            wan_sync_url: cfg.sync_url.clone(),
            ..Default::default()
        }));
        let net_targets_tx = spawn_network_telemetry_thread(initial_targets, net_state.clone());

        let preset = TargetPreset::TrpCage1L2Y;
        let live_state = Arc::new(Mutex::new(LiveFoldingState::new_preset(
            preset,
            cfg.use_gpu,
        )));
        let sandbox_tx = spawn_live_folding_worker(live_state.clone(), preset, cfg.use_gpu);

        Self {
            active_tab: WorkspaceTab::FoldingStudio3D,
            lan_peer_input: cfg.lan_peer,
            cluster_secret_input: cfg.cluster_secret,
            wan_sync_url_input: cfg.sync_url,
            vault_dir_input: cfg.vault_dir,
            export_pdb_path: "./beofold_supervisor_best.pdb".to_string(),
            selected_preset: preset,
            steps_per_slice: 150,
            use_gpu: cfg.use_gpu,
            custom_seq_name: "CustomPeptide".to_string(),
            custom_seq_input: "NLYIQWLKDGGPSSGRPPPS".to_string(),
            last_import_msg: None,
            net_state,
            net_targets_tx,
            live_state,
            sandbox_tx,
            viewport: ConformationViewport::default(),
            compare_start_vs_folded: false,
            msm_tab: MsmLandscapeTab::default(),
            last_export_msg: None,
            show_about: false,
        }
    }

    fn apply_network_targets(&self) {
        let parsed_peer = self.lan_peer_input.trim().parse::<SocketAddr>().ok();
        let wan_opt = if self.wan_sync_url_input.trim().is_empty() {
            None
        } else {
            Some(self.wan_sync_url_input.trim().to_string())
        };
        let _ = self.net_targets_tx.send(PollerTargets {
            lan_peer: parsed_peer,
            cluster_secret: self.cluster_secret_input.clone(),
            wan_sync_url: wan_opt,
        });
    }

    fn render_top_bar(&mut self, ui: &mut egui::Ui, net: &NetworkTelemetryState) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(concat!("BEOFOLD v", env!("CARGO_PKG_VERSION")))
                    .strong()
                    .size(16.0)
                    .color(Color32::from_rgb(56, 189, 248)),
            );
            ui.separator();

            ui.selectable_value(
                &mut self.active_tab,
                WorkspaceTab::FoldingStudio3D,
                "3D Conformation & MREMD Studio",
            );
            ui.selectable_value(
                &mut self.active_tab,
                WorkspaceTab::LanBeowulfMatrix,
                format!(
                    "LAN Beowulf Matrix ({})",
                    net.lan_snapshot
                        .as_ref()
                        .map(|s| s.peers.len())
                        .unwrap_or(0)
                ),
            );
            ui.selectable_value(
                &mut self.active_tab,
                WorkspaceTab::GlobalWanFederation,
                format!(
                    "Apache WAN Federation ({})",
                    net.wan_snapshot
                        .as_ref()
                        .map(|w| w.clusters.len())
                        .unwrap_or(0)
                ),
            );
            ui.selectable_value(
                &mut self.active_tab,
                WorkspaceTab::MsmFreeEnergyFlux,
                "MSM Free-Energy & Flux",
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let wan_badge = if net.wan_connected {
                    ("WAN: SYNCED", Color32::from_rgb(52, 211, 153))
                } else if self.wan_sync_url_input.trim().is_empty() {
                    ("WAN: OFFLINE", Color32::from_rgb(120, 132, 150))
                } else {
                    ("WAN: RETRYING", Color32::from_rgb(251, 191, 36))
                };
                ui.label(RichText::new(wan_badge.0).monospace().color(wan_badge.1));
                ui.separator();

                let lan_badge = if net.lan_connected {
                    ("LAN QUIC: ONLINE", Color32::from_rgb(52, 211, 153))
                } else {
                    ("LAN QUIC: STANDBY", Color32::from_rgb(148, 163, 184))
                };
                ui.label(RichText::new(lan_badge.0).monospace().color(lan_badge.1));
            });
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("LAN Peer:").monospace().size(11.5));
            ui.add(egui::TextEdit::singleline(&mut self.lan_peer_input).desired_width(135.0));

            ui.label(RichText::new("Secret:").monospace().size(11.5));
            ui.add(
                egui::TextEdit::singleline(&mut self.cluster_secret_input)
                    .password(true)
                    .desired_width(110.0),
            );

            ui.label(RichText::new("Apache Sync URL:").monospace().size(11.5));
            ui.add(egui::TextEdit::singleline(&mut self.wan_sync_url_input).desired_width(240.0));

            if ui.button("Connect / Refresh").clicked() {
                self.apply_network_targets();
            }
            let scan_label = if net.lan_discovery_scanning {
                "Scanning LAN..."
            } else {
                "Discover LAN"
            };
            if ui
                .add_enabled(!net.lan_discovery_scanning, egui::Button::new(scan_label))
                .on_hover_text(
                    "Browse _beofold._udp.local. (mDNS) and localhost 7443..7446 for Beowulf cluster nodes.",
                )
                .clicked()
            {
                spawn_one_click_mdns_scan(
                    self.cluster_secret_input.clone(),
                    self.wan_sync_url_input.clone(),
                    self.net_state.clone(),
                    self.net_targets_tx.clone(),
                    true,
                );
            }
        });

        if !net.discovered_lan_peers.is_empty() || net.lan_discovery_status.is_some() {
            ui.horizontal_wrapped(|ui| {
                if let Some(ref st) = net.lan_discovery_status {
                    ui.label(
                        RichText::new(st)
                            .monospace()
                            .size(10.5)
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                }
                let mut picked_peer: Option<String> = None;
                for peer in &net.discovered_lan_peers {
                    let chip_col = if peer.matches_secret {
                        Color32::from_rgb(52, 211, 153)
                    } else {
                        Color32::from_rgb(251, 191, 36)
                    };
                    let chip_text = format!(
                        "{} [Tier {} | {}c]",
                        peer.addr, peer.tier, peer.cores
                    );
                    if ui
                        .small_button(RichText::new(chip_text).monospace().color(chip_col))
                        .clicked()
                    {
                        picked_peer = Some(peer.addr.to_string());
                    }
                }
                if let Some(addr_str) = picked_peer {
                    self.lan_peer_input = addr_str;
                    self.apply_network_targets();
                }
            });
        }
    }

    fn render_studio_tab(&mut self, ctx: &egui::Context, live: &LiveFoldingState) {
        egui::SidePanel::left("studio_left_panel")
            .resizable(true)
            .default_width(335.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Local MREMD Folding Runner");
                    ui.label(
                        RichText::new(&live.active_backend_label)
                            .monospace()
                            .size(11.5)
                            .color(Color32::from_rgb(56, 189, 248)),
                    );
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        let btn_label = if live.running {
                            "Pause Folding"
                        } else {
                            "Start Live MREMD"
                        };
                        if ui.button(RichText::new(btn_label).strong()).clicked() {
                            let _ = self
                                .sandbox_tx
                                .send(SandboxCommand::SetRunning(!live.running));
                        }
                        if ui.button("Reset Seed").clicked() {
                            let _ = self.sandbox_tx.send(SandboxCommand::ResetTarget {
                                preset: self.selected_preset,
                                use_gpu: self.use_gpu,
                                steps_per_slice: self.steps_per_slice,
                            });
                        }
                    });

                    ui.add_space(6.0);
                    egui::ComboBox::from_label("Target")
                        .selected_text(self.selected_preset.label())
                        .show_ui(ui, |ui| {
                            for p in [TargetPreset::TrpCage1L2Y, TargetPreset::VillinHP35] {
                                if ui
                                    .selectable_value(&mut self.selected_preset, p, p.label())
                                    .changed()
                                {
                                    self.last_import_msg = None;
                                    let _ = self.sandbox_tx.send(SandboxCommand::ResetTarget {
                                        preset: self.selected_preset,
                                        use_gpu: self.use_gpu,
                                        steps_per_slice: self.steps_per_slice,
                                    });
                                }
                            }
                        });

                    ui.horizontal(|ui| {
                        if ui.button("Import .pdb / .cif...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Protein Structure", &["pdb", "ent", "cif", "mmcif"])
                                .pick_file()
                            {
                                match ProteinTopology::from_pdb_file(&path) {
                                    Ok((topo, coords)) => {
                                        let disp = format!(
                                            "{} ({} aa, PDB)",
                                            topo.name, topo.residue_count
                                        );
                                        self.last_import_msg =
                                            Some(format!("Loaded PDB: {}", path.display()));
                                        let _ = self.sandbox_tx.send(
                                            SandboxCommand::LoadCustomTarget {
                                                name: disp,
                                                topology: topo,
                                                positions: coords,
                                                from_pdb: true,
                                                use_gpu: self.use_gpu,
                                                steps_per_slice: self.steps_per_slice,
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        self.last_import_msg =
                                            Some(format!("PDB import error: {err}"));
                                    }
                                }
                            }
                        }
                    });

                    ui.collapsing("Custom Sequence (FASTA / 1-Letter)", |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Name:").size(11.0));
                            ui.add(
                                egui::TextEdit::singleline(&mut self.custom_seq_name)
                                    .desired_width(175.0),
                            );
                        });
                        ui.add(
                            egui::TextEdit::multiline(&mut self.custom_seq_input)
                                .desired_rows(2)
                                .desired_width(290.0),
                        );
                        if ui.button("Build & Load Sequence").clicked() {
                            match sanitize_amino_acid_sequence(&self.custom_seq_input) {
                                Ok(clean_seq) => {
                                    let name = if self.custom_seq_name.trim().is_empty() {
                                        "CustomPeptide"
                                    } else {
                                        self.custom_seq_name.trim()
                                    };
                                    let (topo, coords) = ProteinTopology::build_polypeptide_from_sequence(
                                        name,
                                        &clean_seq,
                                    );
                                    let disp = format!(
                                        "{} ({} aa, Seq)",
                                        topo.name, topo.residue_count
                                    );
                                    self.last_import_msg = Some(format!(
                                        "Built {} ({} residues)",
                                        topo.name, topo.residue_count
                                    ));
                                    let _ =
                                        self.sandbox_tx.send(SandboxCommand::LoadCustomTarget {
                                            name: disp,
                                            topology: topo,
                                            positions: coords,
                                            from_pdb: false,
                                            use_gpu: self.use_gpu,
                                            steps_per_slice: self.steps_per_slice,
                                        });
                                }
                                Err(err) => {
                                    self.last_import_msg =
                                        Some(format!("Sequence error: {err}"));
                                }
                            }
                        }
                    });
                    if let Some(ref msg) = self.last_import_msg {
                        ui.label(
                            RichText::new(msg)
                                .monospace()
                                .size(10.5)
                                .color(classify_status_color(msg, Color32::from_rgb(56, 189, 248))),
                        );
                    }

                    if ui
                        .checkbox(&mut self.use_gpu, "Enable wgpu WGSL Compute Shaders")
                        .changed()
                    {
                        let _ = self
                            .sandbox_tx
                            .send(SandboxCommand::SetUseGpu(self.use_gpu));
                    }

                    if ui
                        .add(
                            egui::Slider::new(&mut self.steps_per_slice, 50..=1500)
                                .text("Steps / Slice"),
                        )
                        .changed()
                    {
                        let _ = self
                            .sandbox_tx
                            .send(SandboxCommand::SetStepsPerSlice(self.steps_per_slice));
                    }

                    ui.separator();
                    ui.heading("3D Viewport & PDB Export");
                    egui::ComboBox::from_label("Style")
                        .selected_text(self.viewport.representation.label())
                        .show_ui(ui, |ui| {
                            for rep in [
                                RenderRepresentation::BackboneAndSidechains,
                                RenderRepresentation::CAlphaRibbonOnly,
                                RenderRepresentation::AllHeavyAtoms,
                            ] {
                                ui.selectable_value(
                                    &mut self.viewport.representation,
                                    rep,
                                    rep.label(),
                                );
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.viewport.auto_spin, "Auto-Spin");
                        ui.checkbox(
                            &mut self.viewport.show_best_conformation,
                            "Lock Lowest-E Fold",
                        );
                        if ui.button("Reset Cam").clicked() {
                            self.viewport.reset_camera();
                        }
                    });
                    ui.checkbox(
                        &mut self.compare_start_vs_folded,
                        "Side-by-Side: Unfolded Start vs. Folded End",
                    );

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.export_pdb_path)
                                .desired_width(140.0),
                        );
                        if ui.button("Save .pdb...").clicked() {
                            let default_name = PathBuf::from(self.export_pdb_path.trim())
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("beofold_supervisor_best.pdb")
                                .to_string();
                            if let Some(out) = rfd::FileDialog::new()
                                .add_filter("PDB Structure", &["pdb"])
                                .set_file_name(&default_name)
                                .save_file()
                            {
                                self.export_pdb_path = out.display().to_string();
                                match live.export_best_pdb(&out) {
                                    Ok(()) => {
                                        self.last_export_msg =
                                            Some(format!("Saved PDB to {}", out.display()));
                                    }
                                    Err(e) => {
                                        self.last_export_msg =
                                            Some(format!("Export failed: {}", e));
                                    }
                                }
                            }
                        }
                        if ui.button("Export").clicked() {
                            match non_empty_trimmed_path(&self.export_pdb_path) {
                                Ok(out) => match live.export_best_pdb(&out) {
                                    Ok(()) => {
                                        self.last_export_msg =
                                            Some(format!("Saved PDB to {}", out.display()));
                                    }
                                    Err(e) => {
                                        self.last_export_msg = Some(format!("Export failed: {}", e));
                                    }
                                },
                                Err(msg) => self.last_export_msg = Some(msg),
                            }
                        }
                    });
                    if let Some(ref msg) = self.last_export_msg {
                        ui.label(
                            RichText::new(msg)
                                .monospace()
                                .size(11.0)
                                .color(classify_status_color(msg, Color32::from_rgb(52, 211, 153))),
                        );
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.vault_dir_input)
                                .desired_width(120.0),
                        );
                        if ui.button("Browse...").clicked() {
                            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                self.vault_dir_input = folder.display().to_string();
                            }
                        }
                        if ui.button("Load Vault").clicked() {
                            let _ = self.sandbox_tx.send(SandboxCommand::LoadFromVault {
                                data_dir: PathBuf::from(self.vault_dir_input.trim()),
                                project_id: 1,
                            });
                        }
                    });

                    ui.separator();
                    ui.heading("4-Rung MREMD Replica Ladder");
                    ui.label(
                        RichText::new(format!(
                            "Throughput: {:.2} ns/day | Lowest V_pot: {:.2} kcal/mol ({})",
                            live.instantaneous_ns_per_day, live.best_potential_kcal, live.best_wu
                        ))
                        .monospace()
                        .size(11.0),
                    );

                    egui::Grid::new("mremd_rung_grid")
                        .striped(true)
                        .num_columns(5)
                        .show(ui, |ui| {
                            ui.label(RichText::new("Rung").strong().monospace());
                            ui.label(RichText::new("Bath").strong().monospace());
                            ui.label(RichText::new("Gen").strong().monospace());
                            ui.label(RichText::new("V_pot").strong().monospace());
                            ui.label(RichText::new("Rg / RMSD").strong().monospace());
                            ui.end_row();

                            for r in &live.rungs {
                                ui.label(RichText::new(format!("R{}", r.rung_idx)).monospace());
                                ui.label(
                                    RichText::new(format!("{:.0}K", r.temperature_kelvin))
                                        .monospace(),
                                );
                                ui.label(RichText::new(format!("G{}", r.generation)).monospace());
                                ui.label(
                                    RichText::new(format!("{:.1}", r.potential_kcal)).monospace(),
                                );
                                ui.label(
                                    RichText::new(format!(
                                        "{:.1} / {:.1}A",
                                        r.rg_angstrom, r.rmsd_seed_ang
                                    ))
                                    .monospace(),
                                );
                                ui.end_row();
                            }
                        });

                    ui.add_space(6.0);
                    ui.label(RichText::new("Adjacent Rung Swap Acceptance:").strong());
                    for (idx, &(acc, trials)) in live.swap_pair_stats.iter().enumerate() {
                        let rate = if trials > 0 {
                            (acc as f32) / (trials as f32)
                        } else {
                            0.0
                        };
                        ui.add(
                            egui::ProgressBar::new(rate).text(format!(
                                "R{} <-> R{}: {}/{} ({:.1}%)",
                                idx,
                                idx + 1,
                                acc,
                                trials,
                                rate * 100.0
                            )),
                        );
                    }
                });
            });

        egui::SidePanel::right("studio_right_plots")
            .resizable(true)
            .default_width(350.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Thermodynamic & MSM Traces");
                    ui.add_space(4.0);
                    draw_scientific_trace(
                        ui,
                        "Potential Energy V_total (green = min)",
                        "kcal/mol",
                        &live.potential_history,
                        Some(&live.min_potential_history),
                        Color32::from_rgb(56, 189, 248),
                        145.0,
                    );
                    ui.add_space(6.0);
                    draw_scientific_trace(
                        ui,
                        "Radius of Gyration R_g",
                        "A",
                        &live.rg_history,
                        None,
                        Color32::from_rgb(245, 190, 65),
                        130.0,
                    );
                    ui.add_space(6.0);
                    draw_scientific_trace(
                        ui,
                        "C-Alpha RMSD from Unfolded Seed",
                        "A",
                        &live.rmsd_history,
                        None,
                        Color32::from_rgb(192, 132, 252),
                        130.0,
                    );
                    ui.add_space(6.0);
                    let q_title = match live.native_source {
                        NativeReferenceSource::PdbCrystal => "Native Contacts Q(t) [PDB Ref]",
                        NativeReferenceSource::LowestEnergyFold => {
                            "Native Contacts Q(t) [Low-E Ref]"
                        }
                    };
                    draw_scientific_trace(
                        ui,
                        q_title,
                        "Q",
                        &live.q_history,
                        None,
                        Color32::from_rgb(52, 211, 153),
                        130.0,
                    );
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.compare_start_vs_folded {
                let max_radius = |pts: &[[f32; 3]]| -> f32 {
                    if pts.is_empty() {
                        return 6.0;
                    }
                    let inv = 1.0 / (pts.len() as f32);
                    let mut mx = 0.0f32;
                    let mut my = 0.0f32;
                    let mut mz = 0.0f32;
                    for p in pts {
                        mx += p[0] * inv;
                        my += p[1] * inv;
                        mz += p[2] * inv;
                    }
                    let mut r_max = 6.0f32;
                    for p in pts {
                        let d = ((p[0] - mx).powi(2) + (p[1] - my).powi(2) + (p[2] - mz).powi(2)).sqrt();
                        if d > r_max {
                            r_max = d;
                        }
                    }
                    r_max
                };
                let unfolded_sub = format!(
                    "START (t=0 Unfolded) | Rg={:.2} A | V={:.1}",
                    live.seed_rg_angstrom, live.seed_potential_kcal
                );
                let cold_rung = live.rungs.iter().min_by(|a, b| {
                    a.temperature_kelvin
                        .partial_cmp(&b.temperature_kelvin)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let (right_coords, folded_sub) = if self.viewport.show_best_conformation {
                    (
                        &live.best_positions,
                        format!(
                            "FOLDED (Low-E {}) | Rg={:.2} A | V={:.1}",
                            live.best_wu, live.best_rg_angstrom, live.best_potential_kcal
                        ),
                    )
                } else if let Some(ref lbl) = live.inspected_label {
                    (
                        &live.current_positions,
                        format!(
                            "MSM {} | V={:.1} kcal/mol",
                            lbl, live.current_potential_kcal
                        ),
                    )
                } else if let Some(r) = cold_rung {
                    (
                        &live.current_positions,
                        format!(
                            "FOLDING END (G{}) | Rg={:.2} A | RMSD={:.1} A",
                            r.generation, r.rg_angstrom, r.rmsd_seed_ang
                        ),
                    )
                } else {
                    (
                        &live.current_positions,
                        format!(
                            "FOLDING END | Rg={:.2} A | V={:.1}",
                            live.best_rg_angstrom, live.best_potential_kcal
                        ),
                    )
                };
                let shared_extent = max_radius(&live.seed_positions)
                    .max(max_radius(right_coords))
                    * 1.12;
                let mut right_vp = self.viewport.clone();
                right_vp.auto_spin = false;
                ui.columns(2, |cols| {
                    self.viewport.ui_with_scale(
                        &mut cols[0],
                        &live.topology,
                        &live.seed_positions,
                        &unfolded_sub,
                        Some(shared_extent),
                    );
                    right_vp.ui_with_scale(
                        &mut cols[1],
                        &live.topology,
                        right_coords,
                        &folded_sub,
                        Some(shared_extent),
                    );
                });
            } else {
                let coords = if self.viewport.show_best_conformation {
                    &live.best_positions
                } else {
                    &live.current_positions
                };
                let mode_str = if self.viewport.show_best_conformation {
                    format!(
                        "Lowest-Energy Conformation {} ({:.2} kcal/mol)",
                        live.best_wu, live.best_potential_kcal
                    )
                } else if let Some(ref lbl) = live.inspected_label {
                    format!(
                        "MSM {} — {} ({:.2} kcal/mol)",
                        lbl, live.current_wu, live.current_potential_kcal
                    )
                } else {
                    format!(
                        "Active Conformation {} ({:.2} kcal/mol)",
                        live.current_wu, live.current_potential_kcal
                    )
                };
                self.viewport.ui(ui, &live.topology, coords, &mode_str);
            }
        });
    }

    fn render_lan_matrix_tab(&mut self, ctx: &egui::Context, net: &NetworkTelemetryState) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Home Beowulf LAN Cluster Matrix (QUIC TLS 1.3 + Raft)");
            if let Some(ref err) = net.lan_last_error {
                ui.label(
                    RichText::new(format!("LAN poll status: {}", err))
                        .monospace()
                        .color(Color32::from_rgb(251, 191, 36)),
                );
            }

            let Some(ref snap) = net.lan_snapshot else {
                ui.add_space(12.0);
                ui.label(
                    "No active LAN Beowulf node connected yet. Launch `fold-node mesh --init-trpcage` on your machines and click 'Discover LAN' or 'Connect / Refresh' above.",
                );
                ui.add_space(6.0);
                if ui
                    .add_enabled(!net.lan_discovery_scanning, egui::Button::new("Discover LAN Peers Now"))
                    .clicked()
                {
                    spawn_one_click_mdns_scan(
                        self.cluster_secret_input.clone(),
                        self.wan_sync_url_input.clone(),
                        self.net_state.clone(),
                        self.net_targets_tx.clone(),
                        true,
                    );
                }
                return;
            };

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Raft Term: {}", snap.raft_term))
                        .strong()
                        .monospace(),
                );
                ui.separator();
                let ldr_str = snap
                    .leader
                    .as_ref()
                    .map(|l| format!("{} ({}..)", l.addr, &to_hex(&l.node_pubkey)[0..10]))
                    .unwrap_or_else(|| "Election in progress".into());
                ui.label(RichText::new(format!("Leader: {}", ldr_str)).monospace());
                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "PRCG Units: {} done / {} leased / {} queued ({} total)",
                        snap.completed_workunits,
                        snap.leased_workunits,
                        snap.queued_workunits,
                        snap.total_workunits
                    ))
                    .monospace(),
                );
                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "MREMD Swaps: {}/{} | Quarantined: {}",
                        snap.mremd_swaps, snap.mremd_trials, snap.quarantined_events
                    ))
                    .monospace(),
                );
            });

            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("lan_peer_cards_grid")
                    .striped(true)
                    .num_columns(7)
                    .spacing([16.0, 10.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Hostname").strong());
                        ui.label(RichText::new("OS / Arch").strong());
                        ui.label(RichText::new("Compute Tier").strong());
                        ui.label(RichText::new("Villin Rate").strong());
                        ui.label(RichText::new("Active PRCG Chunk").strong());
                        ui.label(RichText::new("CPU Load").strong());
                        ui.label(RichText::new("Endpoint").strong());
                        ui.end_row();

                        for peer in &snap.peers {
                            ui.label(RichText::new(&peer.capability.hostname).monospace());
                            ui.label(
                                RichText::new(format!(
                                    "{}/{} ({}c)",
                                    peer.capability.os,
                                    peer.capability.arch,
                                    peer.capability.logical_cores
                                ))
                                .monospace(),
                            );
                            let tier_col = match peer.capability.tier {
                                ComputeTier::TierS => Color32::from_rgb(52, 211, 153),
                                ComputeTier::TierA => Color32::from_rgb(56, 189, 248),
                                ComputeTier::TierB => Color32::from_rgb(245, 190, 65),
                            };
                            ui.label(
                                RichText::new(format!("{}", peer.capability.tier))
                                    .monospace()
                                    .color(tier_col),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{:.2} ns/day",
                                    peer.capability.villin_ns_per_day
                                ))
                                .monospace(),
                            );
                            let wu_label = peer
                                .active_wu
                                .map(|w| format!("{}", w))
                                .unwrap_or_else(|| "IDLE / STAGING".into());
                            ui.label(RichText::new(wu_label).monospace());
                            let throt_str = if peer.throttled {
                                format!("{:.0}% [THROTTLED]", peer.cpu_usage_pct)
                            } else {
                                format!("{:.0}%", peer.cpu_usage_pct)
                            };
                            ui.label(RichText::new(throt_str).monospace());
                            ui.label(RichText::new(&peer.addr).monospace());
                            ui.end_row();
                        }
                    });
            });
        });
    }

    fn render_wan_federation_tab(&mut self, ctx: &egui::Context, net: &NetworkTelemetryState) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Global Multi-Cluster Federation (Apache HTTP Sync Server)");
            if let Some(ref err) = net.wan_last_error {
                ui.label(
                    RichText::new(format!("WAN sync poll error: {}", err))
                        .monospace()
                        .color(Color32::from_rgb(248, 113, 113)),
                );
            }

            let Some(ref wan) = net.wan_snapshot else {
                ui.add_space(12.0);
                ui.label(
                    "Enter your Apache Web Sync URL (e.g. `https://yourdomain.com/beofold/api/v1` or `http://127.0.0.1:8780/beofold/api/v1`) in the top bar and click 'Connect / Refresh'.",
                );
                return;
            };

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "Global Aggregate Throughput: {:.2} ns/day",
                        wan.aggregate_ns_per_day
                    ))
                    .strong()
                    .color(Color32::from_rgb(52, 211, 153)),
                );
                ui.separator();
                ui.label(RichText::new(format!("Quorum Mode: {}", wan.quorum_mode)).monospace());
                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "Canary Audits Passed: {} | Quarantined: {}",
                        wan.canary_audits_passed, wan.quarantined_events
                    ))
                    .monospace(),
                );
            });

            ui.add_space(4.0);
            let done_frac = if wan.total_workunits > 0 {
                (wan.completed_workunits as f32) / (wan.total_workunits as f32)
            } else {
                0.0
            };
            ui.add(egui::ProgressBar::new(done_frac).text(format!(
                "Global PRCG Progress: {} completed | {} leased | {} quorum-pending | {} queued ({} total)",
                wan.completed_workunits,
                wan.leased_workunits,
                wan.quorum_pending_workunits,
                wan.queued_workunits,
                wan.total_workunits
            )));

            if let Some((best_wu, best_e)) = wan.lowest_potential_kcal {
                ui.label(
                    RichText::new(format!(
                        "Lowest Global Conformation Energy: {:.3} kcal/mol ({})",
                        best_e, best_wu
                    ))
                    .monospace()
                    .color(Color32::from_rgb(56, 189, 248)),
                );
            }

            ui.separator();
            ui.heading(format!(
                "Registered Remote Beowulf Clusters ({})",
                wan.clusters.len()
            ));
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("wan_clusters_grid")
                    .striped(true)
                    .num_columns(7)
                    .spacing([16.0, 10.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Cluster Name").strong());
                        ui.label(RichText::new("Ed25519 Key").strong());
                        ui.label(RichText::new("Nodes (S/A/B)").strong());
                        ui.label(RichText::new("Aggregate Rate").strong());
                        ui.label(RichText::new("OS Mix").strong());
                        ui.label(RichText::new("Trust").strong());
                        ui.label(RichText::new("Completed / Quarantined").strong());
                        ui.end_row();

                        for c in &wan.clusters {
                            ui.label(RichText::new(&c.cluster_name).strong().monospace());
                            ui.label(
                                RichText::new(&to_hex(&c.cluster_pubkey)[0..12]).monospace(),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{} ({}/{}/{})",
                                    c.total_nodes, c.tier_s_nodes, c.tier_a_nodes, c.tier_b_nodes
                                ))
                                .monospace(),
                            );
                            ui.label(
                                RichText::new(format!("{:.2} ns/day", c.aggregate_ns_per_day))
                                    .monospace(),
                            );
                            ui.label(RichText::new(&c.os_summary).monospace());
                            let trust_txt = if c.trusted {
                                ("TRUSTED (q=1)", Color32::from_rgb(52, 211, 153))
                            } else {
                                ("EXTERNAL (q=2)", Color32::from_rgb(251, 191, 36))
                            };
                            ui.label(RichText::new(trust_txt.0).monospace().color(trust_txt.1));
                            ui.label(
                                RichText::new(format!(
                                    "{} / {}",
                                    c.completed_units, c.quarantined_units
                                ))
                                .monospace(),
                            );
                            ui.end_row();
                        }
                    });
            });
        });
    }
}

impl BeoFoldGuiApp {
    pub fn populate_for_screenshot(&mut self, view: u8) {
        let rounds = match view {
            2 => 24,
            3 => 10,
            _ => 220,
        };
        let _ = self
            .sandbox_tx
            .send(SandboxCommand::RunBurstRounds { rounds });
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(25) {
            let ready = self
                .live_state
                .lock()
                .map(|g| {
                    g.rungs
                        .first()
                        .map(|r| r.generation >= rounds)
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            if ready {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }

        match view {
            2 => {
                self.active_tab = WorkspaceTab::MsmFreeEnergyFlux;
                self.msm_tab.min_flux_threshold = 0.001;
                let live_snap = self.live_state.lock().map(|g| g.clone()).ok();
                if let Some(live) = live_snap {
                    let _ = self
                        .msm_tab
                        .compute_live_msm_blocking(&live, &self.sandbox_tx);
                }
            }
            3 => {
                self.active_tab = WorkspaceTab::GlobalWanFederation;
                self.lan_peer_input = "192.168.1.10:7443".to_string();
                self.wan_sync_url_input = "https://fold.crvscience.org/beofold/api/v1".to_string();
                let (best_wu, best_e) = self
                    .live_state
                    .lock()
                    .map(|g| (g.best_wu, g.best_potential_kcal))
                    .unwrap_or((WorkUnitId::new(1, 0, 0, 10), -112.40));
                if let Ok(mut net) = self.net_state.lock() {
                    net.lan_target = self.lan_peer_input.clone();
                    net.lan_connected = true;
                    net.lan_last_error = None;
                    net.lan_snapshot = Some(ClusterStateSnapshot {
                        raft_term: 14,
                        leader: Some(PeerEndpoint {
                            node_pubkey: [0xA1; 32],
                            addr: "192.168.1.10:7443".to_string(),
                        }),
                        active_project_id: Some(1),
                        active_project_name: Some("Trp-Cage (1L2Y) MREMD".to_string()),
                        peers: vec![
                            LivePeerStatus {
                                capability: NodeCapabilityProfile {
                                    node_pubkey: [0xA1; 32],
                                    hostname: "helix-gpu-01".to_string(),
                                    os: "windows".to_string(),
                                    arch: "x86_64".to_string(),
                                    logical_cores: 24,
                                    total_ram_mb: 65_536,
                                    villin_ns_per_day: 68.40,
                                    tier: ComputeTier::TierS,
                                },
                                addr: "192.168.1.10:7443".to_string(),
                                active_wu: Some(WorkUnitId::new(1, 0, 0, 11)),
                                step_progress: 740_000,
                                target_steps: 1_000_000,
                                current_pot_kcal: best_e,
                                cpu_usage_pct: 64.0,
                                throttled: false,
                                last_seen_unix_sec: 1_763_180_000,
                            },
                            LivePeerStatus {
                                capability: NodeCapabilityProfile {
                                    node_pubkey: [0xB2; 32],
                                    hostname: "rack-epyc-02".to_string(),
                                    os: "linux".to_string(),
                                    arch: "x86_64".to_string(),
                                    logical_cores: 32,
                                    total_ram_mb: 131_072,
                                    villin_ns_per_day: 29.15,
                                    tier: ComputeTier::TierA,
                                },
                                addr: "192.168.1.11:7443".to_string(),
                                active_wu: Some(WorkUnitId::new(1, 0, 1, 11)),
                                step_progress: 185_000,
                                target_steps: 250_000,
                                current_pot_kcal: best_e + 4.82,
                                cpu_usage_pct: 91.0,
                                throttled: false,
                                last_seen_unix_sec: 1_763_180_000,
                            },
                            LivePeerStatus {
                                capability: NodeCapabilityProfile {
                                    node_pubkey: [0xC3; 32],
                                    hostname: "studio-m3-03".to_string(),
                                    os: "macos".to_string(),
                                    arch: "aarch64".to_string(),
                                    logical_cores: 14,
                                    total_ram_mb: 32_768,
                                    villin_ns_per_day: 19.60,
                                    tier: ComputeTier::TierA,
                                },
                                addr: "192.168.1.12:7443".to_string(),
                                active_wu: Some(WorkUnitId::new(1, 0, 2, 11)),
                                step_progress: 162_500,
                                target_steps: 250_000,
                                current_pot_kcal: best_e + 9.15,
                                cpu_usage_pct: 78.0,
                                throttled: false,
                                last_seen_unix_sec: 1_763_180_000,
                            },
                            LivePeerStatus {
                                capability: NodeCapabilityProfile {
                                    node_pubkey: [0xD4; 32],
                                    hostname: "lab-nuc-04".to_string(),
                                    os: "linux".to_string(),
                                    arch: "x86_64".to_string(),
                                    logical_cores: 8,
                                    total_ram_mb: 16_384,
                                    villin_ns_per_day: 6.45,
                                    tier: ComputeTier::TierB,
                                },
                                addr: "192.168.1.13:7443".to_string(),
                                active_wu: Some(WorkUnitId::new(1, 0, 3, 11)),
                                step_progress: 34_000,
                                target_steps: 50_000,
                                current_pot_kcal: best_e + 14.30,
                                cpu_usage_pct: 82.0,
                                throttled: false,
                                last_seen_unix_sec: 1_763_180_000,
                            },
                        ],
                        total_workunits: 128,
                        queued_workunits: 40,
                        leased_workunits: 4,
                        completed_workunits: 84,
                        mremd_swaps: 42,
                        mremd_trials: 112,
                        quarantined_events: 0,
                        lowest_potential_kcal: Some((best_wu, best_e)),
                    });

                    let cluster_specs = [
                        ("crv-pacific-core", 0xA1_u8, 6, 3, 2, 1, 148.60, "win64/linux_x86_64", true, 412_u64, 0_u64),
                        ("stanford-fold-lab", 0xB2, 8, 4, 3, 1, 184.20, "linux_x86_64", true, 528, 0),
                        ("zurich-eth-biocomp", 0xC3, 5, 2, 2, 1, 102.45, "linux_x86_64", true, 319, 0),
                        ("kyoto-riken-arm", 0xD4, 6, 1, 4, 1, 76.90, "linux_aarch64", true, 244, 0),
                        ("boston-biotech-gpu", 0xE5, 4, 3, 1, 0, 134.80, "win64/linux", true, 381, 0),
                        ("austin-tacc-epyc", 0xF6, 7, 1, 5, 1, 91.15, "linux_x86_64", false, 198, 1),
                        ("berlin-mpi-cluster", 0x17, 4, 1, 2, 1, 58.30, "linux_x86_64", true, 176, 0),
                        ("toronto-vector-node", 0x28, 3, 2, 1, 0, 88.75, "linux_x86_64", true, 265, 0),
                        ("cambridge-biomesh", 0x39, 5, 1, 3, 1, 64.40, "linux/macos", false, 142, 0),
                        ("seattle-garage-rack", 0x4A, 4, 1, 2, 1, 51.20, "win64", true, 129, 0),
                        ("tokyo-home-m3max", 0x5B, 3, 0, 2, 1, 34.85, "macos_aarch64", false, 94, 1),
                        ("sydney-unsw-grid", 0x6C, 4, 1, 2, 1, 47.60, "linux_x86_64", false, 118, 0),
                    ];
                    let clusters: Vec<RegisteredClusterRecord> = cluster_specs
                        .iter()
                        .map(|&(name, kb, tot, ts, ta, tb, rate, os, trusted, done, quar)| {
                            RegisteredClusterRecord {
                                cluster_pubkey: [kb; 32],
                                cluster_name: name.to_string(),
                                total_nodes: tot,
                                tier_s_nodes: ts,
                                tier_a_nodes: ta,
                                tier_b_nodes: tb,
                                aggregate_ns_per_day: rate,
                                os_summary: os.to_string(),
                                trusted,
                                completed_units: done,
                                quarantined_units: quar,
                                last_seen_unix_sec: 1_763_180_000,
                            }
                        })
                        .collect();
                    let agg_rate: f64 = clusters.iter().map(|c| c.aggregate_ns_per_day).sum();
                    net.wan_sync_url = self.wan_sync_url_input.clone();
                    net.wan_connected = true;
                    net.wan_last_error = None;
                    net.wan_snapshot = Some(GlobalFederationTelemetry {
                        server_unix_sec: 1_763_180_000,
                        quorum_mode: 2,
                        clusters,
                        aggregate_ns_per_day: agg_rate,
                        active_projects: vec![1, 2],
                        total_workunits: 3600,
                        queued_workunits: 512,
                        leased_workunits: 64,
                        quorum_pending_workunits: 18,
                        completed_workunits: 3006,
                        mremd_swaps: 1420,
                        mremd_trials: 3890,
                        quarantined_events: 2,
                        canary_audits_passed: 284,
                        lowest_potential_kcal: Some((best_wu, best_e)),
                    });
                }
            }
            _ => {
                self.active_tab = WorkspaceTab::FoldingStudio3D;
                self.compare_start_vs_folded = true;
                self.viewport.show_best_conformation = false;
                self.viewport.auto_spin = false;
                self.viewport.zoom = 1.18;
            }
        }
    }

    pub fn render_ui(&mut self, ctx: &egui::Context) {
        if self.show_about && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.show_about = false;
        } else if !ctx.wants_keyboard_input() && ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.show_about = !self.show_about;
        }

        let net_snap = self
            .net_state
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        let live_snap = self
            .live_state
            .lock()
            .expect("live folding state mutex poisoned")
            .clone();

        egui::TopBottomPanel::top("beofold_top_bar").show(ctx, |ui| {
            self.render_top_bar(ui, &net_snap);
        });

        egui::TopBottomPanel::bottom("beofold_status_footer").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("About")
                        .on_hover_text("About BeoFold & CRV Science Network (F1)")
                        .clicked()
                    {
                        self.show_about = !self.show_about;
                    }
                    ui.separator();
                    ui.label(
                        RichText::new("Copyright CRV Science Network")
                            .monospace()
                            .size(11.0)
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.separator();
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(
                            egui::Label::new(
                                RichText::new(&live_snap.status_banner)
                                    .monospace()
                                    .size(11.5)
                                    .color(Color32::from_rgb(180, 194, 212)),
                            )
                            .truncate(),
                        );
                    });
                });
            });
        });

        if self.show_about {
            egui::Window::new("About BeoFold")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .open(&mut self.show_about)
                .show(ctx, |ui| {
                    ui.heading(concat!("BeoFold v", env!("CARGO_PKG_VERSION")));
                    ui.label("Distributed Protein Folding & Markov State Model Supervisor");
                    ui.add_space(6.0);
                    ui.label("Copyright (C) 2026 CRV Science Network");
                    ui.label("License: GNU General Public License v3.0 or later (GPL-3.0-or-later)");
                    ui.add_space(6.0);
                    ui.hyperlink_to("CRV Science Network (https://crvscience.com)", "https://crvscience.com");
                    ui.hyperlink_to("CRV Science Apps Hub (https://crvscience.com/apps)", "https://crvscience.com/apps");
                });
        }

        match self.active_tab {
            WorkspaceTab::FoldingStudio3D => self.render_studio_tab(ctx, &live_snap),
            WorkspaceTab::LanBeowulfMatrix => self.render_lan_matrix_tab(ctx, &net_snap),
            WorkspaceTab::GlobalWanFederation => self.render_wan_federation_tab(ctx, &net_snap),
            WorkspaceTab::MsmFreeEnergyFlux => {
                if self.msm_tab.render(
                    ctx,
                    &live_snap,
                    &mut self.vault_dir_input,
                    &self.sandbox_tx,
                ) {
                    self.viewport.show_best_conformation = false;
                    self.active_tab = WorkspaceTab::FoldingStudio3D;
                }
            }
        }

        if live_snap.running || self.viewport.auto_spin || self.msm_tab.is_animating() {
            ctx.request_repaint_after(Duration::from_millis(33));
        } else {
            ctx.request_repaint_after(Duration::from_millis(500));
        }
    }
}

impl eframe::App for BeoFoldGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render_ui(ctx);
    }
}

impl Drop for BeoFoldGuiApp {
    fn drop(&mut self) {
        let _ = self.sandbox_tx.send(SandboxCommand::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_banner_color_classifies_error_amber_and_default() {
        let default_ok = Color32::from_rgb(52, 211, 153);
        let red = Color32::from_rgb(248, 113, 113);
        let amber = Color32::from_rgb(251, 191, 36);

        assert_eq!(
            classify_status_color("Export error: output path cannot be empty.", default_ok),
            red
        );
        assert_eq!(
            classify_status_color("Export FAILED: permission denied", default_ok),
            red
        );
        assert_eq!(
            classify_status_color(
                "Live studio currently has 4 conformations (need >= 8).",
                default_ok
            ),
            amber
        );
        assert_eq!(
            classify_status_color("Saved PDB to ./beofold_supervisor_best.pdb", default_ok),
            default_ok
        );
        assert_eq!(
            non_empty_trimmed_path("   "),
            Err("Export error: output path cannot be empty.".to_string())
        );
        assert_eq!(
            non_empty_trimmed_path("  ./out.pdb "),
            Ok(PathBuf::from("./out.pdb"))
        );
    }

    #[test]
    fn f1_and_escape_toggle_about_modal_headlessly() {
        let ctx = egui::Context::default();
        let mut app = BeoFoldGuiApp::new_with_ctx(
            &ctx,
            GuiLaunchConfig {
                lan_peer: String::new(),
                cluster_secret: "test".to_string(),
                sync_url: String::new(),
                vault_dir: "./beofold_data".to_string(),
                use_gpu: false,
            },
        );
        assert!(!app.show_about);

        let run_key = |app: &mut BeoFoldGuiApp, key: egui::Key| {
            let mut raw = egui::RawInput::default();
            raw.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            ));
            raw.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            });
            let _ = ctx.run(raw, |c| app.render_ui(c));
        };

        run_key(&mut app, egui::Key::F1);
        assert!(app.show_about, "F1 should open About modal");

        run_key(&mut app, egui::Key::Escape);
        assert!(!app.show_about, "Escape should dismiss About modal");

        run_key(&mut app, egui::Key::F1);
        assert!(app.show_about, "F1 should reopen About modal");

        run_key(&mut app, egui::Key::F1);
        assert!(!app.show_about, "Second F1 should toggle About modal closed");
    }
}
