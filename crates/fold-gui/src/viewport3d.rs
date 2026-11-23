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

use egui::{
    Align2, Color32, CornerRadius, FontId, PointerButton, Pos2, Sense, Stroke, StrokeKind,
    Ui, Vec2,
};
use fold_physics::ProteinTopology;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRepresentation {
    BackboneAndSidechains,
    CAlphaRibbonOnly,
    AllHeavyAtoms,
}

impl RenderRepresentation {
    pub fn label(self) -> &'static str {
        match self {
            RenderRepresentation::BackboneAndSidechains => "Backbone + Sidechain Beads",
            RenderRepresentation::CAlphaRibbonOnly => "C-Alpha Trace",
            RenderRepresentation::AllHeavyAtoms => "All Heavy Atoms",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConformationViewport {
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub zoom: f32,
    pub pan: Vec2,
    pub auto_spin: bool,
    pub show_best_conformation: bool,
    pub representation: RenderRepresentation,
}

impl Default for ConformationViewport {
    fn default() -> Self {
        Self {
            yaw_rad: 0.55,
            pitch_rad: 0.32,
            zoom: 1.15,
            pan: Vec2::ZERO,
            auto_spin: true,
            show_best_conformation: false,
            representation: RenderRepresentation::BackboneAndSidechains,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
}

enum DepthPrimitive {
    Bond {
        z_mid: f32,
        p0: Pos2,
        p1: Pos2,
        width: f32,
        color: Color32,
    },
    Ribbon {
        z_mid: f32,
        p0: Pos2,
        p1: Pos2,
        width: f32,
        color: Color32,
        highlight: Color32,
    },
    Sphere {
        z_depth: f32,
        screen: Pos2,
        radius: f32,
        color: Color32,
        atom_idx: usize,
    },
}

impl ConformationViewport {
    pub fn reset_camera(&mut self) {
        self.yaw_rad = 0.55;
        self.pitch_rad = 0.32;
        self.zoom = 1.15;
        self.pan = Vec2::ZERO;
    }

    pub fn ui(
        &mut self,
        ui: &mut Ui,
        topology: &ProteinTopology,
        positions: &[[f32; 3]],
        subtitle: &str,
    ) {
        self.ui_with_scale(ui, topology, positions, subtitle, None);
    }

    pub fn ui_with_scale(
        &mut self,
        ui: &mut Ui,
        topology: &ProteinTopology,
        positions: &[[f32; 3]],
        subtitle: &str,
        extent_override: Option<f32>,
    ) {
        let avail = ui.available_size();
        let canvas_size = Vec2::new(avail.x.max(280.0), (avail.y - 4.0).max(260.0));
        let (response, painter) = ui.allocate_painter(canvas_size, Sense::click_and_drag());
        let rect = response.rect;

        painter.rect_filled(rect, CornerRadius::same(8), Color32::from_rgb(13, 18, 26));
        painter.rect_stroke(
            rect,
            CornerRadius::same(8),
            Stroke::new(1.0_f32, Color32::from_rgb(42, 54, 72)),
            StrokeKind::Inside,
        );

        if response.dragged_by(PointerButton::Primary) {
            let d = response.drag_delta();
            self.yaw_rad += d.x * 0.011;
            self.pitch_rad = (self.pitch_rad + d.y * 0.011).clamp(-1.48, 1.48);
            self.auto_spin = false;
        } else if response.dragged_by(PointerButton::Secondary)
            || response.dragged_by(PointerButton::Middle)
        {
            self.pan += response.drag_delta();
        }

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                self.zoom = (self.zoom * (1.0 + scroll * 0.0018)).clamp(0.35, 4.5);
            }
        }

        if self.auto_spin {
            self.yaw_rad += 0.009;
        }

        if positions.is_empty() || positions.len() != topology.atoms.len() {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "No atomic coordinates loaded",
                FontId::monospace(13.0),
                Color32::from_rgb(140, 155, 175),
            );
            return;
        }

        // Center molecule around geometric centroid and compute bounding radius
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        for p in positions {
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        let inv_n = 1.0 / (positions.len() as f32);
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        let mut max_extent = 6.0f32;
        for p in positions {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > max_extent {
                max_extent = dist;
            }
        }

        let (sy, cy_ang) = self.yaw_rad.sin_cos();
        let (sp, cp_ang) = self.pitch_rad.sin_cos();
        let scale_extent = extent_override.unwrap_or(max_extent).max(1.0);
        let cam_dist = scale_extent * 3.1;
        let base_scale = rect.width().min(rect.height()) * 0.40 * self.zoom / scale_extent;
        let screen_origin = rect.center() + self.pan;

        let project_point = |p: [f32; 3]| -> (Pos2, f32, f32) {
            let x0 = p[0] - cx;
            let y0 = p[1] - cy;
            let z0 = p[2] - cz;

            // Yaw around Y, then Pitch around X
            let x1 = x0 * cy_ang + z0 * sy;
            let z1 = -x0 * sy + z0 * cy_ang;
            let y2 = y0 * cp_ang - z1 * sp;
            let z2 = y0 * sp + z1 * cp_ang;

            let persp = cam_dist / (cam_dist - z2).max(2.0);
            let sx = screen_origin.x + x1 * base_scale * persp;
            let sy_scr = screen_origin.y - y2 * base_scale * persp;
            (Pos2::new(sx, sy_scr), z2, persp)
        };

        // Project every atom into `(screen_pos, view_z, perspective_factor)`
        let mut projected: Vec<(Pos2, f32, f32)> = Vec::with_capacity(positions.len());
        for &p in positions {
            projected.push(project_point(p));
        }

        let mut prims: Vec<DepthPrimitive> = Vec::with_capacity(positions.len() * 6);

        // 1. Sequential C-alpha Catmull-Rom spline ribbon colored by secondary structure (Helix/Sheet/Coil)
        let ca_indices: Vec<usize> = topology
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(idx, a)| if a.is_ca { Some(idx) } else { None })
            .collect();
        let ss_classes = classify_secondary_structure(topology, positions, &ca_indices);

        let ca_len = ca_indices.len();
        const SPLINE_SUBSTEPS: usize = 6;
        for w_idx in 0..ca_len.saturating_sub(1) {
            let idx1 = ca_indices[w_idx];
            let idx2 = ca_indices[w_idx + 1];
            let p1 = positions[idx1];
            let p2 = positions[idx2];
            let p0 = if w_idx > 0 {
                positions[ca_indices[w_idx - 1]]
            } else {
                [
                    2.0 * p1[0] - p2[0],
                    2.0 * p1[1] - p2[1],
                    2.0 * p1[2] - p2[2],
                ]
            };
            let p3 = if w_idx + 2 < ca_len {
                positions[ca_indices[w_idx + 2]]
            } else {
                [
                    2.0 * p2[0] - p1[0],
                    2.0 * p2[1] - p1[1],
                    2.0 * p2[2] - p1[2],
                ]
            };

            let ss_seg = if ss_classes[w_idx] == ss_classes[w_idx + 1] {
                ss_classes[w_idx]
            } else if ss_classes[w_idx] != SecondaryStructure::Coil {
                ss_classes[w_idx]
            } else {
                ss_classes[w_idx + 1]
            };
            let base_col = secondary_structure_color(ss_seg);
            let width_factor = match (self.representation, ss_seg) {
                (RenderRepresentation::CAlphaRibbonOnly, SecondaryStructure::Helix) => 6.4,
                (RenderRepresentation::CAlphaRibbonOnly, SecondaryStructure::Sheet) => 5.8,
                (RenderRepresentation::CAlphaRibbonOnly, SecondaryStructure::Coil) => 3.4,
                (RenderRepresentation::BackboneAndSidechains, SecondaryStructure::Helix) => 5.0,
                (RenderRepresentation::BackboneAndSidechains, SecondaryStructure::Sheet) => 4.6,
                (RenderRepresentation::BackboneAndSidechains, SecondaryStructure::Coil) => 3.0,
                (RenderRepresentation::AllHeavyAtoms, _) => 1.8,
            };

            let mut prev_proj = project_point(p1);
            for sub in 1..=SPLINE_SUBSTEPS {
                let t = (sub as f32) / (SPLINE_SUBSTEPS as f32);
                let pt_3d = catmull_rom_point(p0, p1, p2, p3, t);
                let cur_proj = project_point(pt_3d);
                let z_mid = 0.5 * (prev_proj.1 + cur_proj.1);
                let persp_mid = 0.5 * (prev_proj.2 + cur_proj.2);
                let body_col = depth_cue_color(base_col, z_mid, max_extent);
                let hi_col = ribbon_highlight_color(body_col);

                if self.representation == RenderRepresentation::AllHeavyAtoms {
                    prims.push(DepthPrimitive::Bond {
                        z_mid,
                        p0: prev_proj.0,
                        p1: cur_proj.0,
                        width: (width_factor * persp_mid).clamp(1.2, 4.0),
                        color: body_col,
                    });
                } else {
                    prims.push(DepthPrimitive::Ribbon {
                        z_mid,
                        p0: prev_proj.0,
                        p1: cur_proj.0,
                        width: (width_factor * persp_mid).clamp(1.6, 11.5),
                        color: body_col,
                        highlight: hi_col,
                    });
                }
                prev_proj = cur_proj;
            }
        }

        // 2. Sidechain C-alpha -> C-beta stems or full covalent bond sticks
        match self.representation {
            RenderRepresentation::CAlphaRibbonOnly => {
                for (order, &idx) in ca_indices.iter().enumerate() {
                    let col = depth_cue_color(
                        secondary_structure_color(ss_classes[order]),
                        projected[idx].1,
                        max_extent,
                    );
                    prims.push(DepthPrimitive::Sphere {
                        z_depth: projected[idx].1,
                        screen: projected[idx].0,
                        radius: (3.6 * projected[idx].2 * self.zoom.sqrt()).clamp(2.0, 9.5),
                        color: col,
                        atom_idx: idx,
                    });
                }
            }
            RenderRepresentation::BackboneAndSidechains => {
                for b in &topology.bonds {
                    let i = b.i as usize;
                    let j = b.j as usize;
                    let ai = &topology.atoms[i];
                    let aj = &topology.atoms[j];
                    if (ai.is_ca && !aj.is_backbone) || (aj.is_ca && !ai.is_backbone) {
                        let z_mid = 0.5 * (projected[i].1 + projected[j].1);
                        let p_mid = 0.5 * (projected[i].2 + projected[j].2);
                        prims.push(DepthPrimitive::Bond {
                            z_mid,
                            p0: projected[i].0,
                            p1: projected[j].0,
                            width: (2.2 * p_mid).clamp(1.0, 4.5),
                            color: depth_cue_color(
                                Color32::from_rgb(148, 163, 184),
                                z_mid,
                                max_extent,
                            ),
                        });
                    }
                }
                for (idx, atom) in topology.atoms.iter().enumerate() {
                    if !atom.is_ca && atom.is_backbone {
                        continue;
                    }
                    let base_col = residue_biophysical_color(&atom.res_name, atom.is_ca);
                    let rad_base = if atom.is_ca {
                        3.8
                    } else {
                        atom.lj_sigma_ang * 1.35
                    };
                    prims.push(DepthPrimitive::Sphere {
                        z_depth: projected[idx].1,
                        screen: projected[idx].0,
                        radius: (rad_base * projected[idx].2 * self.zoom.sqrt()).clamp(2.2, 16.0),
                        color: depth_cue_color(base_col, projected[idx].1, max_extent),
                        atom_idx: idx,
                    });
                }
            }
            RenderRepresentation::AllHeavyAtoms => {
                for b in &topology.bonds {
                    let i = b.i as usize;
                    let j = b.j as usize;
                    let z_mid = 0.5 * (projected[i].1 + projected[j].1);
                    let p_mid = 0.5 * (projected[i].2 + projected[j].2);
                    prims.push(DepthPrimitive::Bond {
                        z_mid,
                        p0: projected[i].0,
                        p1: projected[j].0,
                        width: (2.0 * p_mid).clamp(1.0, 4.2),
                        color: depth_cue_color(
                            Color32::from_rgb(130, 145, 168),
                            z_mid,
                            max_extent,
                        ),
                    });
                }
                for (idx, atom) in topology.atoms.iter().enumerate() {
                    let base_col = element_cpk_color(&atom.element, &atom.res_name, atom.is_backbone);
                    let rad_base = if atom.is_backbone { 3.4 } else { 4.6 };
                    prims.push(DepthPrimitive::Sphere {
                        z_depth: projected[idx].1,
                        screen: projected[idx].0,
                        radius: (rad_base * projected[idx].2 * self.zoom.sqrt()).clamp(2.0, 14.0),
                        color: depth_cue_color(base_col, projected[idx].1, max_extent),
                        atom_idx: idx,
                    });
                }
            }
        }

        // Painter's back-to-front depth sort (`z` ascending = farthest to nearest)
        prims.sort_by(|a, b| {
            let za = match a {
                DepthPrimitive::Bond { z_mid, .. } => *z_mid,
                DepthPrimitive::Ribbon { z_mid, .. } => *z_mid,
                DepthPrimitive::Sphere { z_depth, .. } => *z_depth,
            };
            let zb = match b {
                DepthPrimitive::Bond { z_mid, .. } => *z_mid,
                DepthPrimitive::Ribbon { z_mid, .. } => *z_mid,
                DepthPrimitive::Sphere { z_depth, .. } => *z_depth,
            };
            za.partial_cmp(&zb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let hover_pos = response.hover_pos();
        let mut hovered_atom: Option<(usize, Pos2)> = None;

        for prim in prims {
            match prim {
                DepthPrimitive::Bond {
                    p0,
                    p1,
                    width,
                    color,
                    ..
                } => {
                    painter.line_segment([p0, p1], Stroke::new(width, color));
                }
                DepthPrimitive::Ribbon {
                    p0,
                    p1,
                    width,
                    color,
                    highlight,
                    ..
                } => {
                    let outline = Color32::from_rgba_unmultiplied(10, 14, 22, 210);
                    painter.circle_filled(p0, (width + 1.4) * 0.5, outline);
                    painter.line_segment([p0, p1], Stroke::new(width + 1.4, outline));
                    painter.circle_filled(p0, width * 0.5, color);
                    painter.line_segment([p0, p1], Stroke::new(width, color));
                    painter.line_segment([p0, p1], Stroke::new((width * 0.28).max(0.8), highlight));
                }
                DepthPrimitive::Sphere {
                    screen,
                    radius,
                    color,
                    atom_idx,
                    ..
                } => {
                    painter.circle_filled(screen, radius, color);
                    painter.circle_stroke(
                        screen,
                        radius,
                        Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(10, 14, 20, 180)),
                    );
                    // Subtle top-left specular highlight bead
                    let spec_pos = Pos2::new(screen.x - radius * 0.28, screen.y - radius * 0.28);
                    painter.circle_filled(
                        spec_pos,
                        (radius * 0.28).max(0.8),
                        Color32::from_rgba_unmultiplied(255, 255, 255, 70),
                    );

                    if let Some(hp) = hover_pos {
                        if hp.distance(screen) <= radius + 3.0 {
                            hovered_atom = Some((atom_idx, screen));
                        }
                    }
                }
            }
        }

        // HUD overlay legend & camera hint (clipped to rect and responsive across all panel widths)
        let hud_painter = painter.with_clip_rect(rect);
        let legend_layout = compute_hud_legend_layout(rect.width());
        let header_bottom_y = if legend_layout.two_line_header {
            hud_painter.text(
                Pos2::new(rect.left() + 10.0, rect.top() + 8.0),
                Align2::LEFT_TOP,
                subtitle,
                FontId::monospace(11.0),
                Color32::from_rgb(226, 232, 240),
            );
            hud_painter.text(
                Pos2::new(rect.left() + 10.0, rect.top() + 23.0),
                Align2::LEFT_TOP,
                format!(
                    "{} ({} aa, {} atoms)",
                    topology.name,
                    topology.residue_count,
                    topology.atoms.len()
                ),
                FontId::monospace(10.0),
                Color32::from_rgb(148, 163, 184),
            );
            rect.top() + 37.0
        } else {
            hud_painter.text(
                Pos2::new(rect.left() + 12.0, rect.top() + 10.0),
                Align2::LEFT_TOP,
                format!(
                    "{} ({} aa, {} atoms) — {}",
                    topology.name,
                    topology.residue_count,
                    topology.atoms.len(),
                    subtitle
                ),
                FontId::monospace(12.0),
                Color32::from_rgb(226, 232, 240),
            );
            rect.top() + 26.0
        };

        let legend_y = rect.bottom() - legend_layout.bottom_offset;
        for (line_idx, line) in legend_layout.legend_lines.iter().enumerate() {
            hud_painter.text(
                Pos2::new(
                    rect.left() + legend_layout.pad_x,
                    legend_y + (line_idx as f32) * 12.5,
                ),
                Align2::LEFT_TOP,
                *line,
                FontId::monospace(legend_layout.font_size),
                Color32::from_rgb(148, 163, 184),
            );
        }

        // Angstrom scale bar & XYZ orientation triad above legend on bottom-right
        let (bar_ang, bar_px) = choose_scale_bar_angstrom(base_scale);
        let scale_y = legend_y - 16.0;
        let scale_x1 = rect.right() - 14.0;
        let scale_x0 = (scale_x1 - bar_px).max(rect.left() + 14.0);
        let bar_stroke = Stroke::new(1.8_f32, Color32::from_rgb(203, 213, 225));
        hud_painter.line_segment(
            [Pos2::new(scale_x0, scale_y), Pos2::new(scale_x1, scale_y)],
            bar_stroke,
        );
        hud_painter.line_segment(
            [
                Pos2::new(scale_x0, scale_y - 4.0),
                Pos2::new(scale_x0, scale_y + 4.0),
            ],
            bar_stroke,
        );
        hud_painter.line_segment(
            [
                Pos2::new(scale_x1, scale_y - 4.0),
                Pos2::new(scale_x1, scale_y + 4.0),
            ],
            bar_stroke,
        );
        hud_painter.text(
            Pos2::new(0.5 * (scale_x0 + scale_x1), scale_y - 5.0),
            Align2::CENTER_BOTTOM,
            format!("{:.0} \u{00C5}", bar_ang),
            FontId::monospace(10.0),
            Color32::from_rgb(226, 232, 240),
        );

        let triad_center = Pos2::new(rect.right() - 36.0, scale_y - 44.0);
        let triad_len = 20.0f32;
        let mut axes = [
            (
                "X",
                [1.0f32, 0.0, 0.0],
                Color32::from_rgb(248, 113, 113),
            ),
            (
                "Y",
                [0.0f32, 1.0, 0.0],
                Color32::from_rgb(52, 211, 153),
            ),
            (
                "Z",
                [0.0f32, 0.0, 1.0],
                Color32::from_rgb(56, 189, 248),
            ),
        ]
        .map(|(lbl, v, col)| {
            let x1 = v[0] * cy_ang + v[2] * sy;
            let z1 = -v[0] * sy + v[2] * cy_ang;
            let y2 = v[1] * cp_ang - z1 * sp;
            let z2 = v[1] * sp + z1 * cp_ang;
            (lbl, x1, y2, z2, col)
        });
        axes.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
        hud_painter.circle_filled(triad_center, 2.0, Color32::from_rgb(203, 213, 225));
        for (lbl, ax, ay, _, col) in axes {
            let tip = Pos2::new(
                triad_center.x + ax * triad_len,
                triad_center.y - ay * triad_len,
            );
            hud_painter.line_segment([triad_center, tip], Stroke::new(1.8_f32, col));
            hud_painter.circle_filled(tip, 2.2, col);
            let lbl_pos = Pos2::new(
                triad_center.x + ax * (triad_len + 8.5),
                triad_center.y - ay * (triad_len + 8.5),
            );
            hud_painter.text(lbl_pos, Align2::CENTER_CENTER, lbl, FontId::monospace(9.5), col);
        }

        // Interactive atom/residue tooltip on hover (positioned cleanly below header)
        if let Some((idx, scr)) = hovered_atom {
            let a = &topology.atoms[idx];
            let p = positions[idx];
            hud_painter.circle_stroke(scr, 10.0, Stroke::new(1.8_f32, Color32::WHITE));
            let tip = format!(
                "{}{} {} (#{}) | ({:.2}, {:.2}, {:.2}) A | q={:+.2}e | sigma={:.2}A",
                a.res_name, a.res_seq, a.name, a.serial, p[0], p[1], p[2], a.charge_e, a.lj_sigma_ang
            );
            let tip_pos = Pos2::new(rect.left() + 12.0, header_bottom_y + 4.0);
            hud_painter.text(
                tip_pos,
                Align2::LEFT_TOP,
                tip,
                FontId::monospace(11.0),
                Color32::from_rgb(56, 189, 248),
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HudLegendLayout {
    pub two_line_header: bool,
    pub pad_x: f32,
    pub bottom_offset: f32,
    pub font_size: f32,
    pub legend_lines: Vec<&'static str>,
    pub estimated_max_line_px: f32,
}

pub(crate) fn compute_hud_legend_layout(rect_width: f32) -> HudLegendLayout {
    if rect_width >= 760.0 {
        let line = "Ribbon: Helix(Rose) Sheet(Sky) Coil(Slate) | Beads: Hydrophobic(Gold) +K/R(Sky) -D/E(Coral) Polar(Teal)";
        let font_size = 9.5;
        let est = (line.len() as f32) * (font_size * 0.60);
        HudLegendLayout {
            two_line_header: false,
            pad_x: 12.0,
            bottom_offset: 22.0,
            font_size,
            legend_lines: vec![line],
            estimated_max_line_px: est,
        }
    } else if rect_width >= 380.0 {
        let line = "Ribbon: Helix/Sheet/Coil | Gold:Hydro +K/R -D/E";
        let font_size = 9.5;
        let est = (line.len() as f32) * (font_size * 0.60);
        HudLegendLayout {
            two_line_header: true,
            pad_x: 10.0,
            bottom_offset: 18.0,
            font_size,
            legend_lines: vec![line],
            estimated_max_line_px: est,
        }
    } else {
        let line1 = "Ribbon: Helix/Sheet/Coil";
        let line2 = "Gold:Hydro +K/R -D/E Teal:Polar";
        let font_size = 9.0;
        let max_chars = line1.len().max(line2.len()) as f32;
        HudLegendLayout {
            two_line_header: true,
            pad_x: 10.0,
            bottom_offset: 30.0,
            font_size,
            legend_lines: vec![line1, line2],
            estimated_max_line_px: max_chars * (font_size * 0.60),
        }
    }
}

/// Evaluates uniform Catmull-Rom cubic spline position at `t in [0, 1]` between `p1` and `p2`.
#[inline]
pub(crate) fn catmull_rom_point(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    t: f32,
) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0f32; 3];
    for d in 0..3 {
        out[d] = 0.5
            * ((2.0 * p1[d])
                + (-p0[d] + p2[d]) * t
                + (2.0 * p0[d] - 5.0 * p1[d] + 4.0 * p2[d] - p3[d]) * t2
                + (-p0[d] + 3.0 * p1[d] - 3.0 * p2[d] + p3[d]) * t3);
    }
    out
}

/// Selects a clean Ångström scale bar length (`1, 2, 5, 10, 20, 50 Å`) targeting ~55–100 px.
pub(crate) fn choose_scale_bar_angstrom(px_per_ang: f32) -> (f32, f32) {
    let scale = px_per_ang.max(0.25);
    let target_px = 72.0f32;
    let candidates = [1.0f32, 2.0, 5.0, 10.0, 20.0, 50.0];
    let mut best_ang = 5.0f32;
    let mut best_err = f32::INFINITY;
    for &c in &candidates {
        let px = c * scale;
        let penalty = if px < 28.0 || px > 125.0 { 200.0 } else { 0.0 };
        let err = (px - target_px).abs() + penalty;
        if err < best_err {
            best_err = err;
            best_ang = c;
        }
    }
    (best_ang, (best_ang * scale).clamp(20.0, 140.0))
}

#[inline]
fn dist3_f32(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn dihedral_deg(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> Option<f32> {
    let b0 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let b1 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let b2 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];
    let b1_len = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
    if b1_len < 1.0e-4 {
        return None;
    }
    let b1n = [b1[0] / b1_len, b1[1] / b1_len, b1[2] / b1_len];
    let dot = |u: [f32; 3], v: [f32; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = |u: [f32; 3], v: [f32; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let v = [
        b0[0] - dot(b0, b1n) * b1n[0],
        b0[1] - dot(b0, b1n) * b1n[1],
        b0[2] - dot(b0, b1n) * b1n[2],
    ];
    let w = [
        b2[0] - dot(b2, b1n) * b1n[0],
        b2[1] - dot(b2, b1n) * b1n[1],
        b2[2] - dot(b2, b1n) * b1n[2],
    ];
    let x = dot(v, w);
    let y = dot(cross(b1n, v), w);
    if x.abs() < 1.0e-6 && y.abs() < 1.0e-6 {
        return None;
    }
    Some(y.atan2(x).to_degrees())
}

/// Classifies each C-alpha residue into Helix, Sheet, or Coil using backbone `(phi, psi)`
/// dihedrals plus local C-alpha geometry (`d(i, i+2)`, `d(i, i+3)`, virtual torsion) and
/// minimum run-length smoothing (`>= 3` for helices, `>= 2` for sheets).
pub(crate) fn classify_secondary_structure(
    topology: &ProteinTopology,
    positions: &[[f32; 3]],
    ca_indices: &[usize],
) -> Vec<SecondaryStructure> {
    let m = ca_indices.len();
    let mut raw = vec![SecondaryStructure::Coil; m];
    if m < 4 {
        return raw;
    }

    // Locate backbone N and C positions matching each CA residue
    let mut n_pos: Vec<Option<[f32; 3]>> = vec![None; m];
    let mut c_pos: Vec<Option<[f32; 3]>> = vec![None; m];
    for (k, &ca_idx) in ca_indices.iter().enumerate() {
        let ca_atom = &topology.atoms[ca_idx];
        let lo = ca_idx.saturating_sub(4);
        let hi = (ca_idx + 5).min(topology.atoms.len());
        for a_idx in lo..hi {
            let a = &topology.atoms[a_idx];
            if a.chain_id == ca_atom.chain_id && a.res_seq == ca_atom.res_seq {
                if a.name == "N" {
                    n_pos[k] = Some(positions[a_idx]);
                } else if a.name == "C" {
                    c_pos[k] = Some(positions[a_idx]);
                }
            }
        }
    }

    for k in 0..m {
        let ca_k = positions[ca_indices[k]];
        let phi = if k > 0 {
            match (c_pos[k - 1], n_pos[k], c_pos[k]) {
                (Some(pc), Some(nk), Some(ck)) => dihedral_deg(pc, nk, ca_k, ck),
                _ => None,
            }
        } else {
            None
        };
        let psi = if k + 1 < m {
            match (n_pos[k], c_pos[k], n_pos[k + 1]) {
                (Some(nk), Some(ck), Some(nn)) => dihedral_deg(nk, ca_k, ck, nn),
                _ => None,
            }
        } else {
            None
        };

        let d_i_i2 = if k + 2 < m {
            Some(dist3_f32(ca_k, positions[ca_indices[k + 2]]))
        } else if k >= 2 {
            Some(dist3_f32(positions[ca_indices[k - 2]], ca_k))
        } else {
            None
        };
        let d_i_i3 = if k + 3 < m {
            Some(dist3_f32(ca_k, positions[ca_indices[k + 3]]))
        } else if k >= 3 {
            Some(dist3_f32(positions[ca_indices[k - 3]], ca_k))
        } else {
            None
        };
        let ca_tau = if k >= 1 && k + 2 < m {
            dihedral_deg(
                positions[ca_indices[k - 1]],
                ca_k,
                positions[ca_indices[k + 1]],
                positions[ca_indices[k + 2]],
            )
        } else {
            None
        };

        let rama_helix = matches!(
            (phi, psi),
            (Some(ph), Some(ps)) if (-115.0..=-25.0).contains(&ph) && (-80.0..=-5.0).contains(&ps)
        );
        let geom_helix = matches!(
            (d_i_i2, d_i_i3, ca_tau),
            (Some(d2), Some(d3), Some(tau))
                if (4.5..=6.3).contains(&d2)
                    && (4.3..=6.5).contains(&d3)
                    && (10.0..=95.0).contains(&tau)
        );
        let helix_compact = d_i_i3.map(|d3| d3 < 6.8).unwrap_or(true);

        if geom_helix || (rama_helix && helix_compact) {
            raw[k] = SecondaryStructure::Helix;
            continue;
        }

        let has_beta_partner = (0..m).any(|other| {
            other.abs_diff(k) >= 4
                && dist3_f32(ca_k, positions[ca_indices[other]]) <= 6.2
        });
        let rama_beta = matches!(
            (phi, psi),
            (Some(ph), Some(ps)) if (-175.0..=-45.0).contains(&ph) && (ps >= 75.0 || ps <= -145.0)
        );
        let geom_beta = matches!(
            (d_i_i2, ca_tau),
            (Some(d2), Some(tau)) if (5.8..=7.6).contains(&d2) && tau.abs() > 110.0
        );
        if has_beta_partner && (rama_beta || geom_beta) {
            raw[k] = SecondaryStructure::Sheet;
        }
    }

    // Smooth isolated singletons: require run >= 3 for Helix, >= 2 for Sheet
    let mut smoothed = vec![SecondaryStructure::Coil; m];
    let mut idx = 0usize;
    while idx < m {
        let kind = raw[idx];
        let mut end = idx + 1;
        while end < m && raw[end] == kind {
            end += 1;
        }
        let run_len = end - idx;
        let keep = match kind {
            SecondaryStructure::Helix => run_len >= 3,
            SecondaryStructure::Sheet => run_len >= 2,
            SecondaryStructure::Coil => false,
        };
        if keep {
            for item in &mut smoothed[idx..end] {
                *item = kind;
            }
        }
        idx = end;
    }
    smoothed
}

fn secondary_structure_color(ss: SecondaryStructure) -> Color32 {
    match ss {
        SecondaryStructure::Helix => Color32::from_rgb(244, 92, 132),
        SecondaryStructure::Sheet => Color32::from_rgb(56, 189, 248),
        SecondaryStructure::Coil => Color32::from_rgb(156, 176, 202),
    }
}

fn ribbon_highlight_color(c: Color32) -> Color32 {
    let lift = |ch: u8| ((ch as u16) + ((255 - (ch as u16)) * 38 / 100)) as u8;
    Color32::from_rgb(lift(c.r()), lift(c.g()), lift(c.b()))
}

fn residue_biophysical_color(res_name: &str, is_ca: bool) -> Color32 {
    if is_ca {
        return Color32::from_rgb(203, 213, 225);
    }
    match res_name {
        "TRP" | "TYR" | "PHE" | "ILE" | "LEU" | "VAL" | "MET" | "ALA" | "PRO" | "CYS" => {
            Color32::from_rgb(245, 190, 65) // Hydrophobic core gold
        }
        "LYS" | "ARG" | "HIS" => Color32::from_rgb(56, 189, 248), // Basic (+)
        "ASP" | "GLU" => Color32::from_rgb(248, 113, 113),        // Acidic (-)
        _ => Color32::from_rgb(45, 212, 191),                     // Polar uncharged
    }
}

fn element_cpk_color(elem: &str, res_name: &str, is_backbone: bool) -> Color32 {
    if !is_backbone {
        return residue_biophysical_color(res_name, false);
    }
    match elem {
        "N" => Color32::from_rgb(96, 165, 250),
        "O" => Color32::from_rgb(248, 113, 113),
        "S" => Color32::from_rgb(250, 204, 21),
        _ => Color32::from_rgb(180, 192, 208),
    }
}

fn depth_cue_color(c: Color32, z: f32, max_extent: f32) -> Color32 {
    let norm = ((z / max_extent.max(1.0)) * 0.30 + 0.82).clamp(0.42, 1.12);
    let scale = |ch: u8| ((ch as f32) * norm).clamp(14.0, 255.0) as u8;
    Color32::from_rgb(scale(c.r()), scale(c.g()), scale(c.b()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_legend_fits_within_viewport_bounds_across_all_width_tiers() {
        for width in [280.0_f32, 340.0, 394.0, 440.0, 580.0, 715.0, 760.0, 1024.0] {
            let layout = compute_hud_legend_layout(width);
            assert!(
                layout.pad_x * 2.0 + layout.estimated_max_line_px <= width + 4.0,
                "legend overflow at width={width}: est={} pad_x={}",
                layout.estimated_max_line_px,
                layout.pad_x
            );
        }
    }

    #[test]
    fn catmull_rom_spline_scale_bar_and_secondary_structure_helpers() {
        let p0 = [0.0f32, 0.0, 0.0];
        let p1 = [1.0, 2.0, 0.0];
        let p2 = [4.0, 3.0, 1.0];
        let p3 = [7.0, 1.0, 2.0];
        assert_eq!(catmull_rom_point(p0, p1, p2, p3, 0.0), p1);
        let end = catmull_rom_point(p0, p1, p2, p3, 1.0);
        for d in 0..3 {
            assert!((end[d] - p2[d]).abs() < 1e-5);
        }
        let mid = catmull_rom_point(p0, p1, p2, p3, 0.5);
        assert!(mid.iter().all(|c| c.is_finite()));

        let (ang, px) = choose_scale_bar_angstrom(14.0);
        assert!([1.0, 2.0, 5.0, 10.0, 20.0, 50.0].contains(&ang));
        assert!(px >= 20.0 && px <= 140.0);

        // Ideal alpha-helix C-alpha trace (radius 2.3 A, rise 1.5 A/res, turn 100 deg/res)
        let (topo, mut coords) =
            ProteinTopology::build_polypeptide_from_sequence("HELIX10", "AAAAAAAAAA");
        let ca_indices: Vec<usize> = topo
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(idx, a)| if a.is_ca { Some(idx) } else { None })
            .collect();
        for (k, &ca_idx) in ca_indices.iter().enumerate() {
            let theta = (k as f32) * (100.0_f32).to_radians();
            coords[ca_idx] = [2.3 * theta.cos(), 2.3 * theta.sin(), (k as f32) * 1.5];
        }
        let ss = classify_secondary_structure(&topo, &coords, &ca_indices);
        assert_eq!(ss.len(), ca_indices.len());
        assert!(
            ss.iter().filter(|&&s| s == SecondaryStructure::Helix).count() >= 4,
            "Ideal alpha-helix geometry must classify >=4 residues as Helix, got {:?}",
            ss
        );
    }
}
