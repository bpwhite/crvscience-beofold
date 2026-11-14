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

enum DepthPrimitive {
    Bond {
        z_mid: f32,
        p0: Pos2,
        p1: Pos2,
        width: f32,
        color: Color32,
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
        let cam_dist = max_extent * 3.1;
        let base_scale = rect.width().min(rect.height()) * 0.40 * self.zoom / max_extent.max(1.0);
        let screen_origin = rect.center() + self.pan;

        // Project every atom into `(screen_pos, view_z, perspective_factor)`
        let mut projected: Vec<(Pos2, f32, f32)> = Vec::with_capacity(positions.len());
        for p in positions {
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
            projected.push((Pos2::new(sx, sy_scr), z2, persp));
        }

        let mut prims: Vec<DepthPrimitive> = Vec::with_capacity(positions.len() * 2);

        // 1. Always build sequential C-alpha backbone trace colored N-terminus -> C-terminus
        let ca_indices: Vec<usize> = topology
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(idx, a)| if a.is_ca { Some(idx) } else { None })
            .collect();

        let ca_total = ca_indices.len().max(2) as f32;
        for w_idx in 0..ca_indices.len().saturating_sub(1) {
            let a = ca_indices[w_idx];
            let b = ca_indices[w_idx + 1];
            let frac = (w_idx as f32) / (ca_total - 1.0);
            let ribbon_col = n_to_c_gradient(frac);
            let z_mid = 0.5 * (projected[a].1 + projected[b].1);
            let persp_mid = 0.5 * (projected[a].2 + projected[b].2);
            let width = match self.representation {
                RenderRepresentation::CAlphaRibbonOnly => 4.2 * persp_mid,
                RenderRepresentation::BackboneAndSidechains => 3.6 * persp_mid,
                RenderRepresentation::AllHeavyAtoms => 1.8 * persp_mid,
            };
            prims.push(DepthPrimitive::Bond {
                z_mid,
                p0: projected[a].0,
                p1: projected[b].0,
                width: width.clamp(1.2, 8.0),
                color: depth_cue_color(ribbon_col, z_mid, max_extent),
            });
        }

        // 2. Sidechain C-alpha -> C-beta stems or full covalent bond sticks
        match self.representation {
            RenderRepresentation::CAlphaRibbonOnly => {
                for (order, &idx) in ca_indices.iter().enumerate() {
                    let frac = (order as f32) / (ca_total - 1.0);
                    let col = depth_cue_color(n_to_c_gradient(frac), projected[idx].1, max_extent);
                    prims.push(DepthPrimitive::Sphere {
                        z_depth: projected[idx].1,
                        screen: projected[idx].0,
                        radius: (4.8 * projected[idx].2 * self.zoom.sqrt()).clamp(2.5, 13.0),
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
                DepthPrimitive::Sphere { z_depth, .. } => *z_depth,
            };
            let zb = match b {
                DepthPrimitive::Bond { z_mid, .. } => *z_mid,
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

        // HUD overlay legend & camera hint
        painter.text(
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

        painter.text(
            Pos2::new(rect.left() + 12.0, rect.bottom() - 22.0),
            Align2::LEFT_TOP,
            "Gold: Hydrophobic Core  |  Sky: Basic (+K/R)  |  Coral: Acidic (-D/E)  |  Teal: Polar  |  Left-Drag: Orbit  Scroll: Zoom",
            FontId::monospace(10.5),
            Color32::from_rgb(148, 163, 184),
        );

        // Interactive atom/residue tooltip on hover
        if let Some((idx, scr)) = hovered_atom {
            let a = &topology.atoms[idx];
            let p = positions[idx];
            painter.circle_stroke(scr, 10.0, Stroke::new(1.8_f32, Color32::WHITE));
            let tip = format!(
                "{}{} {} (#{}) | ({:.2}, {:.2}, {:.2}) A | q={:+.2}e | sigma={:.2}A",
                a.res_name, a.res_seq, a.name, a.serial, p[0], p[1], p[2], a.charge_e, a.lj_sigma_ang
            );
            let tip_pos = Pos2::new(rect.left() + 12.0, rect.top() + 30.0);
            painter.text(
                tip_pos,
                Align2::LEFT_TOP,
                tip,
                FontId::monospace(11.5),
                Color32::from_rgb(56, 189, 248),
            );
        }
    }
}

fn n_to_c_gradient(t: f32) -> Color32 {
    let f = t.clamp(0.0, 1.0);
    // N-terminus cyan-blue (56, 189, 248) -> midpoint emerald (52, 211, 153) -> C-terminus amber (251, 146, 60)
    let (r, g, b) = if f < 0.5 {
        let u = f * 2.0;
        (
            lerp(56.0, 52.0, u),
            lerp(189.0, 211.0, u),
            lerp(248.0, 153.0, u),
        )
    } else {
        let u = (f - 0.5) * 2.0;
        (
            lerp(52.0, 251.0, u),
            lerp(211.0, 146.0, u),
            lerp(153.0, 60.0, u),
        )
    };
    Color32::from_rgb(r as u8, g as u8, b as u8)
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

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
