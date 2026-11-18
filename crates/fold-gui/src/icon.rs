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

use egui::IconData;

fn blend_rgba(bg: [u8; 4], fg: [u8; 3], alpha: f32) -> [u8; 4] {
    let a = alpha.clamp(0.0, 1.0);
    [
        (bg[0] as f32 * (1.0 - a) + fg[0] as f32 * a).round() as u8,
        (bg[1] as f32 * (1.0 - a) + fg[1] as f32 * a).round() as u8,
        (bg[2] as f32 * (1.0 - a) + fg[2] as f32 * a).round() as u8,
        (bg[3] as f32).max(a * 255.0).round() as u8,
    ]
}

/// Generates BeoFold's 64x64 RGBA window/taskbar icon matching `assets/beofold.ico`:
/// a dark slate-blue `#0F141C` badge with folding funnel contours and a cyan `#38BDF8`
/// to emerald `#34D399` protein alpha-helix ribbon.
pub fn beofold_icon_data() -> IconData {
    let size: u32 = 64;
    let s = size as f32;
    let mut rgba = vec![0_u8; (size * size * 4) as usize];

    let rad = s * 0.20;
    let pad = s * 0.04;
    let cx = s * 0.5;
    let cy = s * 0.5;
    let hx = s * 0.5 - pad - rad;
    let hy = s * 0.5 - pad - rad;

    for y in 0..size {
        for x in 0..size {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let dx = ((px - cx).abs() - hx).max(0.0);
            let dy = ((py - cy).abs() - hy).max(0.0);
            let dist = dx.hypot(dy) - rad;

            let col = if dist > 1.0 {
                [0, 0, 0, 0]
            } else {
                let aa = (0.5 - dist).clamp(0.0, 1.0);
                let mut c = if dist >= -s * 0.045 {
                    [42, 54, 72, (255.0 * aa).round() as u8]
                } else {
                    [15, 20, 28, (255.0 * aa).round() as u8]
                };

                let nx = (px - cx) / (s * 0.42);
                let ny = (py - cy) / (s * 0.42);
                for (level, fcol) in [(0.65_f32, [30_u8, 58, 102]), (0.35_f32, [36, 82, 136])] {
                    let fy = 0.62 - level * 0.42 / (nx.abs() + 0.35);
                    let d_fun = (ny - fy).abs() * s * 0.42;
                    if nx.abs() < 0.88 && d_fun < s * 0.028 {
                        let fa = (1.0 - d_fun / (s * 0.028)) * 0.65 * aa;
                        c = blend_rgba(c, fcol, fa);
                    }
                }

                let t = (py - s * 0.16) / (s * 0.68);
                if (-0.08..=1.08).contains(&t) {
                    let tc = t.clamp(0.0, 1.0);
                    let amp = s * (0.21 - 0.08 * tc);
                    let wave = (tc * std::f32::consts::PI * 3.0).sin();
                    let hx1 = cx + amp * wave;
                    let hx2 = cx - amp * wave;
                    let thick = (s * 0.068).max(1.3);
                    let d1 = (px - hx1).abs();
                    let d2 = (px - hx2).abs();
                    if d2 < thick + 1.0 {
                        let a2 = ((thick + 0.6 - d2) / 1.2).clamp(0.0, 1.0) * 0.55 * aa;
                        c = blend_rgba(c, [45, 160, 210], a2);
                    }
                    if d1 < thick + 1.2 {
                        let a1 = ((thick + 0.7 - d1) / 1.2).clamp(0.0, 1.0) * aa;
                        let rc = (56.0 + (52.0 - 56.0) * tc) as u8;
                        let gc = (189.0 + (211.0 - 189.0) * tc) as u8;
                        let bc = (248.0 + (153.0 - 248.0) * tc) as u8;
                        c = blend_rgba(c, [rc, gc, bc], a1);
                    }
                }

                let bx = cx;
                let by = s * 0.78;
                let bd = (px - bx).hypot(py - by);
                let brad = (s * 0.075).max(1.5);
                if bd < brad + 1.0 {
                    let ba = ((brad + 0.5 - bd) / 1.0).clamp(0.0, 1.0) * aa;
                    c = blend_rgba(c, [245, 190, 65], ba);
                }
                c
            };

            let off = ((y * size + x) * 4) as usize;
            rgba[off..off + 4].copy_from_slice(&col);
        }
    }

    IconData {
        rgba,
        width: size,
        height: size,
    }
}
