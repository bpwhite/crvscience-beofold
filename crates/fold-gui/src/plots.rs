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
    Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, Vec2,
};
use std::collections::VecDeque;

/// Draws an interactive vector scientific time-series trace with gridlines, min/max bounds,
/// and cursor readout inside `egui` without external plotting crate dependencies.
pub fn draw_scientific_trace(
    ui: &mut Ui,
    title: &str,
    unit: &str,
    series: &VecDeque<f64>,
    overlay_min_series: Option<&VecDeque<f64>>,
    line_color: Color32,
    height: f32,
) {
    let width = ui.available_width().max(220.0);
    let (response, painter) = ui.allocate_painter(Vec2::new(width, height), Sense::hover());
    let rect = response.rect;

    painter.rect_filled(rect, CornerRadius::same(6), Color32::from_rgb(16, 22, 32));
    painter.rect_stroke(
        rect,
        CornerRadius::same(6),
        Stroke::new(1.0_f32, Color32::from_rgb(38, 50, 68)),
        StrokeKind::Inside,
    );

    if series.is_empty() {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            format!("{} — awaiting trajectory samples", title),
            FontId::monospace(11.5),
            Color32::from_rgb(130, 145, 165),
        );
        return;
    }

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &v in series {
        if v.is_finite() {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if let Some(ov) = overlay_min_series {
        for &v in ov {
            if v.is_finite() {
                y_min = y_min.min(v);
                y_max = y_max.max(v);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let span = (y_max - y_min).abs().max(0.25);
    let y_lo = y_min - 0.08 * span;
    let y_hi = y_max + 0.12 * span;

    let pad_left = 52.0;
    let pad_right = 12.0;
    let pad_top = 26.0;
    let pad_bottom = 30.0;
    let plot_rect = Rect::from_min_max(
        Pos2::new(rect.left() + pad_left, rect.top() + pad_top),
        Pos2::new(rect.right() - pad_right, rect.bottom() - pad_bottom),
    );

    // Subtle horizontal reference gridlines + Y-axis tick labels
    for g in 0..=2 {
        let frac = (g as f32) / 2.0;
        let y = plot_rect.top() + frac * plot_rect.height();
        let val = y_hi - (frac as f64) * (y_hi - y_lo);
        painter.line_segment(
            [Pos2::new(plot_rect.left(), y), Pos2::new(plot_rect.right(), y)],
            Stroke::new(1.0_f32, Color32::from_rgb(26, 35, 48)),
        );
        painter.line_segment(
            [
                Pos2::new(plot_rect.left() - 3.0, y),
                Pos2::new(plot_rect.left(), y),
            ],
            Stroke::new(1.0_f32, Color32::from_rgb(70, 88, 114)),
        );
        let tick_lbl = format_axis_tick(val, span);
        painter.text(
            Pos2::new(plot_rect.left() - 5.0, y),
            Align2::RIGHT_CENTER,
            tick_lbl,
            FontId::monospace(9.5),
            Color32::from_rgb(148, 163, 184),
        );
    }

    // Subtle vertical gridlines + X-axis sample tick labels
    let n = series.len();
    let x_ticks: &[(f32, usize)] = if n <= 1 {
        &[(0.0, 1)]
    } else if n == 2 {
        &[(0.0, 1), (1.0, 2)]
    } else {
        &[(0.0, 1), (0.5, (n + 1) / 2), (1.0, n)]
    };
    for &(xf, sample_num) in x_ticks {
        let x = plot_rect.left() + xf * plot_rect.width();
        painter.line_segment(
            [Pos2::new(x, plot_rect.top()), Pos2::new(x, plot_rect.bottom())],
            Stroke::new(1.0_f32, Color32::from_rgb(24, 32, 44)),
        );
        painter.line_segment(
            [
                Pos2::new(x, plot_rect.bottom()),
                Pos2::new(x, plot_rect.bottom() + 3.0),
            ],
            Stroke::new(1.0_f32, Color32::from_rgb(70, 88, 114)),
        );
        let align = if xf <= 0.05 {
            Align2::LEFT_TOP
        } else if xf >= 0.95 {
            Align2::RIGHT_TOP
        } else {
            Align2::CENTER_TOP
        };
        painter.text(
            Pos2::new(x, plot_rect.bottom() + 4.0),
            align,
            format!("#{}", sample_num),
            FontId::monospace(9.0),
            Color32::from_rgb(130, 145, 165),
        );
    }

    // Crisp plot border
    painter.rect_stroke(
        plot_rect,
        CornerRadius::same(0),
        Stroke::new(1.0_f32, Color32::from_rgb(44, 58, 78)),
        StrokeKind::Inside,
    );

    let map_pt = |idx: usize, count: usize, val: f64| -> Pos2 {
        let x_frac = if count <= 1 {
            0.0
        } else {
            (idx as f32) / ((count - 1) as f32)
        };
        let y_frac = ((val - y_lo) / (y_hi - y_lo)).clamp(0.0, 1.0) as f32;
        Pos2::new(
            plot_rect.left() + x_frac * plot_rect.width(),
            plot_rect.bottom() - y_frac * plot_rect.height(),
        )
    };

    let min_overlay_color = Color32::from_rgba_unmultiplied(52, 211, 153, 210);
    if let Some(ov) = overlay_min_series {
        if ov.len() >= 2 {
            let n_ov = ov.len();
            for i in 0..(n_ov - 1) {
                let p0 = map_pt(i, n_ov, ov[i]);
                let p1 = map_pt(i + 1, n_ov, ov[i + 1]);
                painter.line_segment([p0, p1], Stroke::new(1.4_f32, min_overlay_color));
            }
        }
    }

    if n >= 2 {
        for i in 0..(n - 1) {
            let p0 = map_pt(i, n, series[i]);
            let p1 = map_pt(i + 1, n, series[i + 1]);
            painter.line_segment([p0, p1], Stroke::new(1.9_f32, line_color));
        }
    } else if n == 1 {
        let p0 = map_pt(0, 1, series[0]);
        painter.circle_filled(p0, 3.0, line_color);
    }

    let latest = *series.back().unwrap_or(&0.0);
    let compact_title = title.split(" (").next().unwrap_or(title);
    painter.text(
        Pos2::new(rect.left() + 10.0, rect.top() + 5.0),
        Align2::LEFT_TOP,
        format!("{}: {:.2} {}", compact_title, latest, unit),
        FontId::monospace(11.0),
        Color32::from_rgb(226, 232, 240),
    );

    // Compact series / min legend inside top-right of plot_rect
    let has_min = overlay_min_series.is_some();
    let legend_w = if has_min { 108.0 } else { 62.0 };
    let legend_rect = Rect::from_min_size(
        Pos2::new(plot_rect.right() - legend_w - 4.0, plot_rect.top() + 4.0),
        Vec2::new(legend_w, 16.0),
    );
    painter.rect_filled(
        legend_rect,
        CornerRadius::same(3),
        Color32::from_rgba_unmultiplied(12, 17, 25, 205),
    );
    let ly = legend_rect.center().y;
    let sx0 = legend_rect.left() + 6.0;
    painter.line_segment(
        [Pos2::new(sx0, ly), Pos2::new(sx0 + 12.0, ly)],
        Stroke::new(2.0_f32, line_color),
    );
    painter.text(
        Pos2::new(sx0 + 15.0, ly),
        Align2::LEFT_CENTER,
        "Series",
        FontId::monospace(9.0),
        Color32::from_rgb(203, 213, 225),
    );
    if has_min {
        let mx0 = sx0 + 56.0;
        painter.line_segment(
            [Pos2::new(mx0, ly), Pos2::new(mx0 + 12.0, ly)],
            Stroke::new(1.8_f32, min_overlay_color),
        );
        painter.text(
            Pos2::new(mx0 + 15.0, ly),
            Align2::LEFT_CENTER,
            "Min",
            FontId::monospace(9.0),
            Color32::from_rgb(52, 211, 153),
        );
    }

    painter.text(
        Pos2::new(rect.right() - 10.0, rect.bottom() - 4.0),
        Align2::RIGHT_BOTTOM,
        format!("min {:.2} | max {:.2}", y_min, y_max),
        FontId::monospace(9.5),
        Color32::from_rgb(148, 163, 184),
    );

    if let Some(hp) = response.hover_pos() {
        if plot_rect.contains(hp) && n >= 1 {
            let rel_x = ((hp.x - plot_rect.left()) / plot_rect.width().max(1.0)).clamp(0.0, 1.0);
            let idx = ((rel_x * ((n - 1) as f32)).round() as usize).min(n - 1);
            let val = series[idx];
            let pt = map_pt(idx, n, val);
            painter.line_segment(
                [
                    Pos2::new(pt.x, plot_rect.top()),
                    Pos2::new(pt.x, plot_rect.bottom()),
                ],
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 90)),
            );
            painter.circle_filled(pt, 3.6, Color32::WHITE);
            painter.text(
                Pos2::new(rect.left() + 10.0, rect.bottom() - 4.0),
                Align2::LEFT_BOTTOM,
                format!("Sample #{}: {:.3} {}", idx + 1, val, unit),
                FontId::monospace(10.0),
                line_color,
            );
        }
    }
}

pub(crate) fn format_axis_tick(val: f64, span: f64) -> String {
    if span >= 100.0 || val.abs() >= 1000.0 {
        format!("{:.0}", val)
    } else if span >= 10.0 {
        format!("{:.1}", val)
    } else {
        format!("{:.2}", val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plots_render_ticks_and_legend_headlessly_without_panic() {
        assert_eq!(format_axis_tick(-142.35, 150.0), "-142");
        assert_eq!(format_axis_tick(12.34, 25.0), "12.3");
        assert_eq!(format_axis_tick(0.754, 1.0), "0.75");

        let ctx = egui::Context::default();
        let series: VecDeque<f64> = VecDeque::from([-10.0, -14.5, -18.2, -20.1]);
        let min_series: VecDeque<f64> = VecDeque::from([-10.0, -14.5, -18.2, -20.1]);
        let empty: VecDeque<f64> = VecDeque::new();
        let single: VecDeque<f64> = VecDeque::from([12.5]);

        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(400.0, 600.0),
        ));
        let out = ctx.run(raw, |c| {
            egui::CentralPanel::default().show(c, |ui| {
                draw_scientific_trace(
                    ui,
                    "Potential Energy V_total (green = min)",
                    "kcal/mol",
                    &series,
                    Some(&min_series),
                    Color32::from_rgb(56, 189, 248),
                    140.0,
                );
                draw_scientific_trace(
                    ui,
                    "Rg",
                    "A",
                    &single,
                    None,
                    Color32::from_rgb(245, 190, 65),
                    120.0,
                );
                draw_scientific_trace(
                    ui,
                    "Empty",
                    "Q",
                    &empty,
                    None,
                    Color32::from_rgb(52, 211, 153),
                    120.0,
                );
            });
        });
        assert!(!out.shapes.is_empty());
    }
}
