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

    let pad_left = 10.0;
    let pad_right = 10.0;
    let pad_top = 24.0;
    let pad_bottom = 18.0;
    let plot_rect = Rect::from_min_max(
        Pos2::new(rect.left() + pad_left, rect.top() + pad_top),
        Pos2::new(rect.right() - pad_right, rect.bottom() - pad_bottom),
    );

    // Subtle horizontal reference gridlines
    for g in 0..=2 {
        let frac = (g as f32) / 2.0;
        let y = plot_rect.top() + frac * plot_rect.height();
        painter.line_segment(
            [Pos2::new(plot_rect.left(), y), Pos2::new(plot_rect.right(), y)],
            Stroke::new(1.0_f32, Color32::from_rgb(26, 35, 48)),
        );
    }

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

    if let Some(ov) = overlay_min_series {
        if ov.len() >= 2 {
            let n = ov.len();
            for i in 0..(n - 1) {
                let p0 = map_pt(i, n, ov[i]);
                let p1 = map_pt(i + 1, n, ov[i + 1]);
                painter.line_segment(
                    [p0, p1],
                    Stroke::new(1.4_f32, Color32::from_rgba_unmultiplied(52, 211, 153, 185)),
                );
            }
        }
    }

    let n = series.len();
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
    painter.text(
        Pos2::new(rect.left() + 10.0, rect.top() + 5.0),
        Align2::LEFT_TOP,
        format!("{}: {:.2} {}", title, latest, unit),
        FontId::monospace(11.5),
        Color32::from_rgb(226, 232, 240),
    );
    painter.text(
        Pos2::new(rect.right() - 10.0, rect.top() + 5.0),
        Align2::RIGHT_TOP,
        format!("min {:.2} | max {:.2}", y_min, y_max),
        FontId::monospace(10.5),
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
                Pos2::new(rect.left() + 10.0, rect.bottom() - 15.0),
                Align2::LEFT_TOP,
                format!("Sample #{}: {:.3} {}", idx + 1, val, unit),
                FontId::monospace(10.5),
                line_color,
            );
        }
    }
}
