use crate::clustering::{ConformationalFrame, MicrostateClustering, MsmError};
use crate::transition::ReversibleMsmModel;
use fold_physics::{ProteinTopology, K_BOLTZMANN_KCAL};
use fold_proto::ProjectManifest;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Summary of exported MSM files written by `export_msm_bundle`.
#[derive(Debug, Clone)]
pub struct ExportedMsmArtifacts {
    pub json_report_path: PathBuf,
    pub microstate_csv_path: PathBuf,
    pub pmf_grid_csv_path: PathBuf,
    pub svg_landscape_path: PathBuf,
    pub macrostate_pdb_paths: Vec<PathBuf>,
}

/// Computes a smooth 2D Boltzmann-reweighted Potential of Mean Force (`PMF(R_g, RMSD_best)`)
/// and exports `msm_report.json`, `free_energy_microstates.csv`, `free_energy_surface_2d.csv`,
/// `free_energy_landscape.svg`, and crystallographic `.pdb` structures for each PCCA+ macrostate.
pub fn export_msm_bundle(
    out_dir: &Path,
    manifest: &ProjectManifest,
    topology: &ProteinTopology,
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    msm: &ReversibleMsmModel,
) -> Result<ExportedMsmArtifacts, MsmError> {
    fs::create_dir_all(out_dir)?;

    let micro_csv = out_dir.join("free_energy_microstates.csv");
    write_microstate_csv(&micro_csv, frames, clustering, msm)?;

    let pmf_csv = out_dir.join("free_energy_surface_2d.csv");
    let grid = compute_2d_pmf_grid(frames, clustering, msm, 24, 24);
    write_pmf_grid_csv(&pmf_csv, &grid)?;

    let svg_path = out_dir.join("free_energy_landscape.svg");
    write_academic_landscape_svg(&svg_path, manifest, frames, clustering, msm, &grid)?;

    let mut pdb_paths = Vec::with_capacity(msm.macrostates.len());
    for mac in &msm.macrostates {
        let fname = format!("macrostate_{}_{}.pdb", mac.macro_id, mac.biophysical_label);
        let fpath = out_dir.join(fname);
        let rep = &frames[mac.representative_frame_idx];
        topology.write_pdb_snapshot(
            &rep.positions_angstrom,
            &fpath,
            &format!(
                "PCCA+ Macrostate {} ({}, pi={:.3}, dG={:.2} kcal/mol, p_fold={:.2})",
                mac.macro_id,
                mac.biophysical_label,
                mac.stationary_population,
                mac.free_energy_kcal,
                mac.mean_committor
            ),
            rep.potential_energy_kcal,
        )?;
        pdb_paths.push(fpath);
    }

    let json_path = out_dir.join("msm_report.json");
    write_json_report(&json_path, manifest, frames, clustering, msm, &pdb_paths)?;

    Ok(ExportedMsmArtifacts {
        json_report_path: json_path,
        microstate_csv_path: micro_csv,
        pmf_grid_csv_path: pmf_csv,
        svg_landscape_path: svg_path,
        macrostate_pdb_paths: pdb_paths,
    })
}

#[derive(Debug, Clone)]
pub struct PmfGrid2D {
    pub nx: usize,
    pub ny: usize,
    pub rmsd_min: f64,
    pub rmsd_max: f64,
    pub rg_min: f64,
    pub rg_max: f64,
    /// `values[ix][iy]` in `kcal/mol` (`0.0` at minimum).
    pub free_energy_kcal: Vec<Vec<f64>>,
    pub max_capped_kcal: f64,
}

/// Evaluates an equilibrium MSM-reweighted 2D Gaussian Kernel Density Potential of Mean Force
/// `W(RMSD, R_g) = -k_B * T * ln P_eq(RMSD, R_g)` where each frame inside microstate `i` carries
/// equilibrium statistical weight `pi_i / count(i)`.
pub fn compute_2d_pmf_grid(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    msm: &ReversibleMsmModel,
    nx: usize,
    ny: usize,
) -> PmfGrid2D {
    let mut rmsd_min = f64::INFINITY;
    let mut rmsd_max = f64::NEG_INFINITY;
    let mut rg_min = f64::INFINITY;
    let mut rg_max = f64::NEG_INFINITY;

    for f in frames {
        rmsd_min = rmsd_min.min(f.rmsd_to_lowest_energy_ang);
        rmsd_max = rmsd_max.max(f.rmsd_to_lowest_energy_ang);
        rg_min = rg_min.min(f.rg_angstrom);
        rg_max = rg_max.max(f.rg_angstrom);
    }
    let rmsd_pad = (rmsd_max - rmsd_min).abs().max(1.0) * 0.12;
    let rg_pad = (rg_max - rg_min).abs().max(1.0) * 0.12;
    rmsd_min = (rmsd_min - rmsd_pad).max(0.0);
    rmsd_max += rmsd_pad;
    rg_min = (rg_min - rg_pad).max(1.0);
    rg_max += rg_pad;

    let bw_rmsd = ((rmsd_max - rmsd_min) / 7.5).max(0.25);
    let bw_rg = ((rg_max - rg_min) / 7.5).max(0.25);
    let kbt = K_BOLTZMANN_KCAL * msm.reference_temp_kelvin.max(1.0);

    let frame_weights: Vec<f64> = frames
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let st = clustering.frame_assignments[idx];
            let cnt = clustering.microstate_counts[st].max(1) as f64;
            msm.stationary_pi[st] / cnt
        })
        .collect();

    let mut density = vec![vec![0.0f64; ny]; nx];
    let mut max_dens = 0.0f64;

    for ix in 0..nx {
        let x = rmsd_min + ((ix as f64 + 0.5) / (nx as f64)) * (rmsd_max - rmsd_min);
        for iy in 0..ny {
            let y = rg_min + ((iy as f64 + 0.5) / (ny as f64)) * (rg_max - rg_min);
            let mut sum_p = 0.0f64;
            for (f, &w) in frames.iter().zip(frame_weights.iter()) {
                let dx = (x - f.rmsd_to_lowest_energy_ang) / bw_rmsd;
                let dy = (y - f.rg_angstrom) / bw_rg;
                let r2 = dx * dx + dy * dy;
                if r2 < 16.0 {
                    sum_p += w * (-0.5 * r2).exp();
                }
            }
            density[ix][iy] = sum_p;
            if sum_p > max_dens {
                max_dens = sum_p;
            }
        }
    }

    let cap_kcal = 6.0f64;
    let mut fe = vec![vec![cap_kcal; ny]; nx];
    let denom = max_dens.max(1e-30);
    for ix in 0..nx {
        for iy in 0..ny {
            if density[ix][iy] > denom * 1.0e-8 {
                let val = -kbt * (density[ix][iy] / denom).ln();
                fe[ix][iy] = val.clamp(0.0, cap_kcal);
            }
        }
    }

    PmfGrid2D {
        nx,
        ny,
        rmsd_min,
        rmsd_max,
        rg_min,
        rg_max,
        free_energy_kcal: fe,
        max_capped_kcal: cap_kcal,
    }
}

fn write_microstate_csv(
    path: &Path,
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    msm: &ReversibleMsmModel,
) -> Result<(), MsmError> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "microstate_id,macrostate_id,frame_count,stationary_pi,free_energy_kcal,committor_p_fold,medoid_wu,medoid_potential_kcal,medoid_rg_ang,medoid_rmsd_best_ang,mean_cluster_radius_ang"
    )?;
    for i in 0..msm.num_microstates {
        let med = &frames[clustering.medoid_frame_indices[i]];
        writeln!(
            w,
            "{},{},{},{:.6},{:.4},{:.4},{},{:.3},{:.3},{:.3},{:.3}",
            i,
            msm.microstate_to_macrostate[i],
            clustering.microstate_counts[i],
            msm.stationary_pi[i],
            msm.microstate_free_energy_kcal[i],
            msm.folding_committor[i],
            med.wu_id,
            med.potential_energy_kcal,
            med.rg_angstrom,
            med.rmsd_to_lowest_energy_ang,
            clustering.mean_cluster_radius_ang[i]
        )?;
    }
    w.flush()?;
    Ok(())
}

fn write_pmf_grid_csv(path: &Path, grid: &PmfGrid2D) -> Result<(), MsmError> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "bin_x,bin_y,ca_rmsd_best_ang,rg_angstrom,pmf_kcal_mol")?;
    for ix in 0..grid.nx {
        let x = grid.rmsd_min
            + ((ix as f64 + 0.5) / (grid.nx as f64)) * (grid.rmsd_max - grid.rmsd_min);
        for iy in 0..grid.ny {
            let y = grid.rg_min
                + ((iy as f64 + 0.5) / (grid.ny as f64)) * (grid.rg_max - grid.rg_min);
            writeln!(
                w,
                "{},{},{:.4},{:.4},{:.4}",
                ix, iy, x, y, grid.free_energy_kcal[ix][iy]
            )?;
        }
    }
    w.flush()?;
    Ok(())
}

fn write_academic_landscape_svg(
    path: &Path,
    manifest: &ProjectManifest,
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    msm: &ReversibleMsmModel,
    grid: &PmfGrid2D,
) -> Result<(), MsmError> {
    let mut w = BufWriter::new(File::create(path)?);
    let width = 920.0f64;
    let height = 560.0f64;

    let plot_x0 = 82.0f64;
    let plot_y0 = 72.0f64;
    let plot_w = 500.0f64;
    let plot_h = 410.0f64;

    writeln!(
        w,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"##
    )?;
    writeln!(
        w,
        r##"<rect x="0" y="0" width="{width}" height="{height}" fill="#FAFBFD"/>"##
    )?;

    writeln!(
        w,
        r##"<text x="82" y="34" font-family="system-ui, -apple-system, sans-serif" font-size="17" font-weight="700" fill="#1E293B">BeoFold Markov State Model — {name} ({res} aa)</text>"##,
        name = manifest.name,
        res = manifest.residue_count
    )?;
    writeln!(
        w,
        r##"<text x="82" y="54" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#475569">K = {k} microstates | M = {m} PCCA+ macrostates | lag = {lag:.2} ps | transitions = {tr} | detailed-balance residual = {db:.1e}</text>"##,
        k = msm.num_microstates,
        m = msm.macrostates.len(),
        lag = msm.lag_time_ps,
        tr = msm.total_transitions,
        db = msm.detailed_balance_max_error
    )?;

    let cell_w = plot_w / (grid.nx as f64);
    let cell_h = plot_h / (grid.ny as f64);
    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            let fe = grid.free_energy_kcal[ix][iy];
            let norm = (fe / grid.max_capped_kcal).clamp(0.0, 1.0);
            let col = academic_pmf_hex(norm);
            let sx = plot_x0 + (ix as f64) * cell_w;
            let sy = plot_y0 + plot_h - ((iy + 1) as f64) * cell_h;
            writeln!(
                w,
                r##"<rect x="{sx:.2}" y="{sy:.2}" width="{cw:.2}" height="{ch:.2}" fill="{col}"/>"##,
                cw = cell_w + 0.6,
                ch = cell_h + 0.6
            )?;
        }
    }

    writeln!(
        w,
        r##"<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#334155" stroke-width="1.2"/>"##
    )?;

    let map_xy = |rmsd: f64, rg: f64| -> (f64, f64) {
        let fx =
            ((rmsd - grid.rmsd_min) / (grid.rmsd_max - grid.rmsd_min).max(1e-6)).clamp(0.0, 1.0);
        let fy = ((rg - grid.rg_min) / (grid.rg_max - grid.rg_min).max(1e-6)).clamp(0.0, 1.0);
        (plot_x0 + fx * plot_w, plot_y0 + plot_h - fy * plot_h)
    };

    for i in 0..msm.num_microstates {
        for j in (i + 1)..msm.num_microstates {
            let flux = msm.stationary_pi[i] * msm.transition_matrix[i][j];
            if flux < 0.004 {
                continue;
            }
            let mi = &frames[clustering.medoid_frame_indices[i]];
            let mj = &frames[clustering.medoid_frame_indices[j]];
            let (x1, y1) = map_xy(mi.rmsd_to_lowest_energy_ang, mi.rg_angstrom);
            let (x2, y2) = map_xy(mj.rmsd_to_lowest_energy_ang, mj.rg_angstrom);
            let stroke_w = (0.8 + 18.0 * flux).clamp(0.8, 4.2);
            writeln!(
                w,
                r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="#1E293B" stroke-opacity="0.38" stroke-width="{stroke_w:.2}"/>"##
            )?;
        }
    }

    for i in 0..msm.num_microstates {
        let med = &frames[clustering.medoid_frame_indices[i]];
        let (cx, cy) = map_xy(med.rmsd_to_lowest_energy_ang, med.rg_angstrom);
        let rad = (5.0 + 18.0 * msm.stationary_pi[i].sqrt()).clamp(5.0, 16.0);
        let fill = committor_hex(msm.folding_committor[i]);
        writeln!(
            w,
            r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="{rad:.1}" fill="{fill}" stroke="#0F172A" stroke-width="1.3"/>"##
        )?;
        writeln!(
            w,
            r##"<text x="{cx:.1}" y="{ty:.1}" text-anchor="middle" font-family="monospace" font-size="9.5" font-weight="700" fill="#0F172A">S{i}</text>"##,
            ty = cy + 3.2
        )?;
    }

    for tick in 0..=4 {
        let f = (tick as f64) / 4.0;
        let xv = grid.rmsd_min + f * (grid.rmsd_max - grid.rmsd_min);
        let xp = plot_x0 + f * plot_w;
        let yp = plot_y0 + plot_h;
        writeln!(
            w,
            r##"<line x1="{xp:.1}" y1="{yp:.1}" x2="{xp:.1}" y2="{yt:.1}" stroke="#334155"/>"##,
            yt = yp + 5.0
        )?;
        writeln!(
            w,
            r##"<text x="{xp:.1}" y="{yl:.1}" text-anchor="middle" font-family="monospace" font-size="10.5" fill="#334155">{xv:.2}</text>"##,
            yl = yp + 18.0
        )?;

        let yv = grid.rg_min + f * (grid.rg_max - grid.rg_min);
        let y_scr = plot_y0 + plot_h - f * plot_h;
        writeln!(
            w,
            r##"<line x1="{xl:.1}" y1="{y_scr:.1}" x2="{plot_x0:.1}" y2="{y_scr:.1}" stroke="#334155"/>"##,
            xl = plot_x0 - 5.0
        )?;
        writeln!(
            w,
            r##"<text x="{xt:.1}" y="{yt2:.1}" text-anchor="end" font-family="monospace" font-size="10.5" fill="#334155">{yv:.2}</text>"##,
            xt = plot_x0 - 9.0,
            yt2 = y_scr + 3.5
        )?;
    }

    writeln!(
        w,
        r##"<text x="{mx:.1}" y="{my:.1}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12.5" font-weight="600" fill="#1E293B">C-Alpha RMSD to Lowest-Energy State (Angstroms)</text>"##,
        mx = plot_x0 + plot_w * 0.5,
        my = plot_y0 + plot_h + 40.0
    )?;
    writeln!(
        w,
        r##"<text x="24" y="{ry:.1}" transform="rotate(-90 24 {ry:.1})" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12.5" font-weight="600" fill="#1E293B">Radius of Gyration Rg (Angstroms)</text>"##,
        ry = plot_y0 + plot_h * 0.5
    )?;

    let rx0 = 612.0f64;
    writeln!(
        w,
        r##"<rect x="{rx0}" y="72" width="282" height="190" rx="6" fill="#FFFFFF" stroke="#CBD5E1"/>"##
    )?;
    writeln!(
        w,
        r##"<text x="{tx}" y="95" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#1E293B">Implied Relaxation Timescales (ps)</text>"##,
        tx = rx0 + 14.0
    )?;

    let max_its = msm
        .implied_timescales_ps
        .iter()
        .take(5)
        .cloned()
        .fold(1.0f64, f64::max);
    for (idx, &its) in msm.implied_timescales_ps.iter().take(5).enumerate() {
        let by = 114.0 + (idx as f64) * 28.0;
        let bar_w = ((its / max_its).clamp(0.02, 1.0)) * 165.0;
        writeln!(
            w,
            r##"<text x="{lx}" y="{ty}" font-family="monospace" font-size="11" fill="#334155">t_{m}</text>"##,
            lx = rx0 + 14.0,
            ty = by + 11.0,
            m = idx + 2
        )?;
        writeln!(
            w,
            r##"<rect x="{bx}" y="{by}" width="{bar_w:.1}" height="14" rx="2" fill="#2563EB"/>"##,
            bx = rx0 + 44.0
        )?;
        writeln!(
            w,
            r##"<text x="{vx}" y="{ty}" font-family="monospace" font-size="10.5" fill="#1E293B">{its:.2} ps</text>"##,
            vx = rx0 + 216.0,
            ty = by + 11.0
        )?;
    }

    writeln!(
        w,
        r##"<rect x="{rx0}" y="278" width="282" height="204" rx="6" fill="#FFFFFF" stroke="#CBD5E1"/>"##
    )?;
    writeln!(
        w,
        r##"<text x="{tx}" y="301" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#1E293B">PCCA+ Metastable Macrostates</text>"##,
        tx = rx0 + 14.0
    )?;

    for (idx, mac) in msm.macrostates.iter().take(5).enumerate() {
        let my = 326.0 + (idx as f64) * 32.0;
        writeln!(
            w,
            r##"<text x="{lx}" y="{my}" font-family="monospace" font-size="11" font-weight="700" fill="#0F172A">M{id} ({lbl})</text>"##,
            lx = rx0 + 14.0,
            id = mac.macro_id,
            lbl = mac.biophysical_label
        )?;
        writeln!(
            w,
            r##"<text x="{lx}" y="{my2}" font-family="monospace" font-size="10" fill="#475569">pi={pi:.3} | dG={dg:.2} kcal/mol | p_fold={pf:.2}</text>"##,
            lx = rx0 + 14.0,
            my2 = my + 13.0,
            pi = mac.stationary_population,
            dg = mac.free_energy_kcal,
            pf = mac.mean_committor
        )?;
    }

    writeln!(w, "</svg>")?;
    w.flush()?;
    Ok(())
}

fn academic_pmf_hex(norm: f64) -> String {
    let t = norm.clamp(0.0, 1.0);
    let r = (37.0 + t * (248.0 - 37.0)).round() as u8;
    let g = (99.0 + t * (250.0 - 99.0)).round() as u8;
    let b = (195.0 + t * (252.0 - 195.0)).round() as u8;
    format!("#{:02X}{:02X}{:02X}", r, g, b)
}

fn committor_hex(p_fold: f64) -> String {
    let p = p_fold.clamp(0.0, 1.0);
    let (r, g, b) = if p < 0.5 {
        let u = p * 2.0;
        (
            217.0 + u * (148.0 - 217.0),
            119.0 + u * (163.0 - 119.0),
            6.0 + u * (184.0 - 6.0),
        )
    } else {
        let u = (p - 0.5) * 2.0;
        (
            148.0 + u * (5.0 - 148.0),
            163.0 + u * (150.0 - 163.0),
            184.0 + u * (105.0 - 184.0),
        )
    };
    format!("#{:02X}{:02X}{:02X}", r as u8, g as u8, b as u8)
}

fn write_json_report(
    path: &Path,
    manifest: &ProjectManifest,
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    msm: &ReversibleMsmModel,
    pdb_paths: &[PathBuf],
) -> Result<(), MsmError> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "{{")?;
    writeln!(w, "  \"project_id\": {},", manifest.project_id)?;
    writeln!(w, "  \"project_name\": \"{}\",", manifest.name)?;
    writeln!(w, "  \"sequence\": \"{}\",", manifest.sequence)?;
    writeln!(w, "  \"total_frames\": {},", frames.len())?;
    writeln!(w, "  \"num_microstates\": {},", msm.num_microstates)?;
    writeln!(w, "  \"lag_generations\": {},", msm.lag_generations)?;
    writeln!(w, "  \"lag_time_ps\": {:.4},", msm.lag_time_ps)?;
    writeln!(
        w,
        "  \"reference_temp_kelvin\": {:.2},",
        msm.reference_temp_kelvin
    )?;
    writeln!(w, "  \"total_transitions\": {},", msm.total_transitions)?;
    writeln!(
        w,
        "  \"detailed_balance_max_error\": {:.4e},",
        msm.detailed_balance_max_error
    )?;
    writeln!(w, "  \"eigenvalues\": {:?},", msm.eigenvalues)?;
    writeln!(
        w,
        "  \"implied_timescales_ps\": {:?},",
        msm.implied_timescales_ps
    )?;
    writeln!(w, "  \"stationary_pi\": {:?},", msm.stationary_pi)?;
    writeln!(
        w,
        "  \"microstate_free_energy_kcal\": {:?},",
        msm.microstate_free_energy_kcal
    )?;
    writeln!(w, "  \"folding_committor\": {:?},", msm.folding_committor)?;

    writeln!(w, "  \"macrostates\": [")?;
    for (idx, mac) in msm.macrostates.iter().enumerate() {
        let comma = if idx + 1 < msm.macrostates.len() {
            ","
        } else {
            ""
        };
        let rep = &frames[mac.representative_frame_idx];
        let pdb_name = pdb_paths
            .get(idx)
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        writeln!(
            w,
            "    {{\"macro_id\": {}, \"label\": \"{}\", \"stationary_population\": {:.6}, \"free_energy_kcal\": {:.4}, \"mean_committor\": {:.4}, \"mean_rg_angstrom\": {:.3}, \"mean_rmsd_best_ang\": {:.3}, \"representative_wu\": \"{}\", \"representative_pdb\": \"{}\", \"member_microstates\": {:?}}}{}",
            mac.macro_id,
            mac.biophysical_label,
            mac.stationary_population,
            mac.free_energy_kcal,
            mac.mean_committor,
            mac.mean_rg_angstrom,
            mac.mean_rmsd_best_ang,
            rep.wu_id,
            pdb_name,
            mac.member_microstates,
            comma
        )?;
    }
    writeln!(w, "  ],")?;

    writeln!(w, "  \"transition_matrix\": [")?;
    for (i, row) in msm.transition_matrix.iter().enumerate() {
        let comma = if i + 1 < msm.transition_matrix.len() {
            ","
        } else {
            ""
        };
        let formatted: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(w, "    [{}]{}", formatted.join(", "), comma)?;
    }
    writeln!(w, "  ],")?;
    writeln!(
        w,
        "  \"microstate_counts\": {:?}",
        clustering.microstate_counts
    )?;
    writeln!(w, "}}")?;
    w.flush()?;
    Ok(())
}
