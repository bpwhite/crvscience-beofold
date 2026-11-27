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

use crate::clustering::{ConformationalFrame, MicrostateClustering, MsmError};
use crate::transition::{estimate_reversible_msm_with_max_temp, ReversibleMsmModel};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Implied relaxation timescales (`ITS`) evaluated at lag `k * base_lag`.
#[derive(Debug, Clone)]
pub struct ImpliedTimescalePoint {
    pub lag_multiplier_k: u32,
    pub lag_generations: u32,
    pub lag_time_ps: f64,
    pub active_microstates: usize,
    pub total_transitions: u64,
    pub detailed_balance_max_error: f64,
    pub eigenvalues: Vec<f64>,
    pub implied_timescales_ps: Vec<f64>,
}

/// Single entry of the Prinz et al. (2011) Chapman-Kolmogorov macrostate test:
/// `P_pred(A -> B; k*tau)` vs. `P_obs(A -> B; k*tau)`.
#[derive(Debug, Clone)]
pub struct ChapmanKolmogorovEntry {
    pub lag_multiplier_k: u32,
    pub lag_generations: u32,
    pub lag_time_ps: f64,
    pub from_macro_id: usize,
    pub to_macro_id: usize,
    pub from_label: String,
    pub to_label: String,
    pub predicted_prob: f64,
    pub observed_prob: f64,
    pub abs_error: f64,
}

/// Combined Implied-Timescale (`ITS`) and Chapman-Kolmogorov (`CK`) diagnostic digest.
#[derive(Debug, Clone)]
pub struct MsmDiagnosticDigest {
    pub base_lag_generations: u32,
    pub base_lag_time_ps: f64,
    pub num_microstates: usize,
    pub num_macrostates: usize,
    pub implied_timescales: Vec<ImpliedTimescalePoint>,
    pub chapman_kolmogorov: Vec<ChapmanKolmogorovEntry>,
    pub ck_max_abs_error: f64,
    pub ck_rmse: f64,
    pub markovian_pass: bool,
}

fn mat_mul_square(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn mat_pow_square(mat: &[Vec<f64>], power: u32) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut res = vec![vec![0.0_f64; n]; n];
    for (i, row) in res.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    if power == 0 {
        return res;
    }
    let mut base = mat.to_vec();
    let mut exp = power;
    while exp > 0 {
        if exp & 1 == 1 {
            res = mat_mul_square(&res, &base);
        }
        exp >>= 1;
        if exp > 0 {
            base = mat_mul_square(&base, &base);
        }
    }
    res
}

/// Computes MSM-predicted macrostate transition matrix `[M x M]` from `T_base^k` and `pi`.
fn predicted_macro_transition_matrix(
    base_msm: &ReversibleMsmModel,
    lag_k: u32,
) -> Vec<Vec<f64>> {
    let m = base_msm.macrostates.len();
    let k_micro = base_msm.num_microstates;
    let mut out = vec![vec![0.0_f64; m]; m];
    if m == 0 || k_micro == 0 {
        return out;
    }

    let t_pow = mat_pow_square(&base_msm.transition_matrix, lag_k.max(1));
    for a in 0..m {
        let mut denom = 0.0_f64;
        for i in 0..k_micro {
            if base_msm.microstate_to_macrostate[i] == a {
                denom += base_msm.stationary_pi[i];
            }
        }
        if denom <= 1.0e-18 {
            out[a][a] = 1.0;
            continue;
        }
        for b in 0..m {
            let mut num = 0.0_f64;
            for i in 0..k_micro {
                if base_msm.microstate_to_macrostate[i] != a {
                    continue;
                }
                let pi_i = base_msm.stationary_pi[i];
                for j in 0..k_micro {
                    if base_msm.microstate_to_macrostate[j] == b {
                        num += pi_i * t_pow[i][j];
                    }
                }
            }
            out[a][b] = (num / denom).clamp(0.0, 1.0);
        }
    }
    out
}

/// Counts direct empirical macrostate transitions at lag `lag_generations` along `(run, clone)` chains.
fn observed_macro_transition_matrix(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    base_msm: &ReversibleMsmModel,
    lag_generations: u32,
    max_temperature_kelvin: Option<f64>,
) -> Option<Vec<Vec<f64>>> {
    let m = base_msm.macrostates.len();
    if m == 0 {
        return None;
    }
    let mut chains: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (idx, f) in frames.iter().enumerate() {
        chains
            .entry((f.wu_id.run, f.wu_id.clone))
            .or_default()
            .push(idx);
    }
    for indices in chains.values_mut() {
        indices.sort_by_key(|&idx| frames[idx].chain_step);
    }

    let mut counts = vec![vec![0_u64; m]; m];
    let mut total = 0_u64;
    let lag_u64 = lag_generations.max(1) as u64;

    for indices in chains.values() {
        let mut by_step: HashMap<u64, usize> = HashMap::with_capacity(indices.len());
        for &idx in indices {
            if let Some(tmax) = max_temperature_kelvin {
                if frames[idx].temperature_kelvin > tmax {
                    continue;
                }
            }
            by_step.insert(frames[idx].chain_step as u64, idx);
        }
        for (&step, &i_idx) in &by_step {
            if let Some(&j_idx) = by_step.get(&(step + lag_u64)) {
                let st_i = clustering.frame_assignments[i_idx];
                let st_j = clustering.frame_assignments[j_idx];
                if let (Some(Some(ai)), Some(Some(aj))) = (
                    base_msm.orig_to_active.get(st_i),
                    base_msm.orig_to_active.get(st_j),
                ) {
                    let ma = base_msm.microstate_to_macrostate[*ai];
                    let mb = base_msm.microstate_to_macrostate[*aj];
                    counts[ma][mb] += 1;
                    total += 1;
                }
            }
        }
    }

    if total == 0 {
        return None;
    }

    let mut probs = vec![vec![0.0_f64; m]; m];
    for a in 0..m {
        let row_sum: u64 = counts[a].iter().sum();
        if row_sum > 0 {
            let inv = 1.0 / (row_sum as f64);
            for b in 0..m {
                probs[a][b] = (counts[a][b] as f64) * inv;
            }
        } else {
            probs[a][a] = 1.0;
        }
    }
    Some(probs)
}

/// Evaluates Implied Timescales (`ITS`) across lag multiples `1..=max_lag_multiplier` and
/// the Chapman-Kolmogorov (`CK`) macrostate consistency test against `base_msm`.
pub fn compute_msm_diagnostics(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    base_msm: &ReversibleMsmModel,
    max_lag_multiplier: u32,
    max_temperature_kelvin: Option<f64>,
) -> MsmDiagnosticDigest {
    let base_lag = base_msm.lag_generations.max(1);
    let max_k = max_lag_multiplier.clamp(1, 6);
    let m_req = base_msm.macrostates.len().max(2);
    let cold_cutoff = max_temperature_kelvin.or(Some(base_msm.reference_temp_kelvin + 25.0));

    let mut implied_timescales = Vec::new();
    let mut chapman_kolmogorov = Vec::new();
    let mut sq_err_sum = 0.0_f64;
    let mut ck_count = 0usize;
    let mut max_abs_err = 0.0_f64;

    for k in 1..=max_k {
        let lag_k = base_lag.saturating_mul(k);
        let lag_ps_k = base_msm.lag_time_ps * (k as f64);

        if k == 1 {
            implied_timescales.push(ImpliedTimescalePoint {
                lag_multiplier_k: 1,
                lag_generations: base_msm.lag_generations,
                lag_time_ps: base_msm.lag_time_ps,
                active_microstates: base_msm.num_microstates,
                total_transitions: base_msm.total_transitions,
                detailed_balance_max_error: base_msm.detailed_balance_max_error,
                eigenvalues: base_msm.eigenvalues.clone(),
                implied_timescales_ps: base_msm.implied_timescales_ps.clone(),
            });
        } else if let Ok(msm_k) = estimate_reversible_msm_with_max_temp(
            frames,
            clustering,
            lag_k,
            lag_ps_k,
            base_msm.reference_temp_kelvin,
            m_req,
            max_temperature_kelvin,
        ) {
            implied_timescales.push(ImpliedTimescalePoint {
                lag_multiplier_k: k,
                lag_generations: msm_k.lag_generations,
                lag_time_ps: msm_k.lag_time_ps,
                active_microstates: msm_k.num_microstates,
                total_transitions: msm_k.total_transitions,
                detailed_balance_max_error: msm_k.detailed_balance_max_error,
                eigenvalues: msm_k.eigenvalues,
                implied_timescales_ps: msm_k.implied_timescales_ps,
            });
        }

        let obs_opt = observed_macro_transition_matrix(
            frames,
            clustering,
            base_msm,
            lag_k,
            cold_cutoff,
        )
        .or_else(|| observed_macro_transition_matrix(frames, clustering, base_msm, lag_k, None));

        if let Some(obs) = obs_opt {
            let pred = predicted_macro_transition_matrix(base_msm, k);
            for (a, mac_a) in base_msm.macrostates.iter().enumerate() {
                for (b, mac_b) in base_msm.macrostates.iter().enumerate() {
                    let p_pred = pred[a][b];
                    let p_obs = obs[a][b];
                    let err = (p_pred - p_obs).abs();
                    if k >= 2 {
                        max_abs_err = max_abs_err.max(err);
                        sq_err_sum += err * err;
                        ck_count += 1;
                    }
                    chapman_kolmogorov.push(ChapmanKolmogorovEntry {
                        lag_multiplier_k: k,
                        lag_generations: lag_k,
                        lag_time_ps: lag_ps_k,
                        from_macro_id: mac_a.macro_id,
                        to_macro_id: mac_b.macro_id,
                        from_label: mac_a.biophysical_label.clone(),
                        to_label: mac_b.biophysical_label.clone(),
                        predicted_prob: p_pred,
                        observed_prob: p_obs,
                        abs_error: err,
                    });
                }
            }
        }
    }

    let ck_rmse = if ck_count > 0 {
        (sq_err_sum / (ck_count as f64)).sqrt()
    } else {
        0.0
    };
    let markovian_pass = if ck_count > 0 {
        max_abs_err <= 0.25 && base_msm.detailed_balance_max_error < 1.0e-6
    } else {
        base_msm.detailed_balance_max_error < 1.0e-6
    };

    MsmDiagnosticDigest {
        base_lag_generations: base_msm.lag_generations,
        base_lag_time_ps: base_msm.lag_time_ps,
        num_microstates: base_msm.num_microstates,
        num_macrostates: base_msm.macrostates.len(),
        implied_timescales,
        chapman_kolmogorov,
        ck_max_abs_error: max_abs_err,
        ck_rmse,
        markovian_pass,
    }
}

/// Formats [`MsmDiagnosticDigest`] as deterministic pretty-printed JSON.
pub fn format_msm_diagnostics_json(digest: &MsmDiagnosticDigest) -> String {
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "{{");
    let _ = writeln!(out, "  \"schema_version\": \"beofold-msm-diagnostics/v1\",");
    let _ = writeln!(out, "  \"base_lag_generations\": {},", digest.base_lag_generations);
    let _ = writeln!(out, "  \"base_lag_time_ps\": {:.6},", digest.base_lag_time_ps);
    let _ = writeln!(out, "  \"num_microstates\": {},", digest.num_microstates);
    let _ = writeln!(out, "  \"num_macrostates\": {},", digest.num_macrostates);
    let _ = writeln!(out, "  \"ck_max_abs_error\": {:.6},", digest.ck_max_abs_error);
    let _ = writeln!(out, "  \"ck_rmse\": {:.6},", digest.ck_rmse);
    let _ = writeln!(out, "  \"markovian_pass\": {},", digest.markovian_pass);

    let _ = writeln!(out, "  \"implied_timescales\": [");
    let n_its = digest.implied_timescales.len();
    for (idx, pt) in digest.implied_timescales.iter().enumerate() {
        let comma = if idx + 1 < n_its { "," } else { "" };
        let evals: Vec<String> = pt.eigenvalues.iter().map(|v| format!("{v:.6}")).collect();
        let its: Vec<String> = pt
            .implied_timescales_ps
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect();
        let _ = writeln!(
            out,
            "    {{\"lag_multiplier_k\": {}, \"lag_generations\": {}, \"lag_time_ps\": {:.6}, \"active_microstates\": {}, \"total_transitions\": {}, \"detailed_balance_max_error\": {:.6e}, \"eigenvalues\": [{}], \"implied_timescales_ps\": [{}]}}{}",
            pt.lag_multiplier_k,
            pt.lag_generations,
            pt.lag_time_ps,
            pt.active_microstates,
            pt.total_transitions,
            pt.detailed_balance_max_error,
            evals.join(", "),
            its.join(", "),
            comma
        );
    }
    let _ = writeln!(out, "  ],");

    let _ = writeln!(out, "  \"chapman_kolmogorov\": [");
    let n_ck = digest.chapman_kolmogorov.len();
    for (idx, row) in digest.chapman_kolmogorov.iter().enumerate() {
        let comma = if idx + 1 < n_ck { "," } else { "" };
        let _ = writeln!(
            out,
            "    {{\"lag_multiplier_k\": {}, \"lag_generations\": {}, \"lag_time_ps\": {:.6}, \"from_macro_id\": {}, \"to_macro_id\": {}, \"from_label\": \"{}\", \"to_label\": \"{}\", \"predicted_prob\": {:.6}, \"observed_prob\": {:.6}, \"abs_error\": {:.6}}}{}",
            row.lag_multiplier_k,
            row.lag_generations,
            row.lag_time_ps,
            row.from_macro_id,
            row.to_macro_id,
            row.from_label,
            row.to_label,
            row.predicted_prob,
            row.observed_prob,
            row.abs_error,
            comma
        );
    }
    let _ = writeln!(out, "  ]");
    let _ = writeln!(out, "}}");
    out
}

/// Formats [`MsmDiagnosticDigest`] as a tidy CSV ledger covering both `ITS` and `CK` records.
pub fn format_msm_diagnostics_csv(digest: &MsmDiagnosticDigest) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str(
        "section,lag_multiplier_k,lag_generations,lag_time_ps,mode_or_from,to_macro,predicted_or_its_ps,observed_or_lambda,abs_error_or_db_err,active_microstates,total_transitions\n",
    );
    for pt in &digest.implied_timescales {
        for (mode_idx, &its_ps) in pt.implied_timescales_ps.iter().enumerate() {
            let lambda = pt.eigenvalues.get(mode_idx + 1).copied().unwrap_or(0.0);
            let _ = writeln!(
                out,
                "ITS,{},{},{:.6},mode_{},-,{:.6},{:.6},{:.6e},{},{}",
                pt.lag_multiplier_k,
                pt.lag_generations,
                pt.lag_time_ps,
                mode_idx + 1,
                its_ps,
                lambda,
                pt.detailed_balance_max_error,
                pt.active_microstates,
                pt.total_transitions
            );
        }
    }
    for row in &digest.chapman_kolmogorov {
        let _ = writeln!(
            out,
            "CK,{},{},{:.6},M{}({}),M{}({}),{:.6},{:.6},{:.6},{},-",
            row.lag_multiplier_k,
            row.lag_generations,
            row.lag_time_ps,
            row.from_macro_id,
            row.from_label,
            row.to_macro_id,
            row.to_label,
            row.predicted_prob,
            row.observed_prob,
            row.abs_error,
            digest.num_microstates
        );
    }
    out
}

/// Computes and writes `msm_diagnostics.json` and `msm_diagnostics.csv` into `out_dir`.
pub fn export_msm_diagnostics(
    out_dir: &Path,
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    base_msm: &ReversibleMsmModel,
    max_lag_multiplier: u32,
    max_temperature_kelvin: Option<f64>,
) -> Result<(PathBuf, PathBuf, MsmDiagnosticDigest), MsmError> {
    fs::create_dir_all(out_dir)?;
    let digest = compute_msm_diagnostics(
        frames,
        clustering,
        base_msm,
        max_lag_multiplier,
        max_temperature_kelvin,
    );

    let json_path = out_dir.join("msm_diagnostics.json");
    let mut jw = BufWriter::new(File::create(&json_path)?);
    jw.write_all(format_msm_diagnostics_json(&digest).as_bytes())?;
    jw.flush()?;

    let csv_path = out_dir.join("msm_diagnostics.csv");
    let mut cw = BufWriter::new(File::create(&csv_path)?);
    cw.write_all(format_msm_diagnostics_csv(&digest).as_bytes())?;
    cw.flush()?;

    Ok((json_path, csv_path, digest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::{cluster_conformations_kmedoids, compute_pairwise_rmsd_matrix};
    use crate::transition::estimate_reversible_msm;
    use fold_physics::ProteinTopology;
    use fold_proto::WorkUnitId;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn implied_timescales_and_chapman_kolmogorov_digest_exports_json_and_csv() {
        let (topo, base) = ProteinTopology::build_polypeptide_from_sequence("WKDG", "WKDG");
        let mut frames = Vec::new();
        for run in 0..3u32 {
            for step in 0..14u32 {
                let mut pos = base.clone();
                let shift = ((step as f32) - 6.0).abs() * 0.55 + (run as f32) * 0.3;
                for (ai, p) in pos.iter_mut().enumerate() {
                    p[0] += (ai as f32) * 0.12 * shift;
                }
                frames.push(ConformationalFrame {
                    frame_idx: frames.len(),
                    wu_id: WorkUnitId::new(1, run, 0, step),
                    chain_step: step,
                    checkpoint_blake3: [0u8; 32],
                    temperature_kelvin: 300.0,
                    potential_energy_kcal: -50.0 + (shift as f64) * 3.0,
                    rg_angstrom: 5.5 + (shift as f64) * 0.7,
                    sasa_angstrom2: 600.0,
                    rmsd_to_seed_ang: shift as f64,
                    rmsd_to_lowest_energy_ang: shift as f64,
                    positions_angstrom: pos,
                });
            }
        }

        let rmsd_mat = compute_pairwise_rmsd_matrix(&topo, &frames);
        let clustering = cluster_conformations_kmedoids(&frames, &rmsd_mat, 5, None);
        let msm = estimate_reversible_msm(&frames, &clustering, 1, 0.5, 300.0, 2)
            .expect("estimate reversible MSM");

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = std::env::temp_dir().join(format!("beofold_ck_{}_{}", std::process::id(), nanos));
        let (json_path, csv_path, digest) =
            export_msm_diagnostics(&tmp, &frames, &clustering, &msm, 3, None)
                .expect("export MSM diagnostic digest");

        assert!(!digest.implied_timescales.is_empty());
        assert!(!digest.chapman_kolmogorov.is_empty());
        assert!(digest.ck_rmse.is_finite() && digest.ck_rmse >= 0.0);
        assert!(digest.ck_max_abs_error.is_finite() && digest.ck_max_abs_error <= 1.0);

        let json_txt = fs::read_to_string(&json_path).expect("read msm_diagnostics.json");
        assert!(json_txt.contains("\"schema_version\": \"beofold-msm-diagnostics/v1\""));
        assert!(json_txt.contains("\"implied_timescales\": ["));
        assert!(json_txt.contains("\"chapman_kolmogorov\": ["));

        let csv_txt = fs::read_to_string(&csv_path).expect("read msm_diagnostics.csv");
        assert!(csv_txt.starts_with("section,lag_multiplier_k,lag_generations,lag_time_ps,"));
        assert!(csv_txt.contains("\nITS,"));
        assert!(csv_txt.contains("\nCK,"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
