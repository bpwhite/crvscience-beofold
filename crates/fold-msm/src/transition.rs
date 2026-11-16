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
use fold_physics::K_BOLTZMANN_KCAL;
use std::collections::HashMap;

/// Complete reversible Markov State Model (MSM) kinetic, thermodynamic, and PCCA+ decomposition.
#[derive(Debug, Clone)]
pub struct ReversibleMsmModel {
    pub num_microstates: usize,
    pub lag_generations: u32,
    pub lag_time_ps: f64,
    pub reference_temp_kelvin: f64,
    pub total_transitions: u64,
    /// Raw integer transition count matrix `C_ij(tau)`.
    pub count_matrix: Vec<Vec<u64>>,
    /// Reversible Maximum-Likelihood row-stochastic transition matrix `T_ij` (`pi_i T_ij = pi_j T_ji`).
    pub transition_matrix: Vec<Vec<f64>>,
    /// Equilibrium Boltzmann stationary distribution `pi_i` (`sum_i pi_i = 1`).
    pub stationary_pi: Vec<f64>,
    /// Microstate relative free energy `F_i = -k_B * T_ref * ln(pi_i / pi_max)` (`kcal/mol`, min = `0.0`).
    pub microstate_free_energy_kcal: Vec<f64>,
    /// Maximum detailed-balance residual `max_{i,j} |pi_i T_ij - pi_j T_ji|`.
    pub detailed_balance_max_error: f64,
    /// Descending real eigenvalues `1.0 = lambda_1 >= lambda_2 >= ... >= lambda_K`.
    pub eigenvalues: Vec<f64>,
    /// Right eigenvectors `psi_k(i)` (`k = 0..K`, where `psi_0 = [1, 1, ...]`).
    pub right_eigenvectors: Vec<Vec<f64>>,
    /// Implied relaxation timescales `t_k = -lag_time_ps / ln(lambda_k)` (`ps`) for slow modes `k >= 1`.
    pub implied_timescales_ps: Vec<f64>,
    /// Transition Path Theory (`TPT`) forward folding committor `p_fold(i)` (`0.0 = unfolded -> 1.0 = folded`).
    pub folding_committor: Vec<f64>,
    pub unfolded_source_microstate: usize,
    pub folded_sink_microstate: usize,
    /// PCCA+ coarse-grained metastable macrostates.
    pub macrostates: Vec<MetastableMacrostate>,
    /// Crisp macrostate assignment (`0 .. macrostates.len()`) for each microstate.
    pub microstate_to_macrostate: Vec<usize>,
    /// Fuzzy PCCA+ simplex membership matrix `chi[microstate][macrostate]` (`sum_A chi_{i,A} = 1`).
    pub fuzzy_memberships: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct MetastableMacrostate {
    pub macro_id: usize,
    pub biophysical_label: String,
    pub member_microstates: Vec<usize>,
    pub stationary_population: f64,
    pub free_energy_kcal: f64,
    pub representative_frame_idx: usize,
    pub mean_rg_angstrom: f64,
    pub mean_rmsd_best_ang: f64,
    pub mean_potential_kcal: f64,
    pub mean_committor: f64,
}

/// Counts sequential transitions `(state(t) -> state(t + lag_generations))` strictly within each
/// independent `(Project, Run, Clone)` trajectory chain and estimates a reversible detailed-balance
/// Markov State Model via Prinz-Noé fixed-point MLE + Cyclic Jacobi eigendecomposition + PCCA+.
pub fn estimate_reversible_msm(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    lag_generations: u32,
    lag_time_ps: f64,
    reference_temp_kelvin: f64,
    requested_macrostates: usize,
) -> Result<ReversibleMsmModel, MsmError> {
    let k = clustering.num_microstates;
    let lag = lag_generations.max(1);

    // Group frame indices by `(run, clone)` chain
    let mut chains: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (idx, f) in frames.iter().enumerate() {
        chains
            .entry((f.wu_id.run, f.wu_id.clone))
            .or_default()
            .push(idx);
    }

    let mut counts = vec![vec![0u64; k]; k];
    let mut total_transitions = 0u64;

    for chain_indices in chains.values_mut() {
        chain_indices.sort_by_key(|&idx| frames[idx].chain_step);
        // Map `chain_step -> frame_idx` for exact `step -> step + lag` lookup
        let step_map: HashMap<u32, usize> = chain_indices
            .iter()
            .map(|&idx| (frames[idx].chain_step, idx))
            .collect();

        for &idx_a in chain_indices.iter() {
            let target_step = frames[idx_a].chain_step + lag;
            if let Some(&idx_b) = step_map.get(&target_step) {
                let s_a = clustering.frame_assignments[idx_a];
                let s_b = clustering.frame_assignments[idx_b];
                counts[s_a][s_b] += 1;
                total_transitions += 1;
            }
        }
    }

    if total_transitions == 0 {
        return Err(MsmError::ZeroTransitions { lag });
    }

    // Prinz-Noé-Pande Iterative Fixed-Point Reversible Detailed-Balance MLE:
    // Weak symmetric connectivity prior (`1e-5`) prevents reducible disconnected subgraphs on sparse vaults
    // while preserving high-count transition rates to >99.99%.
    let prior_eps = 1.0e-5f64;
    let mut c_reg = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let diag_boost = if i == j { 1.0e-4 } else { prior_eps };
            c_reg[i][j] = (counts[i][j] as f64) + diag_boost;
        }
    }

    let mut c_sym = vec![vec![0.0f64; k]; k];
    let mut r_row = vec![0.0f64; k];
    for i in 0..k {
        for j in 0..k {
            c_sym[i][j] = c_reg[i][j] + c_reg[j][i];
            r_row[i] += c_reg[i][j];
        }
    }

    let mut x_mat = c_sym.clone();
    for _iter in 0..5_000 {
        let x_row: Vec<f64> = (0..k).map(|i| x_mat[i].iter().sum::<f64>().max(1e-30)).collect();
        let z_old: f64 = x_row.iter().sum::<f64>().max(1e-30);

        let mut x_next = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in i..k {
                let num = c_sym[i][j];
                if num > 0.0 {
                    let denom = (r_row[i] / x_row[i]) + (r_row[j] / x_row[j]);
                    let val = num / denom.max(1e-30);
                    x_next[i][j] = val;
                    x_next[j][i] = val;
                }
            }
        }

        let x_next_row: Vec<f64> = (0..k).map(|i| x_next[i].iter().sum::<f64>()).collect();
        let z_new: f64 = x_next_row.iter().sum::<f64>().max(1e-30);
        let mut max_pi_delta = 0.0f64;
        for i in 0..k {
            let diff = (x_next_row[i] / z_new - x_row[i] / z_old).abs();
            if diff > max_pi_delta {
                max_pi_delta = diff;
            }
        }
        x_mat = x_next;
        if max_pi_delta < 1.0e-13 {
            break;
        }
    }

    let x_row: Vec<f64> = (0..k).map(|i| x_mat[i].iter().sum::<f64>().max(1e-30)).collect();
    let z_sum: f64 = x_row.iter().sum::<f64>().max(1e-30);
    let stationary_pi: Vec<f64> = x_row.iter().map(|&xi| xi / z_sum).collect();

    let mut transition_matrix = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            transition_matrix[i][j] = x_mat[i][j] / x_row[i];
        }
    }

    let mut db_err = 0.0f64;
    for i in 0..k {
        for j in 0..k {
            let r = (stationary_pi[i] * transition_matrix[i][j]
                - stationary_pi[j] * transition_matrix[j][i])
                .abs();
            if r > db_err {
                db_err = r;
            }
        }
    }

    let pi_max = stationary_pi
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1e-30);
    let kbt = K_BOLTZMANN_KCAL * reference_temp_kelvin.max(1.0);
    let microstate_free_energy_kcal: Vec<f64> = stationary_pi
        .iter()
        .map(|&pi_i| -kbt * (pi_i.max(1e-30) / pi_max).ln())
        .collect();

    // Symmetrized Markov operator `S_ij = x_ij / sqrt(X_i * X_j)` for exact real eigendecomposition
    let mut s_sym = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            s_sym[i][j] = x_mat[i][j] / (x_row[i] * x_row[j]).sqrt();
        }
    }

    let (eigenvalues, sym_eigvecs) = cyclic_jacobi_symmetric_eigen(&s_sym);

    // Transform symmetric eigenvectors `u_m(i)` to right eigenvectors `psi_m(i) = u_m(i) / sqrt(pi_i)`
    let mut right_eigenvectors = vec![vec![0.0f64; k]; k];
    for mode in 0..k {
        for i in 0..k {
            right_eigenvectors[mode][i] =
                sym_eigvecs[i][mode] / stationary_pi[i].max(1e-30).sqrt();
        }
        // Normalize mode 0 to +1 convention
        if mode == 0 && right_eigenvectors[0][0] < 0.0 {
            for val in &mut right_eigenvectors[0] {
                *val = -*val;
            }
        }
    }

    let mut implied_timescales_ps = Vec::with_capacity(k.saturating_sub(1));
    for &lam in eigenvalues.iter().skip(1) {
        if lam > 1.0e-9 && lam < 0.999_999_999 {
            implied_timescales_ps.push(-lag_time_ps / lam.ln());
        } else if lam >= 0.999_999_999 {
            implied_timescales_ps.push(lag_time_ps * 1.0e6);
        } else {
            implied_timescales_ps.push(0.0);
        }
    }

    // Identify Unfolded Source (`U`: largest medoid R_g) and Folded Sink (`F`: lowest medoid V_pot)
    let mut unfolded_src = 0usize;
    let mut folded_snk = 0usize;
    for m in 1..k {
        let f_m = &frames[clustering.medoid_frame_indices[m]];
        let f_u = &frames[clustering.medoid_frame_indices[unfolded_src]];
        let f_f = &frames[clustering.medoid_frame_indices[folded_snk]];
        if f_m.rg_angstrom > f_u.rg_angstrom {
            unfolded_src = m;
        }
        if f_m.potential_energy_kcal < f_f.potential_energy_kcal {
            folded_snk = m;
        }
    }
    if unfolded_src == folded_snk && k >= 2 {
        unfolded_src = (folded_snk + 1) % k;
    }

    let folding_committor =
        solve_forward_committor(&transition_matrix, unfolded_src, folded_snk);

    // PCCA+ Simplex Spectral Coarse-Graining into `M` Metastable Macrostates
    let num_macro = requested_macrostates.clamp(2, k.max(2));
    let (fuzzy_memberships, microstate_to_macrostate, macrostates) = build_pcca_macrostates(
        frames,
        clustering,
        &stationary_pi,
        &right_eigenvectors,
        &folding_committor,
        num_macro,
        kbt,
    );

    Ok(ReversibleMsmModel {
        num_microstates: k,
        lag_generations: lag,
        lag_time_ps,
        reference_temp_kelvin,
        total_transitions,
        count_matrix: counts,
        transition_matrix,
        stationary_pi,
        microstate_free_energy_kcal,
        detailed_balance_max_error: db_err,
        eigenvalues,
        right_eigenvectors,
        implied_timescales_ps,
        folding_committor,
        unfolded_source_microstate: unfolded_src,
        folded_sink_microstate: folded_snk,
        macrostates,
        microstate_to_macrostate,
        fuzzy_memberships,
    })
}

/// Pure-Rust Cyclic Jacobi Symmetric Eigensolver (`A = V * diag(eigenvalues) * V^T`),
/// returning eigenvalues sorted descending alongside column-oriented matrix `V[row][col]`.
pub fn cyclic_jacobi_symmetric_eigen(sym: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = sym.len();
    let mut a = sym.to_vec();
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _sweep in 0..100 {
        let mut off_norm2 = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_norm2 += a[p][q] * a[p][q];
            }
        }
        if off_norm2.sqrt() < 1.0e-14 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < 1.0e-15 {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let app = a[p][p];
                let aqq = a[q][q];
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * arp + c * arq;
                        a[q][r] = a[r][q];
                    }
                }
                for r in 0..n {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[j][j]
            .partial_cmp(&a[i][i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_evals: Vec<f64> = order.iter().map(|&col| a[col][col]).collect();
    let mut sorted_vecs = vec![vec![0.0f64; n]; n];
    for r in 0..n {
        for (new_col, &old_col) in order.iter().enumerate() {
            sorted_vecs[r][new_col] = v[r][old_col];
        }
    }
    (sorted_evals, sorted_vecs)
}

/// Solves the exact Dirichlet committor equation `(I - T_II) q_I^+ = T_{I, F}` for `p_fold`.
fn solve_forward_committor(
    t_mat: &[Vec<f64>],
    unfolded_src: usize,
    folded_snk: usize,
) -> Vec<f64> {
    let k = t_mat.len();
    let mut q = vec![0.0f64; k];
    q[folded_snk] = 1.0;
    if unfolded_src == folded_snk || k <= 2 {
        return q;
    }

    let inter: Vec<usize> = (0..k)
        .filter(|&i| i != unfolded_src && i != folded_snk)
        .collect();
    let m = inter.len();
    let mut mat = vec![vec![0.0f64; m]; m];
    let mut rhs = vec![0.0f64; m];

    for (r, &st_i) in inter.iter().enumerate() {
        for (c, &st_j) in inter.iter().enumerate() {
            mat[r][c] = (if r == c { 1.0 } else { 0.0 }) - t_mat[st_i][st_j];
        }
        rhs[r] = t_mat[st_i][folded_snk];
    }

    if let Some(sol) = gaussian_solve(mat, rhs) {
        for (idx, &st) in inter.iter().enumerate() {
            q[st] = sol[idx].clamp(0.0, 1.0);
        }
    }
    q
}

fn gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        let mut max_abs = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > max_abs {
                max_abs = a[r][col].abs();
                piv = r;
            }
        }
        if max_abs < 1.0e-14 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] / diag;
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }

    let mut x = vec![0.0f64; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in (r + 1)..n {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

/// PCCA+ Simplex Spectral Coarse-Graining:
/// Embeds microstates in the `(M - 1)` dominant slow right eigenvectors `(psi_1 .. psi_{M-1})`,
/// identifies `M` extreme simplex vertices via furthest-point selection, computes fuzzy simplex
/// memberships `chi_{i, A}` via inverse-distance / barycentric projection, and orders macrostates
/// from lowest free energy (`Macrostate 0`) upward.
fn build_pcca_macrostates(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    stationary_pi: &[f64],
    right_eigenvectors: &[Vec<f64>],
    committor: &[f64],
    num_macro: usize,
    kbt: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<MetastableMacrostate>) {
    let k = stationary_pi.len();
    let dim = (num_macro - 1).min(k.saturating_sub(1)).max(1);

    let coords: Vec<Vec<f64>> = (0..k)
        .map(|i| (1..=dim).map(|mode| right_eigenvectors[mode][i]).collect())
        .collect();

    // Seed Vertex 0 as microstate with largest slow-subspace norm
    let mut vertices: Vec<usize> = Vec::with_capacity(num_macro);
    let mut v0 = 0usize;
    let mut max_n2 = -1.0f64;
    for (i, pt) in coords.iter().enumerate() {
        let n2: f64 = pt.iter().map(|x| x * x).sum();
        if n2 > max_n2 {
            max_n2 = n2;
            v0 = i;
        }
    }
    vertices.push(v0);

    while vertices.len() < num_macro {
        let mut best_idx = 0usize;
        let mut best_min_d2 = -1.0f64;
        for i in 0..k {
            if vertices.contains(&i) {
                continue;
            }
            let min_d2 = vertices
                .iter()
                .map(|&v| sq_dist(&coords[i], &coords[v]))
                .fold(f64::INFINITY, f64::min);
            if min_d2 > best_min_d2 {
                best_min_d2 = min_d2;
                best_idx = i;
            }
        }
        vertices.push(best_idx);
    }

    // Compute fuzzy PCCA+ simplex memberships `chi[i][A]` (`sum_A chi[i][A] == 1.0`)
    let mut raw_chi = vec![vec![0.0f64; num_macro]; k];
    for i in 0..k {
        if let Some(pos) = vertices.iter().position(|&v| v == i) {
            raw_chi[i][pos] = 1.0;
        } else {
            let mut inv_sum = 0.0f64;
            for (m_idx, &v) in vertices.iter().enumerate() {
                let d = sq_dist(&coords[i], &coords[v]).sqrt().max(1.0e-6);
                let w = 1.0 / (d * d);
                raw_chi[i][m_idx] = w;
                inv_sum += w;
            }
            for m_idx in 0..num_macro {
                raw_chi[i][m_idx] /= inv_sum.max(1.0e-30);
            }
        }
    }

    // Order macrostates by descending stationary population `Pi_A = sum_i pi_i * chi[i][A]`
    let raw_pops: Vec<f64> = (0..num_macro)
        .map(|a| (0..k).map(|i| stationary_pi[i] * raw_chi[i][a]).sum())
        .collect();
    let mut macro_order: Vec<usize> = (0..num_macro).collect();
    macro_order.sort_by(|&a, &b| {
        raw_pops[b]
            .partial_cmp(&raw_pops[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fuzzy_memberships = vec![vec![0.0f64; num_macro]; k];
    let mut crisp_assign = vec![0usize; k];
    for i in 0..k {
        let mut best_a = 0usize;
        let mut best_w = -1.0f64;
        for (new_a, &old_a) in macro_order.iter().enumerate() {
            let w = raw_chi[i][old_a];
            fuzzy_memberships[i][new_a] = w;
            if w > best_w {
                best_w = w;
                best_a = new_a;
            }
        }
        crisp_assign[i] = best_a;
    }

    let max_pop = raw_pops.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
    let mut macrostates = Vec::with_capacity(num_macro);

    for new_a in 0..num_macro {
        let old_a = macro_order[new_a];
        let pop = raw_pops[old_a].max(1e-30);
        let fe_kcal = -kbt * (pop / max_pop).ln();

        let members: Vec<usize> = (0..k).filter(|&i| crisp_assign[i] == new_a).collect();
        let active_set = if members.is_empty() {
            vec![vertices[old_a]]
        } else {
            members.clone()
        };

        // Pick representative frame = lowest potential energy medoid inside this macrostate
        let mut rep_micro = active_set[0];
        for &m_st in &active_set[1..] {
            let f_cand = &frames[clustering.medoid_frame_indices[m_st]];
            let f_cur = &frames[clustering.medoid_frame_indices[rep_micro]];
            if f_cand.potential_energy_kcal < f_cur.potential_energy_kcal {
                rep_micro = m_st;
            }
        }
        let rep_frame_idx = clustering.medoid_frame_indices[rep_micro];

        let mut w_sum = 0.0f64;
        let mut rg_acc = 0.0f64;
        let mut rmsd_acc = 0.0f64;
        let mut pot_acc = 0.0f64;
        let mut comm_acc = 0.0f64;
        for i in 0..k {
            let w = stationary_pi[i] * fuzzy_memberships[i][new_a];
            let med = &frames[clustering.medoid_frame_indices[i]];
            w_sum += w;
            rg_acc += w * med.rg_angstrom;
            rmsd_acc += w * med.rmsd_to_lowest_energy_ang;
            pot_acc += w * med.potential_energy_kcal;
            comm_acc += w * committor[i];
        }
        let inv_w = 1.0 / w_sum.max(1e-30);
        let mean_comm = comm_acc * inv_w;

        let label = if mean_comm >= 0.72 {
            "folded_basin".to_string()
        } else if mean_comm <= 0.28 {
            "unfolded_ensemble".to_string()
        } else {
            format!("intermediate_{}", new_a)
        };

        macrostates.push(MetastableMacrostate {
            macro_id: new_a,
            biophysical_label: label,
            member_microstates: members,
            stationary_population: pop,
            free_energy_kcal: fe_kcal,
            representative_frame_idx: rep_frame_idx,
            mean_rg_angstrom: rg_acc * inv_w,
            mean_rmsd_best_ang: rmsd_acc * inv_w,
            mean_potential_kcal: pot_acc * inv_w,
            mean_committor: mean_comm,
        });
    }

    (fuzzy_memberships, crisp_assign, macrostates)
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_eigensolver_recovers_exact_stochastic_spectrum() {
        let sym = vec![
            vec![0.85, 0.12, 0.03],
            vec![0.12, 0.78, 0.10],
            vec![0.03, 0.10, 0.87],
        ];
        let (evals, evecs) = cyclic_jacobi_symmetric_eigen(&sym);
        assert!((evals[0] - 1.0).abs() < 1e-12);
        for col in 0..3 {
            for i in 0..3 {
                let lhs: f64 = (0..3).map(|j| sym[i][j] * evecs[j][col]).sum();
                assert!((lhs - evals[col] * evecs[i][col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn end_to_end_rmsd_clustering_reversible_msm_and_pmf() {
        use crate::clustering::{
            cluster_conformations_kmedoids, compute_pairwise_rmsd_matrix, ConformationalFrame,
        };
        use crate::landscape::compute_2d_pmf_grid;
        use fold_physics::ProteinTopology;
        use fold_proto::WorkUnitId;

        let (topo, base_pos) = ProteinTopology::build_trp_cage_seed();
        let wu = WorkUnitId::new(1, 0, 0, 0);

        let mut frames = Vec::with_capacity(6);
        for step in 0..6u32 {
            let mut pos = base_pos.clone();
            if step == 1 {
                // Pure rigid translation to test Kabsch/Horn RMSD translation invariance
                for p in &mut pos {
                    p[0] += 15.0;
                    p[1] -= 9.5;
                    p[2] += 4.25;
                }
            } else if step >= 2 {
                let amp = 0.35 * ((step - 1) as f32);
                for (atom_i, p) in pos.iter_mut().enumerate() {
                    let phase = (atom_i as f32) * 0.23 + (step as f32) * 0.7;
                    p[0] += amp * phase.sin();
                    p[1] += amp * phase.cos();
                }
            }
            let rmsd_seed = fold_physics::compute_ca_rmsd(&topo, &base_pos, &pos);
            frames.push(ConformationalFrame {
                frame_idx: step as usize,
                wu_id: wu,
                chain_step: step,
                checkpoint_blake3: [step as u8; 32],
                temperature_kelvin: 300.0,
                potential_energy_kcal: -120.0 + (step as f64) * 3.5,
                rg_angstrom: fold_physics::compute_radius_of_gyration(&topo, &pos),
                sasa_angstrom2: 1400.0 - (step as f64) * 15.0,
                rmsd_to_seed_ang: rmsd_seed,
                rmsd_to_lowest_energy_ang: rmsd_seed,
                positions_angstrom: pos,
            });
        }

        let rmsd_mat = compute_pairwise_rmsd_matrix(&topo, &frames);
        for i in 0..frames.len() {
            assert!(rmsd_mat[i][i].abs() < 1e-6);
            for j in 0..frames.len() {
                assert!((rmsd_mat[i][j] - rmsd_mat[j][i]).abs() < 1e-9);
            }
        }
        assert!(rmsd_mat[0][1] < 1e-4, "rigid translation must yield ~0 RMSD");

        let clustering = cluster_conformations_kmedoids(&frames, &rmsd_mat, 3, None);
        assert!(clustering.num_microstates >= 2 && clustering.num_microstates <= 3);

        let msm = estimate_reversible_msm(&frames, &clustering, 1, 1.0, 300.0, 2)
            .expect("estimate reversible MSM");
        for row in &msm.transition_matrix {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
        let pi_sum: f64 = msm.stationary_pi.iter().sum();
        assert!((pi_sum - 1.0).abs() < 1e-9);
        assert!(msm.detailed_balance_max_error < 1e-8);

        let grid = compute_2d_pmf_grid(&frames, &clustering, &msm, 12, 12);
        let mut min_fe = f64::INFINITY;
        for col in &grid.free_energy_kcal {
            for &val in col {
                assert!(val.is_finite());
                assert!(val >= 0.0 && val <= grid.max_capped_kcal);
                if val < min_fe {
                    min_fe = val;
                }
            }
        }
        assert!(min_fe.abs() < 1e-9);
    }
}
