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
    /// Maps active MSM microstate index `0..num_microstates` -> original `MicrostateClustering` index.
    pub active_microstates: Vec<usize>,
    /// Maps original `MicrostateClustering` index `0..clustering.num_microstates` -> active MSM index.
    pub orig_to_active: Vec<Option<usize>>,
    /// Original clustering microstate indices pruned during Tarjan SCC / cold-rung filtering.
    pub pruned_microstates: Vec<usize>,
    /// Röblitz–Weber PCCA+ linear transformation matrix `A` (`M x M`) satisfying `chi = Psi * A`.
    pub pcca_transform_a: Vec<Vec<f64>>,
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
/// independent `(Project, Run, Clone)` trajectory chain (filtering to cold MREMD rungs `T <= T_ref + 25 K`
/// with safe fallback if no cold transitions exist), prunes to the largest ergodic Tarjan SCC, and
/// estimates a reversible detailed-balance MSM via Prinz-Noé MLE + Cyclic Jacobi + Röblitz–Weber PCCA+.
pub fn estimate_reversible_msm(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    lag_generations: u32,
    lag_time_ps: f64,
    reference_temp_kelvin: f64,
    requested_macrostates: usize,
) -> Result<ReversibleMsmModel, MsmError> {
    estimate_reversible_msm_with_max_temp(
        frames,
        clustering,
        lag_generations,
        lag_time_ps,
        reference_temp_kelvin,
        requested_macrostates,
        None,
    )
}

/// Estimates a reversible detailed-balance MSM with an optional explicit upper temperature cutoff
/// (`max_temperature_kelvin`). When `None`, defaults to `reference_temp_kelvin + 25.0 K` with
/// automatic fallback to all temperatures if fewer than 2 connected microstates exist below cutoff.
pub fn estimate_reversible_msm_with_max_temp(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    lag_generations: u32,
    lag_time_ps: f64,
    reference_temp_kelvin: f64,
    requested_macrostates: usize,
    max_temperature_kelvin: Option<f64>,
) -> Result<ReversibleMsmModel, MsmError> {
    let k_orig = clustering.num_microstates;
    let lag = lag_generations.max(1);

    // Group frame indices by `(run, clone)` chain
    let mut chains: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (idx, f) in frames.iter().enumerate() {
        chains
            .entry((f.wu_id.run, f.wu_id.clone))
            .or_default()
            .push(idx);
    }
    for chain_indices in chains.values_mut() {
        chain_indices.sort_by_key(|&idx| frames[idx].chain_step);
    }

    let default_cold_tmax = reference_temp_kelvin + 25.0;
    let (counts_orig, total_transitions) = match max_temperature_kelvin {
        Some(tmax) => count_chain_transitions(frames, clustering, &chains, lag, k_orig, Some(tmax)),
        None => {
            let (cold_counts, cold_total) = count_chain_transitions(
                frames,
                clustering,
                &chains,
                lag,
                k_orig,
                Some(default_cold_tmax),
            );
            let cold_has_ergodic_core = cold_total > 0
                && (tarjan_largest_scc(&cold_counts).len() >= 2
                    || largest_undirected_component(&cold_counts).len() >= 2);
            if cold_has_ergodic_core {
                (cold_counts, cold_total)
            } else {
                count_chain_transitions(frames, clustering, &chains, lag, k_orig, None)
            }
        }
    };

    if total_transitions == 0 {
        return Err(MsmError::ZeroTransitions { lag });
    }

    // Tarjan SCC largest ergodic component pruning (zero dense `prior_eps` pseudocounts)
    let best_scc = tarjan_largest_scc(&counts_orig);
    let (active_microstates, is_strict_scc) = if best_scc.len() >= 2 {
        (best_scc, true)
    } else {
        let best_ucc = largest_undirected_component(&counts_orig);
        if best_ucc.len() < 2 {
            return Err(MsmError::ZeroTransitions { lag });
        }
        (best_ucc, false)
    };

    let k = active_microstates.len();
    let mut orig_to_active = vec![None; k_orig];
    for (act_idx, &orig_idx) in active_microstates.iter().enumerate() {
        orig_to_active[orig_idx] = Some(act_idx);
    }
    let pruned_microstates: Vec<usize> = (0..k_orig)
        .filter(|&orig_idx| orig_to_active[orig_idx].is_none())
        .collect();

    let mut counts = vec![vec![0u64; k]; k];
    let mut active_total_transitions = 0u64;
    for (a_i, &orig_i) in active_microstates.iter().enumerate() {
        for (a_j, &orig_j) in active_microstates.iter().enumerate() {
            let c = counts_orig[orig_i][orig_j];
            counts[a_i][a_j] = c;
            active_total_transitions += c;
        }
    }
    if active_total_transitions == 0 {
        return Err(MsmError::ZeroTransitions { lag });
    }

    // Prinz-Noé-Pande Iterative Fixed-Point Reversible Detailed-Balance MLE strictly on the
    // ergodic Tarjan SCC (zero dense off-diagonal pseudocounts).
    let mut c_reg = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            c_reg[i][j] = counts[i][j] as f64;
        }
    }
    if !is_strict_scc {
        // For strictly acyclic trajectories where max SCC < 2, add minimal diagonal-only
        // self-loop stabilizer only on zero-out-degree sink states to keep MLE well-defined.
        for i in 0..k {
            let row_sum: f64 = c_reg[i].iter().sum();
            if row_sum <= 0.0 {
                c_reg[i][i] = 1.0e-12;
            }
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
    }
    // Enforce exact Perron constant right eigenvector `psi_0 = +1` across all active states
    if k > 0 {
        for val in &mut right_eigenvectors[0] {
            *val = 1.0;
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
        let f_m = &frames[clustering.medoid_frame_indices[active_microstates[m]]];
        let f_u = &frames[clustering.medoid_frame_indices[active_microstates[unfolded_src]]];
        let f_f = &frames[clustering.medoid_frame_indices[active_microstates[folded_snk]]];
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

    // Röblitz–Weber PCCA+ Inner-Simplex Spectral Coarse-Graining (`chi = Psi * A`)
    let num_macro = requested_macrostates.min(k).max(2);
    let (fuzzy_memberships, microstate_to_macrostate, macrostates, pcca_transform_a) =
        build_pcca_macrostates(
            frames,
            clustering,
            &active_microstates,
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
        total_transitions: active_total_transitions,
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
        active_microstates,
        orig_to_active,
        pruned_microstates,
        pcca_transform_a,
    })
}

fn count_chain_transitions(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    chains: &HashMap<(u32, u32), Vec<usize>>,
    lag: u32,
    k: usize,
    max_temp_kelvin: Option<f64>,
) -> (Vec<Vec<u64>>, u64) {
    let mut counts = vec![vec![0u64; k]; k];
    let mut total_transitions = 0u64;

    for chain_indices in chains.values() {
        let step_map: HashMap<u32, usize> = chain_indices
            .iter()
            .map(|&idx| (frames[idx].chain_step, idx))
            .collect();

        for &idx_a in chain_indices {
            if let Some(tmax) = max_temp_kelvin {
                if frames[idx_a].temperature_kelvin > tmax {
                    continue;
                }
            }
            let target_step = frames[idx_a].chain_step + lag;
            if let Some(&idx_b) = step_map.get(&target_step) {
                if let Some(tmax) = max_temp_kelvin {
                    if frames[idx_b].temperature_kelvin > tmax {
                        continue;
                    }
                }
                let s_a = clustering.frame_assignments[idx_a];
                let s_b = clustering.frame_assignments[idx_b];
                counts[s_a][s_b] += 1;
                total_transitions += 1;
            }
        }
    }

    (counts, total_transitions)
}

/// Tarjan's Strongly Connected Components algorithm on directed count graph `counts[i][j] > 0`.
/// Returns the largest SCC (tie-broken by total internal transition counts, then sorted node order).
fn tarjan_largest_scc(counts: &[Vec<u64>]) -> Vec<usize> {
    let n = counts.len();
    struct TarjanState {
        next_index: usize,
        indices: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        sccs: Vec<Vec<usize>>,
    }

    fn strongconnect(v: usize, counts: &[Vec<u64>], st: &mut TarjanState) {
        st.indices[v] = Some(st.next_index);
        st.lowlink[v] = st.next_index;
        st.next_index += 1;
        st.stack.push(v);
        st.on_stack[v] = true;

        for w in 0..counts.len() {
            if counts[v][w] == 0 {
                continue;
            }
            if st.indices[w].is_none() {
                strongconnect(w, counts, st);
                st.lowlink[v] = st.lowlink[v].min(st.lowlink[w]);
            } else if st.on_stack[w] {
                st.lowlink[v] = st.lowlink[v].min(st.indices[w].unwrap());
            }
        }

        if Some(st.lowlink[v]) == st.indices[v] {
            let mut scc = Vec::new();
            while let Some(w) = st.stack.pop() {
                st.on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            scc.sort_unstable();
            st.sccs.push(scc);
        }
    }

    let mut st = TarjanState {
        next_index: 0,
        indices: vec![None; n],
        lowlink: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        sccs: Vec::new(),
    };

    for v in 0..n {
        if st.indices[v].is_none() {
            strongconnect(v, counts, &mut st);
        }
    }

    let scc_weight = |comp: &[usize]| -> u64 {
        let mut sum = 0u64;
        for &u in comp {
            for &v in comp {
                sum += counts[u][v];
            }
        }
        sum
    };

    st.sccs
        .into_iter()
        .max_by(|a, b| {
            a.len()
                .cmp(&b.len())
                .then_with(|| scc_weight(a).cmp(&scc_weight(b)))
                .then_with(|| b.cmp(a))
        })
        .unwrap_or_default()
}

/// Largest connected component of the undirected symmetrized graph `(counts[i][j] + counts[j][i]) > 0`.
fn largest_undirected_component(counts: &[Vec<u64>]) -> Vec<usize> {
    let n = counts.len();
    let mut visited = vec![false; n];
    let mut best: Vec<usize> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut comp = Vec::new();
        while let Some(u) = stack.pop() {
            comp.push(u);
            for v in 0..n {
                if !visited[v] && (counts[u][v] > 0 || counts[v][u] > 0) {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        comp.sort_unstable();
        if comp.len() > best.len() {
            best = comp;
        }
    }
    best
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

/// True Röblitz–Weber (2013) Inner-Simplex PCCA+ Spectral Coarse-Graining (`chi = Psi * A`):
/// Embeds active microstates in `M` dominant slow right eigenvectors `Psi = [psi_0=1, psi_1, ..., psi_{M-1}]`,
/// finds `M` extreme simplex vertices via Inner Simplex Algorithm (ISA) orthogonal-projection pivoting,
/// inverts `Psi_verts` to seed `A^(0)`, optimizes `A` over the feasible non-negative simplex polytope
/// to maximize crispness `I(A) = sum_{a,r} A_{r,a}^2 / A_{0,a}` (satisfying `chi = Psi * A >= 0`,
/// `sum_a chi_{i,a} = 1`), and orders macrostates by descending equilibrium population `Pi_A`.
#[allow(clippy::too_many_arguments)]
fn build_pcca_macrostates(
    frames: &[ConformationalFrame],
    clustering: &MicrostateClustering,
    active_microstates: &[usize],
    stationary_pi: &[f64],
    right_eigenvectors: &[Vec<f64>],
    committor: &[f64],
    num_macro: usize,
    kbt: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<MetastableMacrostate>, Vec<Vec<f64>>) {
    let k = stationary_pi.len();
    let m = num_macro.min(k).max(2);

    // Form `Psi` (`k x m`) with `psi[i][0] = 1.0` and `psi[i][r] = right_eigenvectors[r][i]`
    let mut psi = vec![vec![0.0f64; m]; k];
    for i in 0..k {
        psi[i][0] = 1.0;
        for r in 1..m {
            psi[i][r] = right_eigenvectors[r][i];
        }
    }

    // Step 1: Inner Simplex Algorithm (ISA) vertex pivoting in `R^m`
    let mut vertices: Vec<usize> = Vec::with_capacity(m);
    let mut v0 = 0usize;
    let mut max_n2 = -1.0f64;
    for (i, row) in psi.iter().enumerate() {
        let n2: f64 = row.iter().skip(1).map(|x| x * x).sum();
        if n2 > max_n2 {
            max_n2 = n2;
            v0 = i;
        }
    }
    vertices.push(v0);

    // Subtract `psi[v0]` so all translated vectors lie in `span(psi_1..psi_{m-1})`
    let mut orth: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..m).map(|r| psi[i][r] - psi[v0][r]).collect())
        .collect();

    while vertices.len() < m {
        let mut best_idx = 0usize;
        let mut best_n2 = -1.0f64;
        for (i, row) in orth.iter().enumerate() {
            if vertices.contains(&i) {
                continue;
            }
            let n2: f64 = row.iter().map(|x| x * x).sum();
            if n2 > best_n2 {
                best_n2 = n2;
                best_idx = i;
            }
        }
        vertices.push(best_idx);

        if vertices.len() < m {
            let dir = orth[best_idx].clone();
            let norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1.0e-12 {
                let u: Vec<f64> = dir.into_iter().map(|x| x / norm).collect();
                for row in &mut orth {
                    let dot: f64 = row.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                    for r in 0..m {
                        row[r] -= dot * u[r];
                    }
                }
            }
        }
    }

    // Step 2: Seed `A^(0) = Psi_verts^{-1}`
    let psi_verts: Vec<Vec<f64>> = vertices.iter().map(|&v| psi[v].clone()).collect();
    let mut best_a = invert_square_matrix(&psi_verts).unwrap_or_else(|| {
        let mut eye = vec![vec![0.0f64; m]; m];
        for c in 0..m {
            eye[0][c] = 1.0 / (m as f64);
        }
        eye
    });

    // Enforce exact inner-simplex feasibility & normalize `A^(0)` via `fill_a`
    let mut best_crisp = fill_a_matrix(&psi, &mut best_a).unwrap_or_else(|| {
        best_a = vec![vec![0.0f64; m]; m];
        for c in 0..m {
            best_a[0][c] = 1.0 / (m as f64);
        }
        1.0
    });

    // Step 3: Optimize Röblitz–Weber crispness `I(A) = sum_{a,r} A_{r,a}^2 / A_{0,a}`
    // over the `(m - 1) * (m - 1)` free subspace parameters via multi-scale coordinate pattern search.
    if m >= 2 {
        let mut candidate = best_a.clone();
        for &step_scale in &[0.25_f64, 0.08, 0.025, 0.008, 0.002, 0.0005] {
            for _pass in 0..6 {
                let mut improved = false;
                for r in 1..m {
                    for c in 0..(m - 1) {
                        let base_val = best_a[r][c];
                        let delta = step_scale * base_val.abs().max(0.2);
                        for dir in [-1.0_f64, 1.0_f64] {
                            candidate.clone_from(&best_a);
                            candidate[r][c] = base_val + dir * delta;
                            if let Some(crisp) = fill_a_matrix(&psi, &mut candidate) {
                                if crisp > best_crisp + 1.0e-10 {
                                    best_crisp = crisp;
                                    best_a.clone_from(&candidate);
                                    improved = true;
                                }
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
    }

    // Step 4: Evaluate exact linear subspace projection `chi = Psi * A`
    let mut raw_chi = vec![vec![0.0f64; m]; k];
    for i in 0..k {
        let mut row_sum = 0.0f64;
        for a in 0..m {
            let mut v = 0.0f64;
            for r in 0..m {
                v += psi[i][r] * best_a[r][a];
            }
            if v < 0.0 && v > -1.0e-11 {
                v = 0.0;
            }
            raw_chi[i][a] = v.max(0.0);
            row_sum += raw_chi[i][a];
        }
        if row_sum > 1.0e-30 && (row_sum - 1.0).abs() > 1.0e-13 {
            for a in 0..m {
                raw_chi[i][a] /= row_sum;
            }
        }
    }

    // Order macrostates by descending stationary population `Pi_A = sum_i pi_i * chi[i][A]`
    let raw_pops: Vec<f64> = (0..m)
        .map(|a| (0..k).map(|i| stationary_pi[i] * raw_chi[i][a]).sum())
        .collect();
    let mut macro_order: Vec<usize> = (0..m).collect();
    macro_order.sort_by(|&a, &b| {
        raw_pops[b]
            .partial_cmp(&raw_pops[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut reordered_a = vec![vec![0.0f64; m]; m];
    for r in 0..m {
        for (new_a, &old_a) in macro_order.iter().enumerate() {
            reordered_a[r][new_a] = best_a[r][old_a];
        }
    }

    let mut fuzzy_memberships = vec![vec![0.0f64; m]; k];
    let mut crisp_assign = vec![0usize; k];
    for i in 0..k {
        let mut best_a_idx = 0usize;
        let mut best_w = -1.0f64;
        for (new_a, &old_a) in macro_order.iter().enumerate() {
            let w = raw_chi[i][old_a];
            fuzzy_memberships[i][new_a] = w;
            if w > best_w {
                best_w = w;
                best_a_idx = new_a;
            }
        }
        crisp_assign[i] = best_a_idx;
    }

    let max_pop = raw_pops.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
    let mut macrostates = Vec::with_capacity(m);

    for new_a in 0..m {
        let old_a = macro_order[new_a];
        let pop = raw_pops[old_a].max(1e-30);
        let fe_kcal = -kbt * (pop / max_pop).ln();

        let members: Vec<usize> = (0..k).filter(|&i| crisp_assign[i] == new_a).collect();
        let active_set = if members.is_empty() {
            let argmax_i = (0..k)
                .max_by(|&ia, &ib| {
                    fuzzy_memberships[ia][new_a]
                        .partial_cmp(&fuzzy_memberships[ib][new_a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            vec![argmax_i]
        } else {
            members.clone()
        };

        // Pick representative frame = lowest potential energy medoid inside this macrostate
        let mut rep_micro = active_set[0];
        for &m_st in &active_set[1..] {
            let f_cand = &frames[clustering.medoid_frame_indices[active_microstates[m_st]]];
            let f_cur = &frames[clustering.medoid_frame_indices[active_microstates[rep_micro]]];
            if f_cand.potential_energy_kcal < f_cur.potential_energy_kcal {
                rep_micro = m_st;
            }
        }
        let rep_frame_idx = clustering.medoid_frame_indices[active_microstates[rep_micro]];

        let mut w_sum = 0.0f64;
        let mut rg_acc = 0.0f64;
        let mut rmsd_acc = 0.0f64;
        let mut pot_acc = 0.0f64;
        let mut comm_acc = 0.0f64;
        for i in 0..k {
            let w = stationary_pi[i] * fuzzy_memberships[i][new_a];
            let med = &frames[clustering.medoid_frame_indices[active_microstates[i]]];
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

    (fuzzy_memberships, crisp_assign, macrostates, reordered_a)
}

/// Enforces inner-simplex row-sum & non-negativity constraints on `a` (`M x M`) given `psi` (`K x M`)
/// and returns the Röblitz–Weber crispness score `I(A) = sum_{c,r} A[r][c]^2 / A[0][c]`.
fn fill_a_matrix(psi: &[Vec<f64>], a: &mut [Vec<f64>]) -> Option<f64> {
    let m = a.len();
    let k = psi.len();
    if m < 2 || k == 0 {
        return None;
    }

    // Enforce zero row-sum for modes r >= 1: A[r][m - 1] = -sum_{c=0}^{m-2} A[r][c]
    for r in 1..m {
        let mut s = 0.0f64;
        for c in 0..(m - 1) {
            if !a[r][c].is_finite() {
                return None;
            }
            s += a[r][c];
        }
        a[r][m - 1] = -s;
    }

    // Set A[0][c] = -min_i sum_{r=1}^{m-1} psi[i][r] * A[r][c] so chi[i][c] >= 0 identically
    let mut sum_a0 = 0.0f64;
    for c in 0..m {
        let mut min_g = f64::INFINITY;
        for row in psi {
            let mut g = 0.0f64;
            for r in 1..m {
                g += row[r] * a[r][c];
            }
            if g < min_g {
                min_g = g;
            }
        }
        let a0c = (-min_g).max(0.0);
        a[0][c] = a0c;
        sum_a0 += a0c;
    }

    if !sum_a0.is_finite() || sum_a0 <= 1.0e-12 {
        return None;
    }

    let inv_s = 1.0 / sum_a0;
    for r in 0..m {
        for c in 0..m {
            a[r][c] *= inv_s;
        }
    }

    let mut crispness = 0.0f64;
    for c in 0..m {
        if a[0][c] <= 1.0e-14 {
            return None;
        }
        let mut col_norm2 = 0.0f64;
        for r in 0..m {
            col_norm2 += a[r][c] * a[r][c];
        }
        crispness += col_norm2 / a[0][c];
    }
    if crispness.is_finite() {
        Some(crispness)
    } else {
        None
    }
}

fn invert_square_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 || mat.iter().any(|r| r.len() != n) {
        return None;
    }
    let mut aug = vec![vec![0.0f64; 2 * n]; n];
    for r in 0..n {
        for c in 0..n {
            aug[r][c] = mat[r][c];
        }
        aug[r][n + r] = 1.0;
    }

    for col in 0..n {
        let mut piv = col;
        let mut max_abs = aug[col][col].abs();
        for r in (col + 1)..n {
            if aug[r][col].abs() > max_abs {
                max_abs = aug[r][col].abs();
                piv = r;
            }
        }
        if max_abs < 1.0e-12 {
            return None;
        }
        if piv != col {
            aug.swap(col, piv);
        }
        let diag = aug[col][col];
        for c in 0..(2 * n) {
            aug[col][c] /= diag;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r][col];
            if factor.abs() > 0.0 {
                for c in 0..(2 * n) {
                    aug[r][c] -= factor * aug[col][c];
                }
            }
        }
    }

    Some(
        aug.into_iter()
            .map(|row| row[n..].to_vec())
            .collect(),
    )
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

    fn synthetic_frame_for_state(
        run: u32,
        step: u32,
        temp_k: f64,
        pot: f64,
        rg: f64,
        rmsd: f64,
    ) -> ConformationalFrame {
        use fold_proto::WorkUnitId;
        ConformationalFrame {
            frame_idx: 0,
            wu_id: WorkUnitId::new(1, run, 0, step),
            chain_step: step,
            checkpoint_blake3: [0u8; 32],
            temperature_kelvin: temp_k,
            potential_energy_kcal: pot,
            rg_angstrom: rg,
            sasa_angstrom2: 1200.0,
            rmsd_to_seed_ang: rmsd,
            rmsd_to_lowest_energy_ang: rmsd,
            positions_angstrom: vec![[0.0, 0.0, 0.0]],
        }
    }

    #[test]
    fn tarjan_scc_prunes_transient_and_disconnected_microstates() {
        // States {0,1,2} form a cyclic ergodic SCC; state 3 is transient (3->0 only); state 4 is isolated (4->4 only).
        let seq = [3usize, 0, 1, 2, 0, 1, 2, 0, 2, 1, 0];
        let mut frames = Vec::new();
        let mut assignments = Vec::new();
        for (step, &st) in seq.iter().enumerate() {
            frames.push(synthetic_frame_for_state(
                0,
                step as u32,
                300.0,
                -100.0 + (st as f64) * 2.0,
                8.0 + (st as f64),
                st as f64,
            ));
            assignments.push(st);
        }
        // Isolated chain for state 4
        frames.push(synthetic_frame_for_state(1, 0, 300.0, -80.0, 14.0, 4.0));
        assignments.push(4);
        frames.push(synthetic_frame_for_state(1, 1, 300.0, -80.0, 14.0, 4.0));
        assignments.push(4);

        let clustering = MicrostateClustering {
            num_microstates: 5,
            medoid_frame_indices: vec![1, 2, 3, 0, seq.len()],
            frame_assignments: assignments,
            frame_to_medoid_rmsd: vec![0.0; frames.len()],
            microstate_counts: vec![4, 3, 3, 1, 2],
            mean_cluster_radius_ang: vec![0.1; 5],
        };

        let msm = estimate_reversible_msm(&frames, &clustering, 1, 1.0, 300.0, 2)
            .expect("Tarjan SCC MSM");
        assert_eq!(msm.num_microstates, 3);
        assert_eq!(msm.active_microstates, vec![0, 1, 2]);
        assert_eq!(msm.pruned_microstates, vec![3, 4]);
        assert_eq!(
            msm.orig_to_active,
            vec![Some(0), Some(1), Some(2), None, None]
        );
        let pi_sum: f64 = msm.stationary_pi.iter().sum();
        assert!((pi_sum - 1.0).abs() < 1e-10);
        assert!(msm.detailed_balance_max_error < 1e-10);
    }

    #[test]
    fn cold_rung_filtering_excludes_hot_mremd_denatured_transitions() {
        // Cold rung at 285 K visits states 0<->1 (4 transitions); hot rung at 400 K visits states 2<->3 (4 transitions).
        let mut frames = Vec::new();
        let mut assignments = Vec::new();
        for (step, &st) in [0usize, 1, 0, 1, 0].iter().enumerate() {
            frames.push(synthetic_frame_for_state(
                0,
                step as u32,
                285.0,
                -110.0 + (st as f64),
                8.0 + (st as f64),
                st as f64,
            ));
            assignments.push(st);
        }
        for (step, &st) in [2usize, 3, 2, 3, 2].iter().enumerate() {
            frames.push(synthetic_frame_for_state(
                1,
                step as u32,
                400.0,
                -40.0 + (st as f64),
                15.0 + (st as f64),
                st as f64,
            ));
            assignments.push(st);
        }

        let clustering = MicrostateClustering {
            num_microstates: 4,
            medoid_frame_indices: vec![0, 1, 5, 6],
            frame_assignments: assignments,
            frame_to_medoid_rmsd: vec![0.0; frames.len()],
            microstate_counts: vec![3, 2, 3, 2],
            mean_cluster_radius_ang: vec![0.1; 4],
        };

        let msm = estimate_reversible_msm(&frames, &clustering, 1, 1.0, 300.0, 2)
            .expect("cold-rung filtered MSM");
        assert_eq!(msm.total_transitions, 4);
        assert_eq!(msm.active_microstates, vec![0, 1]);
        assert_eq!(msm.pruned_microstates, vec![2, 3]);
    }

    #[test]
    fn roblitz_weber_pcca_satisfies_exact_subspace_chi_equals_psi_a() {
        // 6-state 3-basin metastable chain: {0,1} <-> {2,3} <-> {4,5} with strong intra-basin coupling
        let mut frames = Vec::new();
        let mut assignments = Vec::new();
        let mut seq = Vec::new();
        for _ in 0..8 {
            seq.extend_from_slice(&[0usize, 1, 0, 1, 0, 1, 2]);
            seq.extend_from_slice(&[2, 3, 2, 3, 2, 3, 4]);
            seq.extend_from_slice(&[4, 5, 4, 5, 4, 5, 2, 0]);
        }
        for (step, &st) in seq.iter().enumerate() {
            frames.push(synthetic_frame_for_state(
                0,
                step as u32,
                300.0,
                -120.0 + (st as f64) * 3.0,
                7.5 + (st as f64) * 1.2,
                (st as f64) * 1.5,
            ));
            assignments.push(st);
        }
        let mut medoids = vec![0usize; 6];
        let mut counts = vec![0usize; 6];
        for (idx, &st) in assignments.iter().enumerate() {
            if counts[st] == 0 {
                medoids[st] = idx;
            }
            counts[st] += 1;
        }
        let clustering = MicrostateClustering {
            num_microstates: 6,
            medoid_frame_indices: medoids,
            frame_assignments: assignments,
            frame_to_medoid_rmsd: vec![0.0; frames.len()],
            microstate_counts: counts,
            mean_cluster_radius_ang: vec![0.1; 6],
        };

        let msm = estimate_reversible_msm(&frames, &clustering, 1, 1.0, 300.0, 3)
            .expect("3-basin Röblitz-Weber PCCA+");
        assert_eq!(msm.num_microstates, 6);
        assert_eq!(msm.macrostates.len(), 3);
        let m = 3usize;
        assert_eq!(msm.pcca_transform_a.len(), m);

        for i in 0..msm.num_microstates {
            let mut row_sum = 0.0f64;
            for a in 0..m {
                let chi_ia = msm.fuzzy_memberships[i][a];
                assert!(chi_ia >= -1e-12 && chi_ia <= 1.0 + 1e-12);
                row_sum += chi_ia;
                let mut psi_dot_a = 0.0f64;
                for r in 0..m {
                    psi_dot_a += msm.right_eigenvectors[r][i] * msm.pcca_transform_a[r][a];
                }
                assert!(
                    (chi_ia - psi_dot_a).abs() < 1e-9,
                    "chi != Psi * A at ({i}, {a}): {chi_ia} vs {psi_dot_a}"
                );
            }
            assert!((row_sum - 1.0).abs() < 1e-9);
        }
    }
}
