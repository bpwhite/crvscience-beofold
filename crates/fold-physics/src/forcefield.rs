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

use crate::gpu::GpuForceEngine;
use crate::morton::SpatialNeighborGrid;
use crate::topology::{ExclusionPolicy, ProteinTopology};
use crate::COULOMB_KCAL_A_E2;
use rayon::prelude::*;

/// Detailed potential energy decomposition in `kcal/mol`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnergyBreakdown {
    pub bond_kcal: f64,
    pub angle_kcal: f64,
    pub torsion_kcal: f64,
    pub lj_kcal: f64,
    pub electrostatics_gb_kcal: f64,
    pub nonpolar_solvation_kcal: f64,
    pub total_kcal: f64,
}

const NONBONDED_CUTOFF_ANG: f32 = 12.0;

/// Evaluates analytical forces `-grad(V_total)` in `kcal / (mol * A)` and potential energy breakdown
/// across CPU threads via `rayon` and a 3D Morton Z-order spatial cell grid.
pub fn evaluate_forces_and_energy(
    topology: &ProteinTopology,
    positions: &[[f32; 3]],
    forces_out: &mut [[f32; 3]],
) -> EnergyBreakdown {
    evaluate_forces_and_energy_with_gpu(topology, None, positions, forces_out)
}

/// Evaluates bonded forces in double precision on the host and dispatches O(N^2) tiled non-bonded
/// interactions (`12-6 LJ + screened Coulomb/GB + EEF1 solvation`) to `wgpu` when `gpu` is `Some`,
/// falling back seamlessly to the `rayon` 3D Morton Z-order CPU kernel if unavailable.
pub fn evaluate_forces_and_energy_with_gpu(
    topology: &ProteinTopology,
    gpu: Option<&GpuForceEngine>,
    positions: &[[f32; 3]],
    forces_out: &mut [[f32; 3]],
) -> EnergyBreakdown {
    let n = topology.atoms.len();
    assert_eq!(positions.len(), n, "positions length must match topology");
    assert_eq!(forces_out.len(), n, "forces_out length must match topology");

    for f in forces_out.iter_mut() {
        *f = [0.0, 0.0, 0.0];
    }

    let mut breakdown = EnergyBreakdown::default();

    // 1. Harmonic covalent bonds: V = 0.5 * kb * (r - r0)^2
    for b in &topology.bonds {
        let i = b.i as usize;
        let j = b.j as usize;
        let dx = (positions[i][0] - positions[j][0]) as f64;
        let dy = (positions[i][1] - positions[j][1]) as f64;
        let dz = (positions[i][2] - positions[j][2]) as f64;
        let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
        let dr = r - (b.r0_ang as f64);
        let kb = b.kb_kcal_ang2 as f64;

        breakdown.bond_kcal += 0.5 * kb * dr * dr;
        let f_over_r = -kb * dr / r;
        let fx = (f_over_r * dx) as f32;
        let fy = (f_over_r * dy) as f32;
        let fz = (f_over_r * dz) as f32;

        forces_out[i][0] += fx;
        forces_out[i][1] += fy;
        forces_out[i][2] += fz;
        forces_out[j][0] -= fx;
        forces_out[j][1] -= fy;
        forces_out[j][2] -= fz;
    }

    // 2. Harmonic valence angles: V = 0.5 * ktheta * (theta - theta0)^2
    for ang in &topology.angles {
        let i = ang.i as usize;
        let j = ang.j as usize;
        let k = ang.k as usize;

        let u = [
            (positions[i][0] - positions[j][0]) as f64,
            (positions[i][1] - positions[j][1]) as f64,
            (positions[i][2] - positions[j][2]) as f64,
        ];
        let v = [
            (positions[k][0] - positions[j][0]) as f64,
            (positions[k][1] - positions[j][1]) as f64,
            (positions[k][2] - positions[j][2]) as f64,
        ];

        let ru = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-6);
        let rv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        let u_hat = [u[0] / ru, u[1] / ru, u[2] / ru];
        let v_hat = [v[0] / rv, v[1] / rv, v[2] / rv];

        let cos_t = (u_hat[0] * v_hat[0] + u_hat[1] * v_hat[1] + u_hat[2] * v_hat[2])
            .clamp(-0.999_999, 0.999_999);
        let sin_t = (1.0 - cos_t * cos_t).sqrt().max(1e-5);
        let theta = cos_t.acos();
        let dtheta = theta - (ang.theta0_rad as f64);
        let kt = ang.ktheta_kcal_rad2 as f64;

        breakdown.angle_kcal += 0.5 * kt * dtheta * dtheta;
        let dv_dtheta = kt * dtheta;

        let coeff_i = dv_dtheta / (sin_t * ru);
        let coeff_k = dv_dtheta / (sin_t * rv);

        for d in 0..3 {
            let fi = (coeff_i * (v_hat[d] - cos_t * u_hat[d])) as f32;
            let fk = (coeff_k * (u_hat[d] - cos_t * v_hat[d])) as f32;
            forces_out[i][d] += fi;
            forces_out[k][d] += fk;
            forces_out[j][d] -= fi + fk;
        }
    }

    // 3. Periodic dihedral torsions: V = 0.5 * Vn * (1 + cos(n * phi - gamma))
    //    Singularity-free Cartesian derivatives via Blondel & Karplus (1996).
    for tor in &topology.torsions {
        let i = tor.i as usize;
        let j = tor.j as usize;
        let k = tor.k as usize;
        let l = tor.l as usize;

        let f_vec = sub3_f64(positions[i], positions[j]);
        let g_vec = sub3_f64(positions[j], positions[k]);
        let h_vec = sub3_f64(positions[l], positions[k]);

        let a_vec = cross3(f_vec, g_vec);
        let b_vec = cross3(h_vec, g_vec);

        let a_norm2 = dot3(a_vec, a_vec);
        let b_norm2 = dot3(b_vec, b_vec);
        let g_norm2 = dot3(g_vec, g_vec);
        if a_norm2 < 1e-10 || b_norm2 < 1e-10 || g_norm2 < 1e-10 {
            continue;
        }
        let g_norm = g_norm2.sqrt();
        let g_hat = [g_vec[0] / g_norm, g_vec[1] / g_norm, g_vec[2] / g_norm];

        let bxa = cross3(b_vec, a_vec);
        let sin_term = dot3(bxa, g_hat);
        let cos_term = dot3(a_vec, b_vec);
        let phi = sin_term.atan2(cos_term);

        let n_mult = tor.periodicity as f64;
        let arg = n_mult * phi - (tor.phase_rad as f64);
        let vn = tor.vn_kcal as f64;

        breakdown.torsion_kcal += 0.5 * vn * (1.0 + arg.cos());
        let dv_dphi = -0.5 * vn * n_mult * arg.sin();

        let scale_a = -g_norm / a_norm2;
        let scale_b = g_norm / b_norm2;
        let grad_i = [scale_a * a_vec[0], scale_a * a_vec[1], scale_a * a_vec[2]];
        let grad_l = [scale_b * b_vec[0], scale_b * b_vec[1], scale_b * b_vec[2]];

        let fg_over_g2 = dot3(f_vec, g_vec) / g_norm2;
        let hg_over_g2 = dot3(h_vec, g_vec) / g_norm2;

        let mut grad_j = [0.0; 3];
        let mut grad_k = [0.0; 3];
        for d in 0..3 {
            grad_j[d] = (-1.0 - fg_over_g2) * grad_i[d] - hg_over_g2 * grad_l[d];
            grad_k[d] = -(grad_i[d] + grad_j[d] + grad_l[d]);

            forces_out[i][d] -= (dv_dphi * grad_i[d]) as f32;
            forces_out[j][d] -= (dv_dphi * grad_j[d]) as f32;
            forces_out[k][d] -= (dv_dphi * grad_k[d]) as f32;
            forces_out[l][d] -= (dv_dphi * grad_l[d]) as f32;
        }
    }

    // 4. Non-bonded terms: dispatch to cross-vendor WGSL compute pipeline if available,
    //    otherwise run parallel `rayon` 3D Morton Z-order spatial neighbor evaluation.
    let mut gpu_succeeded = false;
    if let Some(gpu_engine) = gpu {
        if let Ok((e_lj, e_elec, e_solv)) = gpu_engine.evaluate_nonbonded(positions, forces_out) {
            breakdown.lj_kcal = e_lj;
            breakdown.electrostatics_gb_kcal = e_elec;
            breakdown.nonpolar_solvation_kcal = e_solv;
            gpu_succeeded = true;
        }
    }

    if !gpu_succeeded {
        evaluate_nonbonded_cpu_morton(topology, positions, forces_out, &mut breakdown);
    }

    breakdown.total_kcal = breakdown.bond_kcal
        + breakdown.angle_kcal
        + breakdown.torsion_kcal
        + breakdown.lj_kcal
        + breakdown.electrostatics_gb_kcal
        + breakdown.nonpolar_solvation_kcal;

    breakdown
}

fn evaluate_nonbonded_cpu_morton(
    topology: &ProteinTopology,
    positions: &[[f32; 3]],
    forces_out: &mut [[f32; 3]],
    breakdown: &mut EnergyBreakdown,
) {
    let n = topology.atoms.len();
    let grid = SpatialNeighborGrid::build(positions, NONBONDED_CUTOFF_ANG);
    let rc = NONBONDED_CUTOFF_ANG as f64;
    let rc2 = rc * rc;

    let per_atom_results: Vec<([f32; 3], f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let pos_i = positions[i];
            let atom_i = &topology.atoms[i];
            let mut fi = [0.0f64; 3];
            let mut e_lj = 0.0f64;
            let mut e_elec = 0.0f64;
            let mut e_solv = 0.0f64;

            grid.for_each_neighbor(pos_i, |j| {
                if i == j {
                    return;
                }
                let policy = topology.pair_policy(i, j);
                if policy == ExclusionPolicy::Excluded {
                    return;
                }
                let (lj_scale, elec_scale) = match policy {
                    ExclusionPolicy::Full => (1.0f64, 1.0f64),
                    ExclusionPolicy::Scaled14 => (0.5f64, 1.0f64 / 1.2f64),
                    ExclusionPolicy::Excluded => unreachable!(),
                };

                let pos_j = positions[j];
                let dx = (pos_i[0] - pos_j[0]) as f64;
                let dy = (pos_i[1] - pos_j[1]) as f64;
                let dz = (pos_i[2] - pos_j[2]) as f64;
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= rc2 || r2 < 1e-8 {
                    return;
                }
                let r = r2.sqrt();
                let atom_j = &topology.atoms[j];

                let sigma = 0.5 * ((atom_i.lj_sigma_ang + atom_j.lj_sigma_ang) as f64);
                let eps = lj_scale
                    * ((atom_i.lj_epsilon_kcal as f64) * (atom_j.lj_epsilon_kcal as f64)).sqrt();

                let mut du_dr_total = 0.0f64;

                if eps > 0.0 {
                    let sr = sigma / r;
                    let sr2 = sr * sr;
                    let sr6 = sr2 * sr2 * sr2;
                    let sr12 = sr6 * sr6;

                    let src = sigma / rc;
                    let src2 = src * src;
                    let src6 = src2 * src2 * src2;
                    let src12 = src6 * src6;

                    let u_lj = 4.0 * eps * (sr12 - sr6);
                    let uc_lj = 4.0 * eps * (src12 - src6);
                    let du_lj = 4.0 * eps * (-12.0 * sr12 + 6.0 * sr6) / r;
                    let duc_lj = 4.0 * eps * (-12.0 * src12 + 6.0 * src6) / rc;

                    let u_lj_shifted = u_lj - uc_lj - (r - rc) * duc_lj;
                    du_dr_total += du_lj - duc_lj;
                    if i < j {
                        e_lj += u_lj_shifted;
                    }
                }

                let q_prod = (atom_i.charge_e as f64) * (atom_j.charge_e as f64);
                if q_prod.abs() > 1e-7 {
                    let k_q = elec_scale * (COULOMB_KCAL_A_E2 / 4.0) * q_prod;
                    let u_el = k_q / r2;
                    let uc_el = k_q / rc2;
                    let du_el = -2.0 * k_q / (r2 * r);
                    let duc_el = -2.0 * k_q / (rc2 * rc);

                    let u_el_shifted = u_el - uc_el - (r - rc) * duc_el;
                    du_dr_total += du_el - duc_el;
                    if i < j {
                        e_elec += u_el_shifted;
                    }
                }

                if !atom_i.is_backbone && !atom_j.is_backbone && policy == ExclusionPolicy::Full {
                    let r_contact = 0.5 * ((atom_i.lj_sigma_ang + atom_j.lj_sigma_ang) as f64) + 1.4;
                    let width = 1.85;
                    let well = 0.28
                        * ((atom_i.lj_epsilon_kcal as f64) * (atom_j.lj_epsilon_kcal as f64)).sqrt();
                    let z = (r - r_contact) / width;
                    let z_cut = (rc - r_contact) / width;

                    let u_g = -well * (-z * z).exp();
                    let uc_g = -well * (-z_cut * z_cut).exp();
                    let du_g = -u_g * (2.0 * z / width);
                    let duc_g = -uc_g * (2.0 * z_cut / width);

                    let u_g_shifted = u_g - uc_g - (r - rc) * duc_g;
                    du_dr_total += du_g - duc_g;
                    if i < j {
                        e_solv += u_g_shifted;
                    }
                }

                let f_over_r = -du_dr_total / r;
                fi[0] += f_over_r * dx;
                fi[1] += f_over_r * dy;
                fi[2] += f_over_r * dz;
            });

            (
                [fi[0] as f32, fi[1] as f32, fi[2] as f32],
                e_lj,
                e_elec,
                e_solv,
            )
        })
        .collect();

    for (i, (fi, e_lj, e_elec, e_solv)) in per_atom_results.into_iter().enumerate() {
        forces_out[i][0] += fi[0];
        forces_out[i][1] += fi[1];
        forces_out[i][2] += fi[2];
        breakdown.lj_kcal += e_lj;
        breakdown.electrostatics_gb_kcal += e_elec;
        breakdown.nonpolar_solvation_kcal += e_solv;
    }
}

/// Computes Shrake-Rupley Solvent-Accessible Surface Area (`Angstrom^2`) using a deterministic
/// 32-point Fibonacci sphere quadrature with a `1.40 Angstrom` water probe radius.
pub fn compute_sasa_angstrom2(topology: &ProteinTopology, positions: &[[f32; 3]]) -> f64 {
    const N_PTS: usize = 32;
    const PROBE_ANG: f32 = 1.40;
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());

    let mut unit_sphere = [[0.0f32; 3]; N_PTS];
    for (k, pt) in unit_sphere.iter_mut().enumerate() {
        let y = 1.0 - (2.0 * (k as f32 + 0.5)) / (N_PTS as f32);
        let rad = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden_angle * (k as f32);
        *pt = [rad * theta.cos(), y, rad * theta.sin()];
    }

    let max_rad = 2.5f32 + PROBE_ANG;
    let grid = SpatialNeighborGrid::build(positions, 2.0 * max_rad);

    (0..topology.atoms.len())
        .into_par_iter()
        .map(|i| {
            let ri = topology.atoms[i].gb_radius_ang + PROBE_ANG;
            let pos_i = positions[i];
            let mut neighbors = Vec::new();
            grid.for_each_neighbor(pos_i, |j| {
                if i == j {
                    return;
                }
                let rj = topology.atoms[j].gb_radius_ang + PROBE_ANG;
                let dx = pos_i[0] - positions[j][0];
                let dy = pos_i[1] - positions[j][1];
                let dz = pos_i[2] - positions[j][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                let sum_r = ri + rj;
                if d2 < sum_r * sum_r {
                    neighbors.push((positions[j], rj * rj));
                }
            });

            let mut exposed = 0usize;
            'pts: for u in &unit_sphere {
                let px = pos_i[0] + ri * u[0];
                let py = pos_i[1] + ri * u[1];
                let pz = pos_i[2] + ri * u[2];
                for &(pos_j, rj2) in &neighbors {
                    let dx = px - pos_j[0];
                    let dy = py - pos_j[1];
                    let dz = pz - pos_j[2];
                    if dx * dx + dy * dy + dz * dz < rj2 {
                        continue 'pts;
                    }
                }
                exposed += 1;
            }
            let area = 4.0 * std::f64::consts::PI * ((ri as f64) * (ri as f64));
            area * (exposed as f64) / (N_PTS as f64)
        })
        .sum()
}

#[inline]
fn sub3_f64(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [
        (a[0] - b[0]) as f64,
        (a[1] - b[1]) as f64,
        (a[2] - b[2]) as f64,
    ]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
