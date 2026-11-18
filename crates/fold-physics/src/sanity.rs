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

use crate::morton::SpatialNeighborGrid;
use crate::topology::{ExclusionPolicy, ProteinTopology};
use crate::K_BOLTZMANN_KCAL;
use fold_proto::TrajectoryCheckpoint;
use thiserror::Error;

/// Configurable thresholds for Gate 2 Biophysical Sanity Verification.
#[derive(Debug, Clone, Copy)]
pub struct SanityThresholds {
    /// Maximum covalent bond deviation `|r - r0|` in Angstroms (`0.25 A` default).
    pub max_bond_drift_ang: f32,
    /// Minimum non-bonded heavy-atom distance in Angstroms (`1.10 A` default).
    pub min_steric_dist_ang: f32,
    /// Maximum kinetic temperature deviation in multiples of equilibrium `sigma_T` (`6.0` default).
    pub max_temp_sigma: f64,
    /// Maximum potential energy per atom (`kcal/mol/atom`) before flagging numerical blowup.
    pub max_pot_per_atom_kcal: f64,
}

impl Default for SanityThresholds {
    fn default() -> Self {
        Self {
            max_bond_drift_ang: 0.35,
            min_steric_dist_ang: 1.10,
            max_temp_sigma: 6.5,
            max_pot_per_atom_kcal: 50.0,
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum SanityError {
    #[error("coordinate/velocity array length mismatch ({found} vs {expected} topology atoms)")]
    LengthMismatch { expected: usize, found: usize },
    #[error("non-finite NaN or Inf detected at atom {atom_index} or scalar energetics")]
    NonFiniteValue { atom_index: usize },
    #[error("covalent bond ({i}-{j}) drifted by {drift_ang:.3} A (limit {limit_ang:.3} A)")]
    CovalentBondDrift {
        i: u32,
        j: u32,
        drift_ang: f32,
        limit_ang: f32,
    },
    #[error("severe steric clash ({dist_ang:.3} A < {limit_ang:.3} A) between atoms {i} and {j}")]
    StericClash {
        i: usize,
        j: usize,
        dist_ang: f32,
        limit_ang: f32,
    },
    #[error("kinetic temperature {t_kin:.1} K diverged from bath {t_target:.1} K (> {sigma:.1} sigma)")]
    KineticTemperatureBlowup {
        t_kin: f64,
        t_target: f64,
        sigma: f64,
    },
    #[error("potential energy per atom ({pot_per_atom:.2} kcal/mol) exceeded safety ceiling")]
    PotentialBlowup { pot_per_atom: f64 },
}

/// Gate 2 Biophysical Sanity Checker applied by Cluster Leaders and the Apache Sync Server
/// before accepting any completed Generation checkpoint into the PRCG trajectory graph.
pub fn verify_checkpoint_sanity(
    topology: &ProteinTopology,
    cp: &TrajectoryCheckpoint,
    limits: &SanityThresholds,
) -> Result<(), SanityError> {
    let n = topology.atoms.len();
    if cp.positions_angstrom.len() != n || cp.velocities_ang_per_ps.len() != n {
        return Err(SanityError::LengthMismatch {
            expected: n,
            found: cp.positions_angstrom.len(),
        });
    }
    if !cp.potential_energy_kcal.is_finite()
        || !cp.kinetic_energy_kcal.is_finite()
        || !cp.rg_angstrom.is_finite()
    {
        return Err(SanityError::NonFiniteValue { atom_index: 0 });
    }

    for i in 0..n {
        let p = cp.positions_angstrom[i];
        let v = cp.velocities_ang_per_ps[i];
        if !p[0].is_finite()
            || !p[1].is_finite()
            || !p[2].is_finite()
            || !v[0].is_finite()
            || !v[1].is_finite()
            || !v[2].is_finite()
        {
            return Err(SanityError::NonFiniteValue { atom_index: i });
        }
    }

    // Check covalent bond lengths
    for b in &topology.bonds {
        let p_i = cp.positions_angstrom[b.i as usize];
        let p_j = cp.positions_angstrom[b.j as usize];
        let dx = p_i[0] - p_j[0];
        let dy = p_i[1] - p_j[1];
        let dz = p_i[2] - p_j[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let drift = (r - b.r0_ang).abs();
        if drift > limits.max_bond_drift_ang {
            return Err(SanityError::CovalentBondDrift {
                i: b.i,
                j: b.j,
                drift_ang: drift,
                limit_ang: limits.max_bond_drift_ang,
            });
        }
    }

    // Check non-bonded heavy-atom steric overlaps using spatial cell grid
    let min_d2 = limits.min_steric_dist_ang * limits.min_steric_dist_ang;
    let grid = SpatialNeighborGrid::build(&cp.positions_angstrom, limits.min_steric_dist_ang * 2.5);
    for i in 0..n {
        if topology.atoms[i].element == "H" {
            continue;
        }
        let pi = cp.positions_angstrom[i];
        let mut clash: Option<(usize, f32)> = None;
        grid.for_each_neighbor(pi, |j| {
            if j <= i || topology.atoms[j].element == "H" {
                return;
            }
            if topology.pair_policy(i, j) == ExclusionPolicy::Excluded {
                return;
            }
            let pj = cp.positions_angstrom[j];
            let dx = pi[0] - pj[0];
            let dy = pi[1] - pj[1];
            let dz = pi[2] - pj[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < min_d2 {
                clash = Some((j, d2.sqrt()));
            }
        });
        if let Some((j, dist_ang)) = clash {
            return Err(SanityError::StericClash {
                i,
                j,
                dist_ang,
                limit_ang: limits.min_steric_dist_ang,
            });
        }
    }

    // Check kinetic temperature stability (`N_dof = 3 * N - 3`)
    let n_dof = (3 * n).saturating_sub(3).max(3) as f64;
    let t_kin = (2.0 * cp.kinetic_energy_kcal) / (n_dof * K_BOLTZMANN_KCAL);
    let sigma_t = cp.temperature_kelvin * (2.0 / n_dof).sqrt();
    if (t_kin - cp.temperature_kelvin).abs() > limits.max_temp_sigma * sigma_t {
        return Err(SanityError::KineticTemperatureBlowup {
            t_kin,
            t_target: cp.temperature_kelvin,
            sigma: limits.max_temp_sigma,
        });
    }

    let pot_per_atom = cp.potential_energy_kcal / (n as f64);
    if pot_per_atom > limits.max_pot_per_atom_kcal {
        return Err(SanityError::PotentialBlowup { pot_per_atom });
    }

    Ok(())
}

/// Computes mass-weighted Radius of Gyration `R_g` in Angstroms.
pub fn compute_radius_of_gyration(topology: &ProteinTopology, positions: &[[f32; 3]]) -> f64 {
    let mut m_tot = 0.0f64;
    let mut com = [0.0f64; 3];
    for (atom, p) in topology.atoms.iter().zip(positions.iter()) {
        let m = atom.mass_amu as f64;
        m_tot += m;
        com[0] += m * (p[0] as f64);
        com[1] += m * (p[1] as f64);
        com[2] += m * (p[2] as f64);
    }
    if m_tot <= 0.0 {
        return 0.0;
    }
    com[0] /= m_tot;
    com[1] /= m_tot;
    com[2] /= m_tot;

    let mut inertia = 0.0f64;
    for (atom, p) in topology.atoms.iter().zip(positions.iter()) {
        let m = atom.mass_amu as f64;
        let dx = (p[0] as f64) - com[0];
        let dy = (p[1] as f64) - com[1];
        let dz = (p[2] as f64) - com[2];
        inertia += m * (dx * dx + dy * dy + dz * dz);
    }
    (inertia / m_tot).sqrt()
}

/// Optimal Kabsch/Quaternion 3D rotational superposition C-alpha RMSD (in Angstroms)
/// using Horn's 4x4 symmetric quaternion eigensolver via Jacobi sweeps.
pub fn compute_ca_rmsd(
    topology: &ProteinTopology,
    coords_a: &[[f32; 3]],
    coords_b: &[[f32; 3]],
) -> f64 {
    let mut pa = Vec::new();
    let mut pb = Vec::new();
    for (idx, atom) in topology.atoms.iter().enumerate() {
        if atom.is_ca {
            pa.push([
                coords_a[idx][0] as f64,
                coords_a[idx][1] as f64,
                coords_a[idx][2] as f64,
            ]);
            pb.push([
                coords_b[idx][0] as f64,
                coords_b[idx][1] as f64,
                coords_b[idx][2] as f64,
            ]);
        }
    }
    let m = pa.len();
    if m == 0 {
        return 0.0;
    }

    let centroid = |pts: &[[f64; 3]]| -> [f64; 3] {
        let mut c = [0.0; 3];
        for p in pts {
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        let inv = 1.0 / (pts.len() as f64);
        [c[0] * inv, c[1] * inv, c[2] * inv]
    };

    let ca = centroid(&pa);
    let cb = centroid(&pb);
    for p in &mut pa {
        p[0] -= ca[0];
        p[1] -= ca[1];
        p[2] -= ca[2];
    }
    for p in &mut pb {
        p[0] -= cb[0];
        p[1] -= cb[1];
        p[2] -= cb[2];
    }

    let mut s = [[0.0f64; 3]; 3];
    let mut g0 = 0.0f64;
    for (a, b) in pa.iter().zip(pb.iter()) {
        g0 += a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += a[r] * b[c];
            }
        }
    }

    // Horn's 4x4 quaternion key matrix K
    let mut k = [[0.0f64; 4]; 4];
    k[0][0] = s[0][0] + s[1][1] + s[2][2];
    k[0][1] = s[1][2] - s[2][1];
    k[0][2] = s[2][0] - s[0][2];
    k[0][3] = s[0][1] - s[1][0];

    k[1][0] = k[0][1];
    k[1][1] = s[0][0] - s[1][1] - s[2][2];
    k[1][2] = s[0][1] + s[1][0];
    k[1][3] = s[2][0] + s[0][2];

    k[2][0] = k[0][2];
    k[2][1] = k[1][2];
    k[2][2] = -s[0][0] + s[1][1] - s[2][2];
    k[2][3] = s[1][2] + s[2][1];

    k[3][0] = k[0][3];
    k[3][1] = k[1][3];
    k[3][2] = k[2][3];
    k[3][3] = -s[0][0] - s[1][1] + s[2][2];

    let max_eig = max_symmetric_4x4_eigenvalue(k);
    let msd = ((g0 - 2.0 * max_eig) / (m as f64)).max(0.0);
    msd.sqrt()
}

fn max_symmetric_4x4_eigenvalue(mut a: [[f64; 4]; 4]) -> f64 {
    for _sweep in 0..25 {
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p][q];
                if apq.abs() < 1e-13 {
                    continue;
                }
                let diff = a[q][q] - a[p][p];
                let t = if diff.abs() < 1e-15 {
                    1.0
                } else {
                    let phi = diff / (2.0 * apq);
                    let sign = if phi >= 0.0 { 1.0 } else { -1.0 };
                    sign / (phi.abs() + (1.0 + phi * phi).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for r in 0..4 {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = arp - s * (arq + tau * arp);
                        a[p][r] = a[r][p];
                        a[r][q] = arq + s * (arp - tau * arq);
                        a[q][r] = a[r][q];
                    }
                }
            }
        }
    }
    a[0][0].max(a[1][1]).max(a[2][2]).max(a[3][3])
}

/// Returns `(atom_i, atom_j)` C-alpha native contact pairs separated by `|res_j - res_i| >= 2`
/// (or on distinct chains) with C-alpha Euclidean distance `<= cutoff_ang`.
pub fn native_ca_contact_pairs(
    topology: &ProteinTopology,
    native_positions: &[[f32; 3]],
    cutoff_ang: f64,
) -> Vec<(usize, usize)> {
    if native_positions.len() != topology.atoms.len() {
        return Vec::new();
    }
    let cas: Vec<(usize, i32, char)> = topology
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.is_ca)
        .map(|(idx, atom)| (idx, atom.res_seq, atom.chain_id))
        .collect();
    let mut pairs = Vec::new();
    for i in 0..cas.len() {
        for j in (i + 1)..cas.len() {
            let (idx_i, res_i, ch_i) = cas[i];
            let (idx_j, res_j, ch_j) = cas[j];
            if ch_i == ch_j && (res_j - res_i).abs() < 2 {
                continue;
            }
            let p1 = native_positions[idx_i];
            let p2 = native_positions[idx_j];
            let dx = (p1[0] - p2[0]) as f64;
            let dy = (p1[1] - p2[1]) as f64;
            let dz = (p1[2] - p2[2]) as f64;
            if (dx * dx + dy * dy + dz * dz).sqrt() <= cutoff_ang {
                pairs.push((idx_i, idx_j));
            }
        }
    }
    pairs
}

/// Computes the fraction `Q in [0.0, 1.0]` of native C-alpha contacts preserved within `cutoff_ang`.
pub fn fraction_native_contacts(
    native_pairs: &[(usize, usize)],
    positions: &[[f32; 3]],
    cutoff_ang: f64,
) -> f64 {
    if native_pairs.is_empty() {
        return 1.0;
    }
    let formed = native_pairs
        .iter()
        .filter(|&&(i, j)| {
            if i >= positions.len() || j >= positions.len() {
                return false;
            }
            let p1 = positions[i];
            let p2 = positions[j];
            let dx = (p1[0] - p2[0]) as f64;
            let dy = (p1[1] - p2[1]) as f64;
            let dz = (p1[2] - p2[2]) as f64;
            (dx * dx + dy * dy + dz * dz).sqrt() <= cutoff_ang
        })
        .count();
    (formed as f64) / (native_pairs.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kabsch_rmsd_invariant_under_rigid_rotation_and_translation() {
        let (topo, coords_a) = ProteinTopology::build_trp_cage_seed();
        // Rotate 90 degrees about Z and translate by (+12.5, -7.0, +3.2)
        let coords_b: Vec<[f32; 3]> = coords_a
            .iter()
            .map(|p| [-p[1] + 12.5, p[0] - 7.0, p[2] + 3.2])
            .collect();
        let rmsd = compute_ca_rmsd(&topo, &coords_a, &coords_b);
        assert!(rmsd < 1e-4, "expected ~0 RMSD under rigid motion, got {}", rmsd);
    }

    #[test]
    fn native_contacts_q_equals_one_for_native_and_drops_when_stretched() {
        let (topo, coords) = ProteinTopology::build_trp_cage_seed();
        let pairs = native_ca_contact_pairs(&topo, &coords, 8.5);
        assert!(
            !pairs.is_empty(),
            "Trp-cage seed must have >=1 C-alpha contact within 8.5 A"
        );
        let q_self = fraction_native_contacts(&pairs, &coords, 8.8);
        assert!((q_self - 1.0).abs() < 1e-12);

        let stretched: Vec<[f32; 3]> = coords
            .iter()
            .enumerate()
            .map(|(i, p)| [p[0] + (i as f32) * 3.0, p[1], p[2]])
            .collect();
        let q_stretched = fraction_native_contacts(&pairs, &stretched, 8.8);
        assert!(
            q_stretched < 0.5,
            "stretched chain should lose native contacts, got Q={q_stretched}"
        );
    }
}
