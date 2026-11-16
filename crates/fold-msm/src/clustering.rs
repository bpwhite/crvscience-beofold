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

use fold_physics::{compute_ca_rmsd, PhysicsError, ProteinTopology};
use fold_proto::{ProjectManifest, WorkUnitId, WorkUnitStatus};
use fold_storage::{FoldStore, StorageError};
use rayon::prelude::*;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MsmError {
    #[error("storage access error: {0}")]
    Storage(#[from] StorageError),
    #[error("physics/PDB export error: {0}")]
    Physics(#[from] PhysicsError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("insufficient completed trajectory checkpoints in vault (found {found}, need at least {required})")]
    InsufficientFrames { found: usize, required: usize },
    #[error("zero valid Markov transitions found at lag={lag} (check completed generation depth or temperature filter)")]
    ZeroTransitions { lag: u32 },
}

/// Extracted phase-space conformation ordered along its `(Project, Run, Clone)` trajectory chain.
#[derive(Debug, Clone)]
pub struct ConformationalFrame {
    pub frame_idx: usize,
    pub wu_id: WorkUnitId,
    /// `0` for the starting `Gen 0` input seed of `(Run, Clone)`, and `g + 1` for completed `Gen g`.
    pub chain_step: u32,
    pub checkpoint_blake3: [u8; 32],
    pub temperature_kelvin: f64,
    pub potential_energy_kcal: f64,
    pub rg_angstrom: f64,
    pub sasa_angstrom2: f64,
    pub rmsd_to_seed_ang: f64,
    pub rmsd_to_lowest_energy_ang: f64,
    pub positions_angstrom: Vec<[f32; 3]>,
}

/// Output of Regular-Space + PAM `k`-Medoids C-alpha RMSD microstate partitioning.
#[derive(Debug, Clone)]
pub struct MicrostateClustering {
    pub num_microstates: usize,
    /// Frame index (`0 .. frames.len()`) acting as the sterically valid medoid center of each microstate.
    pub medoid_frame_indices: Vec<usize>,
    /// Microstate assignment (`0 .. num_microstates`) for each `ConformationalFrame`.
    pub frame_assignments: Vec<usize>,
    /// C-alpha RMSD (`Angstrom`) from each frame to its assigned medoid center.
    pub frame_to_medoid_rmsd: Vec<f64>,
    /// Number of trajectory frames assigned to each microstate.
    pub microstate_counts: Vec<usize>,
    /// Mean within-cluster C-alpha RMSD (`Angstrom`) per microstate.
    pub mean_cluster_radius_ang: Vec<f64>,
}

/// Loads all completed `(Run, Clone, Generation)` trajectory frames for `project_id` from a
/// `FoldStore` (`redb` + LZ4 blobs), including each active clone's `Generation 0` starting frame
/// (`chain_step = 0`) so `G` completed generations yield `G + 1` time points and `G` transitions.
pub fn extract_project_frames(
    store: &FoldStore,
    project_id: u32,
    max_temperature_kelvin: Option<f64>,
) -> Result<(ProjectManifest, ProteinTopology, Vec<ConformationalFrame>), MsmError> {
    let (manifest, topology) = store.load_project(project_id)?;
    let seed_cp = store.get_checkpoint(&manifest.initial_coords_blake3)?;
    let records = store.list_workunits(project_id)?;

    let mut seen_chain_points: HashSet<(u32, u32, u32)> = HashSet::new();
    let mut raw_frames: Vec<ConformationalFrame> = Vec::new();

    for rec in &records {
        let WorkUnitStatus::Completed {
            output_checkpoint_blake3,
        } = rec.status
        else {
            continue;
        };

        // Include starting `Gen 0` seed checkpoint (`chain_step = 0`) for each completed `(Run, Clone)`
        if rec.wu_id.generation == 0
            && seen_chain_points.insert((rec.wu_id.run, rec.wu_id.clone, 0))
        {
            if let Ok(in_cp) = store.get_checkpoint(&rec.input_checkpoint_blake3) {
                if max_temperature_kelvin
                    .map(|tmax| in_cp.temperature_kelvin <= tmax)
                    .unwrap_or(true)
                {
                    let rmsd_seed = compute_ca_rmsd(
                        &topology,
                        &seed_cp.positions_angstrom,
                        &in_cp.positions_angstrom,
                    );
                    raw_frames.push(ConformationalFrame {
                        frame_idx: 0,
                        wu_id: rec.wu_id,
                        chain_step: 0,
                        checkpoint_blake3: rec.input_checkpoint_blake3,
                        temperature_kelvin: in_cp.temperature_kelvin,
                        potential_energy_kcal: in_cp.potential_energy_kcal,
                        rg_angstrom: in_cp.rg_angstrom,
                        sasa_angstrom2: in_cp.sasa_angstrom2,
                        rmsd_to_seed_ang: rmsd_seed,
                        rmsd_to_lowest_energy_ang: 0.0,
                        positions_angstrom: in_cp.positions_angstrom,
                    });
                }
            }
        }

        let out_step = rec.wu_id.generation + 1;
        if seen_chain_points.insert((rec.wu_id.run, rec.wu_id.clone, out_step)) {
            let out_cp = store.get_checkpoint(&output_checkpoint_blake3)?;
            if max_temperature_kelvin
                .map(|tmax| out_cp.temperature_kelvin <= tmax)
                .unwrap_or(true)
            {
                let rmsd_seed = compute_ca_rmsd(
                    &topology,
                    &seed_cp.positions_angstrom,
                    &out_cp.positions_angstrom,
                );
                raw_frames.push(ConformationalFrame {
                    frame_idx: 0,
                    wu_id: rec.wu_id,
                    chain_step: out_step,
                    checkpoint_blake3: output_checkpoint_blake3,
                    temperature_kelvin: out_cp.temperature_kelvin,
                    potential_energy_kcal: out_cp.potential_energy_kcal,
                    rg_angstrom: out_cp.rg_angstrom,
                    sasa_angstrom2: out_cp.sasa_angstrom2,
                    rmsd_to_seed_ang: rmsd_seed,
                    rmsd_to_lowest_energy_ang: 0.0,
                    positions_angstrom: out_cp.positions_angstrom,
                });
            }
        }
    }

    if raw_frames.len() < 4 {
        return Err(MsmError::InsufficientFrames {
            found: raw_frames.len(),
            required: 4,
        });
    }

    // Sort deterministically by `(run, clone, chain_step)`
    raw_frames.sort_by_key(|f| (f.wu_id.run, f.wu_id.clone, f.chain_step));
    for (idx, f) in raw_frames.iter_mut().enumerate() {
        f.frame_idx = idx;
    }

    // Identify global lowest-potential-energy conformation and compute C-alpha RMSD to it
    let mut best_idx = 0usize;
    for i in 1..raw_frames.len() {
        if raw_frames[i].potential_energy_kcal < raw_frames[best_idx].potential_energy_kcal {
            best_idx = i;
        }
    }
    let best_coords = raw_frames[best_idx].positions_angstrom.clone();
    let rmsd_to_best: Vec<f64> = raw_frames
        .par_iter()
        .map(|f| compute_ca_rmsd(&topology, &best_coords, &f.positions_angstrom))
        .collect();
    for (f, r) in raw_frames.iter_mut().zip(rmsd_to_best.into_iter()) {
        f.rmsd_to_lowest_energy_ang = r;
    }

    Ok((manifest, topology, raw_frames))
}

/// Computes the symmetric `N x N` optimal Horn/Kabsch quaternion C-alpha RMSD matrix in parallel via `rayon`.
pub fn compute_pairwise_rmsd_matrix(
    topology: &ProteinTopology,
    frames: &[ConformationalFrame],
) -> Vec<Vec<f64>> {
    let n = frames.len();
    let upper_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0f64; n - i];
            for offset in 1..(n - i) {
                let j = i + offset;
                row[offset] = compute_ca_rmsd(
                    topology,
                    &frames[i].positions_angstrom,
                    &frames[j].positions_angstrom,
                );
            }
            row
        })
        .collect();

    let mut mat = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for offset in 1..(n - i) {
            let j = i + offset;
            let d = upper_rows[i][offset];
            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    mat
}

/// Partitions conformational space into `K` discrete microstates using deterministic Maximin
/// Regular-Space seeding anchored at the lowest-energy frame followed by `PAM` `k`-Medoids
/// refinement so every microstate representative is an actual clash-free molecular conformation.
pub fn cluster_conformations_kmedoids(
    frames: &[ConformationalFrame],
    rmsd_matrix: &[Vec<f64>],
    requested_microstates: usize,
    min_separation_ang: Option<f64>,
) -> MicrostateClustering {
    let n = frames.len();
    let target_k = requested_microstates.clamp(2, n.max(2));

    // Anchor Medoid 0 at the global lowest-potential-energy frame
    let mut best_e_idx = 0usize;
    for i in 1..n {
        if frames[i].potential_energy_kcal < frames[best_e_idx].potential_energy_kcal {
            best_e_idx = i;
        }
    }

    let mut medoids: Vec<usize> = vec![best_e_idx];
    let mut min_dist_to_medoid: Vec<f64> = rmsd_matrix[best_e_idx].clone();

    while medoids.len() < target_k {
        let mut farthest_idx = 0usize;
        let mut farthest_dist = -1.0f64;
        for (i, &d) in min_dist_to_medoid.iter().enumerate() {
            if !medoids.contains(&i) && d > farthest_dist {
                farthest_dist = d;
                farthest_idx = i;
            }
        }
        if farthest_dist <= 1e-9 {
            break;
        }
        if let Some(cutoff) = min_separation_ang {
            if farthest_dist < cutoff && medoids.len() >= 2 {
                break;
            }
        }
        medoids.push(farthest_idx);
        for i in 0..n {
            let d = rmsd_matrix[i][farthest_idx];
            if d < min_dist_to_medoid[i] {
                min_dist_to_medoid[i] = d;
            }
        }
    }

    let k = medoids.len();
    let mut assignments = vec![0usize; n];

    // Iterative PAM k-medoids refinement
    for _iter in 0..40 {
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_d = rmsd_matrix[i][medoids[0]];
            for (c, &m_idx) in medoids.iter().enumerate().skip(1) {
                let d = rmsd_matrix[i][m_idx];
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        // Recompute exact medoid within each non-empty Voronoi cell
        let mut medoid_shifted = false;
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            let mut best_m = medoids[c];
            let mut best_sum = f64::INFINITY;
            for &cand in &members {
                let sum_d: f64 = members.iter().map(|&other| rmsd_matrix[cand][other]).sum();
                if sum_d < best_sum - 1e-12 {
                    best_sum = sum_d;
                    best_m = cand;
                }
            }
            if best_m != medoids[c] {
                medoids[c] = best_m;
                medoid_shifted = true;
            }
        }

        if !changed && !medoid_shifted {
            break;
        }
    }

    let mut counts = vec![0usize; k];
    let mut sum_radii = vec![0.0f64; k];
    let mut frame_to_medoid = vec![0.0f64; n];

    for i in 0..n {
        let c = assignments[i];
        let d = rmsd_matrix[i][medoids[c]];
        frame_to_medoid[i] = d;
        counts[c] += 1;
        sum_radii[c] += d;
    }

    let mean_radii = (0..k)
        .map(|c| {
            if counts[c] > 0 {
                sum_radii[c] / (counts[c] as f64)
            } else {
                0.0
            }
        })
        .collect();

    MicrostateClustering {
        num_microstates: k,
        medoid_frame_indices: medoids,
        frame_assignments: assignments,
        frame_to_medoid_rmsd: frame_to_medoid,
        microstate_counts: counts,
        mean_cluster_radius_ang: mean_radii,
    }
}
