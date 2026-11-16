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

use crate::forcefield::{
    compute_sasa_angstrom2, evaluate_forces_and_energy, evaluate_forces_and_energy_with_gpu,
};
use crate::gpu::GpuForceEngine;
use crate::rng::ChaCha8Rng;
use crate::sanity::compute_radius_of_gyration;
use crate::topology::ProteinTopology;
use crate::{KCAL_TO_AMU_A2_PS2, K_BOLTZMANN_KCAL};
use fold_proto::{TrajectoryCheckpoint, WorkUnitId};

/// Integrator configuration for a single PRCG Generation chunk.
#[derive(Debug, Clone, Copy)]
pub struct LangevinConfig {
    /// Integration timestep in femtoseconds (`1.0` to `2.0 fs` typical).
    pub dt_fs: f64,
    /// Langevin bath friction `gamma` in `ps^-1` (`0.0` selects pure symplectic NVE velocity-Verlet).
    pub friction_per_ps: f64,
    /// Target bath temperature in Kelvin.
    pub temperature_kelvin: f64,
    /// Number of timesteps to execute in this Generation chunk.
    pub steps: u64,
    /// Interval (in steps) at which running mean/min energy statistics are sampled.
    pub sample_stride: u64,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            dt_fs: 1.5,
            friction_per_ps: 1.0,
            temperature_kelvin: 300.0,
            steps: 50_000,
            sample_stride: 100,
        }
    }
}

/// Summary energetics produced over a completed Generation chunk.
#[derive(Debug, Clone)]
pub struct GenerationSummary {
    pub checkpoint: TrajectoryCheckpoint,
    pub mean_potential_kcal: f64,
    pub min_potential_kcal: f64,
    pub min_energy_positions: Vec<[f32; 3]>,
}

/// Relaxes high-energy steric overlaps in raw crystallographic or seed coordinates using
/// capped-step steepest descent with backtracking Armijo line search.
pub fn minimize_steric_clashes(
    topology: &ProteinTopology,
    positions: &mut [[f32; 3]],
    max_iters: usize,
) -> f64 {
    let mut forces = vec![[0.0f32; 3]; topology.atoms.len()];
    let mut current_e = evaluate_forces_and_energy(topology, positions, &mut forces).total_kcal;
    let mut step_scale = 0.002f32;
    let mut trial_pos = positions.to_vec();
    let mut trial_forces = vec![[0.0f32; 3]; topology.atoms.len()];

    for _ in 0..max_iters {
        let mut max_f = 0.0f32;
        for f in &forces {
            let norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            if norm > max_f {
                max_f = norm;
            }
        }
        if max_f < 0.5 {
            break;
        }

        let inv_max = 1.0 / max_f.max(1.0);
        for (tp, (p, f)) in trial_pos.iter_mut().zip(positions.iter().zip(forces.iter())) {
            let dx = (f[0] * inv_max * step_scale).clamp(-0.05, 0.05);
            let dy = (f[1] * inv_max * step_scale).clamp(-0.05, 0.05);
            let dz = (f[2] * inv_max * step_scale).clamp(-0.05, 0.05);
            *tp = [p[0] + dx, p[1] + dy, p[2] + dz];
        }

        let trial_e =
            evaluate_forces_and_energy(topology, &trial_pos, &mut trial_forces).total_kcal;
        if trial_e.is_finite() && trial_e < current_e {
            positions.copy_from_slice(&trial_pos);
            forces.copy_from_slice(&trial_forces);
            current_e = trial_e;
            step_scale = (step_scale * 1.15).min(0.025);
        } else {
            step_scale *= 0.5;
            if step_scale < 1e-6 {
                break;
            }
        }
    }
    current_e
}

/// Creates a `Generation 0` starting checkpoint for a `(Project, Run, Clone)` trajectory with
/// zero-net-momentum Maxwell-Boltzmann velocities drawn from `ChaCha8Rng`.
pub fn initialize_clone_checkpoint(
    topology: &ProteinTopology,
    minimized_positions: &[[f32; 3]],
    wu_id: WorkUnitId,
    temperature_kelvin: f64,
    campaign_salt: &[u8],
) -> TrajectoryCheckpoint {
    let mut rng = ChaCha8Rng::from_prcg_seed(wu_id, campaign_salt);
    let n = topology.atoms.len();
    let mut velocities = vec![[0.0f32; 3]; n];

    let mut total_mass = 0.0f64;
    let mut com_momentum = [0.0f64; 3];

    for (i, atom) in topology.atoms.iter().enumerate() {
        let m = atom.mass_amu as f64;
        let sigma_v = (KCAL_TO_AMU_A2_PS2 * K_BOLTZMANN_KCAL * temperature_kelvin / m).sqrt();
        let g = rng.next_gaussian_vec3();
        let vx = sigma_v * g[0];
        let vy = sigma_v * g[1];
        let vz = sigma_v * g[2];
        velocities[i] = [vx as f32, vy as f32, vz as f32];
        total_mass += m;
        com_momentum[0] += m * vx;
        com_momentum[1] += m * vy;
        com_momentum[2] += m * vz;
    }

    if total_mass > 0.0 {
        let v_com = [
            (com_momentum[0] / total_mass) as f32,
            (com_momentum[1] / total_mass) as f32,
            (com_momentum[2] / total_mass) as f32,
        ];
        for v in &mut velocities {
            v[0] -= v_com[0];
            v[1] -= v_com[1];
            v[2] -= v_com[2];
        }
    }

    let mut forces = vec![[0.0f32; 3]; n];
    let pot = evaluate_forces_and_energy(topology, minimized_positions, &mut forces).total_kcal;
    let kin = compute_kinetic_energy_kcal(topology, &velocities);
    let rg = compute_radius_of_gyration(topology, minimized_positions);
    let sasa = compute_sasa_angstrom2(topology, minimized_positions);
    let (rng_state, rng_counter) = rng.to_raw_parts();

    TrajectoryCheckpoint {
        wu_id,
        cumulative_steps: 0,
        temperature_kelvin,
        rng_state,
        rng_counter,
        positions_angstrom: minimized_positions.to_vec(),
        velocities_ang_per_ps: velocities,
        potential_energy_kcal: pot,
        kinetic_energy_kcal: kin,
        rg_angstrom: rg,
        sasa_angstrom2: sasa,
    }
}

/// Executes a self-contained PRCG Generation chunk on the CPU (`rayon` Morton neighbor lists).
pub fn run_generation_chunk(
    topology: &ProteinTopology,
    input: &TrajectoryCheckpoint,
    output_wu_id: WorkUnitId,
    config: &LangevinConfig,
) -> GenerationSummary {
    run_generation_chunk_with_gpu(topology, None, input, output_wu_id, config)
}

/// Executes a self-contained PRCG Generation chunk using the symmetric Leimkuhler-Matthews
/// Langevin `BAOAB` splitting, dispatching non-bonded forces to `wgpu` when `gpu` is `Some`.
pub fn run_generation_chunk_with_gpu(
    topology: &ProteinTopology,
    gpu: Option<&GpuForceEngine>,
    input: &TrajectoryCheckpoint,
    output_wu_id: WorkUnitId,
    config: &LangevinConfig,
) -> GenerationSummary {
    let n = topology.atoms.len();
    let dt_ps = config.dt_fs * 1.0e-3;
    let half_dt = 0.5 * dt_ps;

    let mut pos = input.positions_angstrom.clone();
    let mut vel = input.velocities_ang_per_ps.clone();
    let mut rng = ChaCha8Rng::from_raw_parts(input.rng_state, input.rng_counter);

    let use_thermostat = config.friction_per_ps > 0.0;
    let c1 = (-config.friction_per_ps * dt_ps).exp();
    let c2 = (1.0 - c1 * c1).max(0.0).sqrt();

    let mut forces = vec![[0.0f32; 3]; n];
    let mut energy =
        evaluate_forces_and_energy_with_gpu(topology, gpu, &pos, &mut forces).total_kcal;

    let mut min_pot = energy;
    let mut min_pos = pos.clone();
    let mut sum_pot = 0.0f64;
    let mut sample_count = 0u64;
    let stride = config.sample_stride.max(1);

    for step in 1..=config.steps {
        // B: Half-step velocity kick from F(t)
        for i in 0..n {
            let inv_m = KCAL_TO_AMU_A2_PS2 / (topology.atoms[i].mass_amu as f64);
            let kick = half_dt * inv_m;
            vel[i][0] = ((vel[i][0] as f64) + kick * (forces[i][0] as f64)) as f32;
            vel[i][1] = ((vel[i][1] as f64) + kick * (forces[i][1] as f64)) as f32;
            vel[i][2] = ((vel[i][2] as f64) + kick * (forces[i][2] as f64)) as f32;
        }

        if use_thermostat {
            // A: First half-step position drift
            for i in 0..n {
                pos[i][0] = ((pos[i][0] as f64) + half_dt * (vel[i][0] as f64)) as f32;
                pos[i][1] = ((pos[i][1] as f64) + half_dt * (vel[i][1] as f64)) as f32;
                pos[i][2] = ((pos[i][2] as f64) + half_dt * (vel[i][2] as f64)) as f32;
            }

            // O: Exact Ornstein-Uhlenbeck stochastic velocity update
            for i in 0..n {
                let m = topology.atoms[i].mass_amu as f64;
                let sigma = c2
                    * (KCAL_TO_AMU_A2_PS2 * K_BOLTZMANN_KCAL * config.temperature_kelvin / m)
                        .sqrt();
                let g = rng.next_gaussian_vec3();
                vel[i][0] = (c1 * (vel[i][0] as f64) + sigma * g[0]) as f32;
                vel[i][1] = (c1 * (vel[i][1] as f64) + sigma * g[1]) as f32;
                vel[i][2] = (c1 * (vel[i][2] as f64) + sigma * g[2]) as f32;
            }

            // A: Second half-step position drift
            for i in 0..n {
                pos[i][0] = ((pos[i][0] as f64) + half_dt * (vel[i][0] as f64)) as f32;
                pos[i][1] = ((pos[i][1] as f64) + half_dt * (vel[i][1] as f64)) as f32;
                pos[i][2] = ((pos[i][2] as f64) + half_dt * (vel[i][2] as f64)) as f32;
            }
        } else {
            // Pure symplectic velocity-Verlet full drift (`dt`) for NVE conservation checks
            for i in 0..n {
                pos[i][0] = ((pos[i][0] as f64) + dt_ps * (vel[i][0] as f64)) as f32;
                pos[i][1] = ((pos[i][1] as f64) + dt_ps * (vel[i][1] as f64)) as f32;
                pos[i][2] = ((pos[i][2] as f64) + dt_ps * (vel[i][2] as f64)) as f32;
            }
        }

        // Evaluate new forces F(t + dt)
        energy = evaluate_forces_and_energy_with_gpu(topology, gpu, &pos, &mut forces).total_kcal;

        // B: Second half-step velocity kick from F(t + dt)
        for i in 0..n {
            let inv_m = KCAL_TO_AMU_A2_PS2 / (topology.atoms[i].mass_amu as f64);
            let kick = half_dt * inv_m;
            vel[i][0] = ((vel[i][0] as f64) + kick * (forces[i][0] as f64)) as f32;
            vel[i][1] = ((vel[i][1] as f64) + kick * (forces[i][1] as f64)) as f32;
            vel[i][2] = ((vel[i][2] as f64) + kick * (forces[i][2] as f64)) as f32;
        }

        if step % stride == 0 || step == config.steps {
            sum_pot += energy;
            sample_count += 1;
            if energy < min_pot {
                min_pot = energy;
                min_pos.copy_from_slice(&pos);
            }
        }
    }

    let kin = compute_kinetic_energy_kcal(topology, &vel);
    let rg = compute_radius_of_gyration(topology, &pos);
    let sasa = compute_sasa_angstrom2(topology, &pos);
    let (rng_state, rng_counter) = rng.to_raw_parts();
    let mean_pot = if sample_count > 0 {
        sum_pot / (sample_count as f64)
    } else {
        energy
    };

    GenerationSummary {
        checkpoint: TrajectoryCheckpoint {
            wu_id: output_wu_id,
            cumulative_steps: input.cumulative_steps + config.steps,
            temperature_kelvin: config.temperature_kelvin,
            rng_state,
            rng_counter,
            positions_angstrom: pos,
            velocities_ang_per_ps: vel,
            potential_energy_kcal: energy,
            kinetic_energy_kcal: kin,
            rg_angstrom: rg,
            sasa_angstrom2: sasa,
        },
        mean_potential_kcal: mean_pot,
        min_potential_kcal: min_pot,
        min_energy_positions: min_pos,
    }
}

pub fn compute_kinetic_energy_kcal(topology: &ProteinTopology, velocities: &[[f32; 3]]) -> f64 {
    let mut mv2 = 0.0f64;
    for (atom, v) in topology.atoms.iter().zip(velocities.iter()) {
        let vx = v[0] as f64;
        let vy = v[1] as f64;
        let vz = v[2] as f64;
        mv2 += (atom.mass_amu as f64) * (vx * vx + vy * vy + vz * vz);
    }
    0.5 * mv2 / KCAL_TO_AMU_A2_PS2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sanity::{verify_checkpoint_sanity, SanityThresholds};

    #[test]
    fn nve_energy_conservation_and_exact_generation_restart_continuity() {
        let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
        minimize_steric_clashes(&topo, &mut coords, 120);

        let seed_wu = WorkUnitId::new(1, 0, 0, 0);
        let gen0 = initialize_clone_checkpoint(&topo, &coords, seed_wu, 300.0, b"unit-test-salt");

        let nve_cfg = LangevinConfig {
            dt_fs: 0.5,
            friction_per_ps: 0.0,
            temperature_kelvin: 300.0,
            steps: 200,
            sample_stride: 20,
        };
        let e0_total = gen0.potential_energy_kcal + gen0.kinetic_energy_kcal;
        let nve_out = run_generation_chunk(&topo, &gen0, WorkUnitId::new(1, 0, 0, 1), &nve_cfg);
        let e1_total =
            nve_out.checkpoint.potential_energy_kcal + nve_out.checkpoint.kinetic_energy_kcal;
        let drift_per_atom = (e1_total - e0_total).abs() / (topo.atoms.len() as f64);
        assert!(
            drift_per_atom < 0.02,
            "excessive NVE energy drift per atom: {}",
            drift_per_atom
        );

        let full_cfg = LangevinConfig {
            dt_fs: 1.0,
            friction_per_ps: 1.0,
            temperature_kelvin: 310.0,
            steps: 100,
            sample_stride: 10,
        };
        let continuous = run_generation_chunk(&topo, &gen0, WorkUnitId::new(1, 0, 0, 2), &full_cfg);

        let half_cfg = LangevinConfig {
            steps: 50,
            ..full_cfg
        };
        let chunk_a = run_generation_chunk(&topo, &gen0, WorkUnitId::new(1, 0, 0, 1), &half_cfg);
        let wire_bytes = chunk_a.checkpoint.to_compressed_frame().unwrap();
        let restored_a = TrajectoryCheckpoint::from_compressed_frame(&wire_bytes).unwrap();
        let chunk_b =
            run_generation_chunk(&topo, &restored_a, WorkUnitId::new(1, 0, 0, 2), &half_cfg);

        assert_eq!(
            continuous.checkpoint.positions_angstrom,
            chunk_b.checkpoint.positions_angstrom
        );
        assert_eq!(
            continuous.checkpoint.velocities_ang_per_ps,
            chunk_b.checkpoint.velocities_ang_per_ps
        );
        assert!(
            verify_checkpoint_sanity(&topo, &chunk_b.checkpoint, &SanityThresholds::default())
                .is_ok()
        );
    }
}
