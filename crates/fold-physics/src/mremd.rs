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
use crate::integrator::{
    initialize_clone_checkpoint, minimize_steric_clashes, run_generation_chunk_with_gpu,
    LangevinConfig,
};
use crate::rng::ChaCha8Rng;
use crate::topology::ProteinTopology;
use crate::K_BOLTZMANN_KCAL;
use fold_proto::{ComputeTier, MremdEndpoint, NodeCapabilityProfile, TrajectoryCheckpoint, WorkUnitId};
use std::time::Instant;
use sysinfo::System;

/// Exact Multiplexed-Replica Exchange Molecular Dynamics (MREMD) Metropolis acceptance probability:
/// `P_accept = min(1.0, exp((beta_1 - beta_2) * (V_1 - V_2)))`
/// where `beta_k = 1 / (k_B * T_k)`.
/// Safely returns `0.0` when either temperature is non-positive/non-finite or either potential is non-finite.
#[inline]
pub fn metropolis_swap_probability(
    temp_1_kelvin: f64,
    pot_1_kcal: f64,
    temp_2_kelvin: f64,
    pot_2_kcal: f64,
) -> f64 {
    if !temp_1_kelvin.is_finite()
        || !temp_2_kelvin.is_finite()
        || temp_1_kelvin <= 0.0
        || temp_2_kelvin <= 0.0
        || !pot_1_kcal.is_finite()
        || !pot_2_kcal.is_finite()
    {
        return 0.0;
    }
    let beta_1 = 1.0 / (K_BOLTZMANN_KCAL * temp_1_kelvin);
    let beta_2 = 1.0 / (K_BOLTZMANN_KCAL * temp_2_kelvin);
    let delta = (beta_1 - beta_2) * (pot_1_kcal - pot_2_kcal);
    if !delta.is_finite() {
        return 0.0;
    }
    if delta >= 0.0 {
        1.0
    } else {
        delta.exp().clamp(0.0, 1.0)
    }
}

/// Applies Sugita-Okamoto (1999) canonical velocity rescaling `v' = v * sqrt(T_new / T_old)`
/// on an MREMD replica exchange from `t_old_kelvin` to `t_new_kelvin`.
/// Leaves velocities untouched and returns `false` if either temperature is `<= 0.0` or non-finite.
#[inline]
pub fn rescale_velocities_sugita_okamoto(
    velocities: &mut [[f32; 3]],
    t_old_kelvin: f64,
    t_new_kelvin: f64,
) -> bool {
    if !t_old_kelvin.is_finite()
        || !t_new_kelvin.is_finite()
        || t_old_kelvin <= 0.0
        || t_new_kelvin <= 0.0
    {
        return false;
    }
    if (t_new_kelvin - t_old_kelvin).abs() < 1.0e-12 {
        return true;
    }
    let scale = (t_new_kelvin / t_old_kelvin).sqrt();
    for v in velocities.iter_mut() {
        v[0] = ((v[0] as f64) * scale) as f32;
        v[1] = ((v[1] as f64) * scale) as f32;
        v[2] = ((v[2] as f64) * scale) as f32;
    }
    true
}

/// Evaluates an asynchronous MREMD swap between two completed generation endpoints on adjacent
/// temperature rungs. On swap acceptance, returns `(temp_2, temp_1)` as the swapped bath
/// temperatures to assign to each trajectory's `Generation G+1`.
pub fn try_mremd_swap(
    ep_a: &MremdEndpoint,
    ep_b: &MremdEndpoint,
    rng: &mut ChaCha8Rng,
) -> (bool, f64, f64) {
    let p_acc = metropolis_swap_probability(
        ep_a.temperature_kelvin,
        ep_a.potential_energy_kcal,
        ep_b.temperature_kelvin,
        ep_b.potential_energy_kcal,
    );
    if p_acc <= 0.0 {
        return (false, ep_a.temperature_kelvin, ep_b.temperature_kelvin);
    }
    let u = rng.next_f64_open01();
    if u < p_acc {
        (true, ep_b.temperature_kelvin, ep_a.temperature_kelvin)
    } else {
        (false, ep_a.temperature_kelvin, ep_b.temperature_kelvin)
    }
}

/// Evaluates an MREMD Metropolis swap directly between two `TrajectoryCheckpoint`s and, when
/// accepted, exchanges their bath temperatures and applies Sugita-Okamoto (1999) velocity
/// rescaling `sqrt(T_new / T_old)` (updating `kinetic_energy_kcal` accordingly).
pub fn try_mremd_swap_checkpoints(
    cp_a: &mut TrajectoryCheckpoint,
    cp_b: &mut TrajectoryCheckpoint,
    rng: &mut ChaCha8Rng,
) -> bool {
    let p_acc = metropolis_swap_probability(
        cp_a.temperature_kelvin,
        cp_a.potential_energy_kcal,
        cp_b.temperature_kelvin,
        cp_b.potential_energy_kcal,
    );
    if p_acc <= 0.0 {
        return false;
    }
    let u = rng.next_f64_open01();
    if u < p_acc {
        let ta = cp_a.temperature_kelvin;
        let tb = cp_b.temperature_kelvin;
        rescale_velocities_sugita_okamoto(&mut cp_a.velocities_ang_per_ps, ta, tb);
        rescale_velocities_sugita_okamoto(&mut cp_b.velocities_ang_per_ps, tb, ta);
        cp_a.temperature_kelvin = tb;
        cp_b.temperature_kelvin = ta;
        if cp_a.kinetic_energy_kcal.is_finite() {
            cp_a.kinetic_energy_kcal *= tb / ta;
        }
        if cp_b.kinetic_energy_kcal.is_finite() {
            cp_b.kinetic_energy_kcal *= ta / tb;
        }
        true
    } else {
        false
    }
}

/// Runs the startup Villin Headpiece (35-residue) micro-benchmark with automatic `wgpu` discrete
/// GPU probing (`ComputeTier::TierS`) and `rayon` CPU fallback (`TierA` / `TierB`).
pub fn benchmark_local_node(node_pubkey: [u8; 32], bench_steps: u64) -> NodeCapabilityProfile {
    benchmark_local_node_with_gpu(node_pubkey, bench_steps, true)
}

/// Runs the startup Villin Headpiece micro-benchmark with explicit `enable_gpu` control.
pub fn benchmark_local_node_with_gpu(
    node_pubkey: [u8; 32],
    bench_steps: u64,
    enable_gpu: bool,
) -> NodeCapabilityProfile {
    let mut sys = System::new_all();
    sys.refresh_memory();

    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or_else(|_| sys.cpus().len().max(1) as u32);
    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let hostname = System::host_name().unwrap_or_else(|| "beowulf-node".to_string());

    let (topo, mut coords) = ProteinTopology::build_villin_benchmark_seed();
    minimize_steric_clashes(&topo, &mut coords, 40);
    let cp0 = initialize_clone_checkpoint(
        &topo,
        &coords,
        WorkUnitId::new(0, 0, 0, 0),
        300.0,
        b"villin-bench",
    );

    let gpu_engine = if enable_gpu {
        GpuForceEngine::try_new(&topo).ok()
    } else {
        None
    };

    let has_discrete_gpu = gpu_engine
        .as_ref()
        .map(|g| g.adapter_summary().is_discrete_gpu)
        .unwrap_or(false);

    let cfg = LangevinConfig {
        dt_fs: 2.0,
        friction_per_ps: 1.0,
        temperature_kelvin: 300.0,
        steps: bench_steps.max(250),
        sample_stride: 250,
    };

    let t0 = Instant::now();
    let _ = run_generation_chunk_with_gpu(
        &topo,
        gpu_engine.as_ref(),
        &cp0,
        WorkUnitId::new(0, 0, 0, 1),
        &cfg,
    );
    let elapsed_sec = t0.elapsed().as_secs_f64().max(1e-6);

    // 1 step at dt = 2.0 fs = 2e-6 ns; 86,400 sec/day
    let simulated_ns = (cfg.steps as f64) * cfg.dt_fs * 1.0e-6;
    let villin_ns_per_day = simulated_ns * (86_400.0 / elapsed_sec);

    let tier = if has_discrete_gpu {
        ComputeTier::TierS
    } else if logical_cores >= 8 && villin_ns_per_day >= 5.0 {
        ComputeTier::TierA
    } else {
        ComputeTier::TierB
    };

    NodeCapabilityProfile {
        node_pubkey,
        hostname,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cores,
        total_ram_mb,
        villin_ns_per_day,
        tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::ProteinTopology;
    use crate::integrator::{initialize_clone_checkpoint, minimize_steric_clashes};

    #[test]
    fn mremd_metropolis_favors_swapping_lower_energy_into_colder_bath() {
        let p_favorable = metropolis_swap_probability(290.0, -40.0, 380.0, -65.0);
        assert_eq!(p_favorable, 1.0);

        let p_uphill = metropolis_swap_probability(290.0, -65.0, 380.0, -40.0);
        assert!(p_uphill > 0.0 && p_uphill < 1.0);
    }

    #[test]
    fn mremd_metropolis_guards_nonpositive_temp_and_nan_potentials() {
        assert_eq!(metropolis_swap_probability(0.0, -50.0, 300.0, -40.0), 0.0);
        assert_eq!(metropolis_swap_probability(300.0, -50.0, -10.0, -40.0), 0.0);
        assert_eq!(
            metropolis_swap_probability(f64::NAN, -50.0, 300.0, -40.0),
            0.0
        );
        assert_eq!(
            metropolis_swap_probability(300.0, f64::NAN, 320.0, -40.0),
            0.0
        );
        assert_eq!(
            metropolis_swap_probability(300.0, -50.0, 320.0, f64::INFINITY),
            0.0
        );

        let ep_bad = MremdEndpoint {
            wu_id: WorkUnitId::new(1, 0, 0, 0),
            rung_index: 0,
            temperature_kelvin: 0.0,
            potential_energy_kcal: f64::NAN,
            output_checkpoint_blake3: [0u8; 32],
        };
        let ep_ok = MremdEndpoint {
            wu_id: WorkUnitId::new(1, 1, 0, 0),
            rung_index: 1,
            temperature_kelvin: 320.0,
            potential_energy_kcal: -50.0,
            output_checkpoint_blake3: [0u8; 32],
        };
        let mut rng = ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 0, 0, 0), b"guard-test");
        let (swapped, ta, tb) = try_mremd_swap(&ep_bad, &ep_ok, &mut rng);
        assert!(!swapped);
        assert_eq!(ta, 0.0);
        assert_eq!(tb, 320.0);
    }

    #[test]
    fn mremd_sugita_okamoto_rescales_velocities_by_sqrt_temp_ratio() {
        let mut v = [[1.0f32, -2.0, 0.5], [0.25, 0.75, -1.5]];
        let orig = v;
        let t_old = 285.0_f64;
        let t_new = 405.0_f64;
        let expected_scale = (t_new / t_old).sqrt() as f32;

        assert!(rescale_velocities_sugita_okamoto(&mut v, t_old, t_new));
        for (a, b) in v.iter().zip(orig.iter()) {
            for d in 0..3 {
                assert!((a[d] - b[d] * expected_scale).abs() < 1.0e-5);
            }
        }
        // Invalid temperature guard leaves velocities untouched
        let before_bad = v;
        assert!(!rescale_velocities_sugita_okamoto(&mut v, 0.0, 300.0));
        assert!(!rescale_velocities_sugita_okamoto(&mut v, 300.0, f64::NAN));
        assert_eq!(v, before_bad);

        // Full checkpoint swap rescales both trajectories when accepted
        let (topo, mut coords) = ProteinTopology::build_polypeptide_from_sequence("WK", "WK");
        minimize_steric_clashes(&topo, &mut coords, 20);
        let mut cp_cold =
            initialize_clone_checkpoint(&topo, &coords, WorkUnitId::new(1, 0, 0, 0), 280.0, b"so");
        let mut cp_hot =
            initialize_clone_checkpoint(&topo, &coords, WorkUnitId::new(1, 1, 0, 0), 420.0, b"so");
        // Guarantee favorable swap: give hot conformation lower potential than cold
        cp_hot.potential_energy_kcal = -100.0;
        cp_cold.potential_energy_kcal = -20.0;
        let v_cold_before = cp_cold.velocities_ang_per_ps.clone();
        let v_hot_before = cp_hot.velocities_ang_per_ps.clone();
        let ke_cold_before = cp_cold.kinetic_energy_kcal;
        let ke_hot_before = cp_hot.kinetic_energy_kcal;

        let mut rng = ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 0, 0, 0), b"so-swap");
        assert!(try_mremd_swap_checkpoints(&mut cp_cold, &mut cp_hot, &mut rng));
        assert_eq!(cp_cold.temperature_kelvin, 420.0);
        assert_eq!(cp_hot.temperature_kelvin, 280.0);

        let scale_up = (420.0_f64 / 280.0).sqrt() as f32;
        let scale_dn = (280.0_f64 / 420.0).sqrt() as f32;
        assert!((cp_cold.velocities_ang_per_ps[0][0] - v_cold_before[0][0] * scale_up).abs() < 1e-4);
        assert!((cp_hot.velocities_ang_per_ps[0][0] - v_hot_before[0][0] * scale_dn).abs() < 1e-4);
        assert!((cp_cold.kinetic_energy_kcal - ke_cold_before * (420.0 / 280.0)).abs() < 1e-4);
        assert!((cp_hot.kinetic_energy_kcal - ke_hot_before * (280.0 / 420.0)).abs() < 1e-4);
    }
}
