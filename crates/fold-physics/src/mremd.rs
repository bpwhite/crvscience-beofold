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
use fold_proto::{ComputeTier, MremdEndpoint, NodeCapabilityProfile, WorkUnitId};
use std::time::Instant;
use sysinfo::System;

/// Exact Multiplexed-Replica Exchange Molecular Dynamics (MREMD) Metropolis acceptance probability:
/// `P_accept = min(1.0, exp((beta_1 - beta_2) * (V_1 - V_2)))`
/// where `beta_k = 1 / (k_B * T_k)`.
#[inline]
pub fn metropolis_swap_probability(
    temp_1_kelvin: f64,
    pot_1_kcal: f64,
    temp_2_kelvin: f64,
    pot_2_kcal: f64,
) -> f64 {
    let beta_1 = 1.0 / (K_BOLTZMANN_KCAL * temp_1_kelvin);
    let beta_2 = 1.0 / (K_BOLTZMANN_KCAL * temp_2_kelvin);
    let delta = (beta_1 - beta_2) * (pot_1_kcal - pot_2_kcal);
    if delta >= 0.0 {
        1.0
    } else {
        delta.exp()
    }
}

/// Evaluates an asynchronous MREMD swap between two completed generation endpoints on adjacent
/// temperature rungs. On swap acceptance, returns `(temp_2, temp_1)` as the swapped bath
/// temperatures to assign to each trajectory's `Generation G+1` (zero coordinate transfer needed).
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
    let u = rng.next_f64_open01();
    if u < p_acc {
        (true, ep_b.temperature_kelvin, ep_a.temperature_kelvin)
    } else {
        (false, ep_a.temperature_kelvin, ep_b.temperature_kelvin)
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

    #[test]
    fn mremd_metropolis_favors_swapping_lower_energy_into_colder_bath() {
        let p_favorable = metropolis_swap_probability(290.0, -40.0, 380.0, -65.0);
        assert_eq!(p_favorable, 1.0);

        let p_uphill = metropolis_swap_probability(290.0, -65.0, 380.0, -40.0);
        assert!(p_uphill > 0.0 && p_uphill < 1.0);
    }
}
