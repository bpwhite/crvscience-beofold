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

pub mod forcefield;
pub mod gpu;
pub mod integrator;
pub mod morton;
pub mod mremd;
pub mod rng;
pub mod sanity;
pub mod topology;

pub use forcefield::{
    compute_sasa_angstrom2, eef1_solvation_weight, evaluate_forces_and_energy,
    evaluate_forces_and_energy_with_gpu, EnergyBreakdown, GB_EPSILON_IN, GB_EPSILON_OUT,
};
pub use gpu::{probe_gpu_adapter, GpuAdapterSummary, GpuComputeError, GpuForceEngine};
pub use integrator::{
    compute_kinetic_energy_kcal, initialize_clone_checkpoint, minimize_steric_clashes,
    run_generation_chunk, run_generation_chunk_with_gpu, GenerationSummary, LangevinConfig,
};
pub use morton::{morton_encode_3d, SpatialNeighborGrid};
pub use mremd::{
    benchmark_local_node, benchmark_local_node_with_gpu, metropolis_swap_probability,
    rescale_velocities_sugita_okamoto, try_mremd_swap, try_mremd_swap_checkpoints,
};
pub use rng::ChaCha8Rng;
pub use sanity::{
    compute_ca_rmsd, compute_radius_of_gyration, fraction_native_contacts, native_ca_contact_pairs,
    verify_checkpoint_sanity, SanityError, SanityThresholds,
};
pub use topology::{
    sanitize_amino_acid_sequence, AngleTerm, AtomRecord, BondTerm, ExclusionPolicy, PhysicsError,
    ProteinTopology, TorsionTerm,
};

/// Exact conversion factor from `[kcal / (mol * A)] / [amu]` to `[A / ps^2]`.
pub const KCAL_TO_AMU_A2_PS2: f64 = 418.4;

/// Boltzmann constant in `kcal / (mol * K)`.
pub const K_BOLTZMANN_KCAL: f64 = 0.001987204259;

/// Electrostatic conversion constant `1 / (4 * pi * eps_0)` in `(kcal * A) / (mol * e^2)`.
pub const COULOMB_KCAL_A_E2: f64 = 332.063713;
