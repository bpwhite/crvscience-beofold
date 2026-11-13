pub mod forcefield;
pub mod gpu;
pub mod integrator;
pub mod morton;
pub mod mremd;
pub mod rng;
pub mod sanity;
pub mod topology;

pub use forcefield::{
    compute_sasa_angstrom2, evaluate_forces_and_energy, evaluate_forces_and_energy_with_gpu,
    EnergyBreakdown,
};
pub use gpu::{probe_gpu_adapter, GpuAdapterSummary, GpuComputeError, GpuForceEngine};
pub use integrator::{
    initialize_clone_checkpoint, minimize_steric_clashes, run_generation_chunk,
    run_generation_chunk_with_gpu, GenerationSummary, LangevinConfig,
};
pub use morton::{morton_encode_3d, SpatialNeighborGrid};
pub use mremd::{
    benchmark_local_node, benchmark_local_node_with_gpu, metropolis_swap_probability,
    try_mremd_swap,
};
pub use rng::ChaCha8Rng;
pub use sanity::{
    compute_ca_rmsd, compute_radius_of_gyration, verify_checkpoint_sanity, SanityError,
    SanityThresholds,
};
pub use topology::{
    AngleTerm, AtomRecord, BondTerm, ExclusionPolicy, PhysicsError, ProteinTopology, TorsionTerm,
};

/// Exact conversion factor from `[kcal / (mol * A)] / [amu]` to `[A / ps^2]`.
pub const KCAL_TO_AMU_A2_PS2: f64 = 418.4;

/// Boltzmann constant in `kcal / (mol * K)`.
pub const K_BOLTZMANN_KCAL: f64 = 0.001987204259;

/// Electrostatic conversion constant `1 / (4 * pi * eps_0)` in `(kcal * A) / (mol * e^2)`.
pub const COULOMB_KCAL_A_E2: f64 = 332.063713;
