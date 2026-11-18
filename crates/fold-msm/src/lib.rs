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

#![forbid(unsafe_code)]

pub mod clustering;
pub mod landscape;
pub mod transition;

pub use clustering::{
    cluster_conformations_kmedoids, compute_pairwise_rmsd_matrix, extract_project_frames,
    ConformationalFrame, MicrostateClustering, MsmError,
};
pub use landscape::{compute_2d_pmf_grid, export_msm_bundle, ExportedMsmArtifacts, PmfGrid2D};
pub use transition::{
    cyclic_jacobi_symmetric_eigen, estimate_reversible_msm, estimate_reversible_msm_with_max_temp,
    MetastableMacrostate, ReversibleMsmModel,
};
