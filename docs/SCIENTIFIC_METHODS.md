# BeoFold (`v1.4.0`) — Scientific & Mathematical Methods

*License: GNU General Public License v3.0 or later (`GPL-3.0-or-later`).*

This document specifies the physical force field, stochastic integrator, replica-exchange thermostatting, cryptographic validation, and Markov State Model (`MSM`) kinetic analysis algorithms implemented in BeoFold (`crates/fold-physics`, `crates/fold-storage`, and `crates/fold-msm`).

---

## 1. Units & Physical Constants

BeoFold uses standard biomolecular MD internal units (`Å`, `ps`, `amu`, `kcal/mol`, elementary charge `e`, Kelvin `K`):

- `KCAL_TO_AMU_A2_PS2 = 418.4` (`[kcal / (mol * Å)] / amu -> Å / ps^2`)
- `k_B = 0.001987204259 kcal / (mol * K)` (Boltzmann constant)
- `k_e = 1 / (4 * pi * eps_0) = 332.063713 (kcal * Å) / (mol * e^2)` (Coulomb constant)

---

## 2. Molecular Mechanics Force Field (`fold-physics`)

Total potential energy `V_total(r)` decomposes into bonded and non-bonded terms (`C^1` smooth up to `r_c = 12.0 Å`):

```text
V_total(r) = V_bond + V_angle + V_torsion + V_LJ + V_elec_GB + V_EEF1
```

### 2.1 Covalent Bonded Terms & Exclusions

Following `AMBER ff14SB` conventions, non-bonded pair policies are:
- `1-2` (covalently bonded) and `1-3` (valence angle) pairs: **Excluded** (`0.0`).
- `1-4` dihedral terminal pairs: **Scaled14** (`0.5 * V_LJ`, `(1 / 1.2) * V_elec_GB`; EEF1 omitted).
- All remaining pairs within `r < r_c`: **Full** (`1.0`).

Bonded potential terms (`forcefield.rs`, evaluated in double precision `f64`):

```text
V_bond(r)       = 0.5 * k_b * (r - r_0)^2
V_angle(theta)  = 0.5 * k_theta * (theta - theta_0)^2
V_torsion(phi)  = 0.5 * V_n * (1 + cos(n * phi - gamma))
```

Cartesian forces for dihedrals use the singularity-free Blondel & Karplus (1996) formulation, avoiding division by `sin(phi)` near planar configurations (`0` or `pi`), with elevated `n = 2`, `gamma = pi`, `V_2 = 5.0 kcal/mol` barriers preserving peptide-bond (`C-N`) planarity.

### 2.2 Force-Shifted 12-6 Lennard-Jones

For atom pair `(i, j)` with Lorentz-Berthelot combining rules `sigma_ij = 0.5 * (sigma_i + sigma_j)` and `eps_ij = lj_scale * sqrt(eps_i * eps_j)`:

```text
u_LJ(r)         = 4 * eps_ij * [ (sigma_ij / r)^12 - (sigma_ij / r)^6 ]
u_LJ_shifted(r) = u_LJ(r) - u_LJ(r_c) - (r - r_c) * (du_LJ / dr)|_{r = r_c}
```

Both potential energy and radial force vanish continuously at `r = r_c = 12.0 Å`.

### 2.3 Still Generalized-Born Screened Electrostatics (`gb_radius_ang`)

Implicit-solvent electrostatic screening combines interior solute polarization (`eps_in = 1.0`) and bulk aqueous screening (`eps_out = 78.5`) via the Still et al. (1990) effective interaction distance `f_GB(r)` parameterized by each atom's effective Born radius `R_i = max(gb_radius_ang[i], 0.5 Å)`:

```text
a          = R_i * R_j
h(r)       = a * exp( -r^2 / (4 * a) )
f_GB(r)    = sqrt( r^2 + h(r) )
```

To eliminate `f32` catastrophic cancellation at long range inside the WGSL GPU kernel (`nonbonded.wgsl`) while matching CPU `f64` evaluation (`forcefield.rs`), the pairwise factor `g(r) = 1/(eps_in * r) - (1/eps_in - 1/eps_out)/f_GB(r)` is evaluated in exact algebraic form:

```text
g(r) = 1 / (eps_out * f_GB) + (1 / eps_in) * h(r) / [ r * f_GB * (f_GB + r) ]

dg/dr = -(1 / eps_out) * [ r * (1 - 0.25 * exp(-r^2 / (4*a))) / f_GB^3 ]
        -(1 / eps_in)  * [ h(r)*(f_GB^2 + f_GB*r + r^2) / (r^2 * f_GB^3 * (f_GB + r))
                           + r * exp(-r^2 / (4*a)) / (4 * f_GB^3) ]
```

The pairwise interaction `U_elec(r) = elec_scale * k_e * q_i * q_j * g(r)` is `C^1` force-shifted at `r_c = 12.0 Å` consistently on both CPU (`rayon` Morton grid) and GPU (`wgpu` 64-thread tiled WGSL compute shader).

### 2.4 Lazaridis-Karplus EEF1 Signed Gaussian Solvation

Non-backbone sidechain heavy atoms carry a signed solvation weight `s_i = eef1_solvation_weight(atom)`:
- **Hydrophobic / apolar (`s_i > 0`)**: `TRP (+1.15)`, `PHE/ILE/LEU (+1.00)`, `MET/VAL (+0.90)`, `TYR (+0.85)`, `CYS (+0.80)`, `PRO (+0.60)`, `ALA (+0.55)`, and apolar PDB `C/S` atoms (`+0.85` / `+0.45`).
- **Polar / charged (`s_i < 0`)**: `SER/THR (-0.55)`, `ASN/GLN/HIS (-0.75)`, `LYS/ARG (-1.10)`, `ASP/GLU (-1.15)`, and charged (`|q| >= 0.25`) or polar (`N/O`) PDB sidechain atoms (`-1.05` / `-0.65`).

For full non-bonded sidechain pairs with mean weight `s_avg = 0.5 * (s_i + s_j)`:
- **Hydrophobic burial (`s_avg > 0`)**: Attractive negative Gaussian well centered at `r_contact = sigma_ij + 1.4 Å` (`width = 1.85 Å`, `well = 0.28 * sqrt(eps_i * eps_j) * s_avg`), driving hydrophobic core collapse:
  ```text
  u_g(r) = -well * exp( -((r - r_contact) / 1.85)^2 )
  ```
- **Polar/charged desolvation (`s_avg < 0`)**: Monotonically repulsive desolvation penalty (`width = 2.20 Å`, `penalty = 0.28 * max(sqrt(eps_i * eps_j), 0.28) * (-s_avg)`), penalizing loss of hydration shell contacts when `r > sigma_ij` (and `C^1`-flat for `r <= sigma_ij`):
  ```text
  u_g(r) = +penalty * exp( -(max(r - sigma_ij, 0) / 2.20)^2 )
  ```

Both branches are `C^1` force-shifted at `r_c = 12.0 Å` identically on CPU and WGSL GPU (`GpuAtomParams.gb_radius_ang`, `GpuAtomParams.solv_weight`).

### 2.5 Solvent-Accessible Surface Area (SASA)

Atomic SASA (`Å^2`) is computed via Shrake-Rupley quadrature with `N = 32` deterministic Fibonacci sphere points per atom and water probe radius `r_probe = 1.40 Å` (`R_eff = gb_radius_ang + 1.40 Å`), accelerated by the 3D Morton spatial grid.

---

## 3. Symmetric BAOAB Langevin Integrator & Safety Guards

Trajectories advance via Leimkuhler-Matthews **BAOAB** splitting (`dt = 1.5–2.0 fs`), which yields second-order weak accuracy and exact configurational sampling:

```text
B (dt/2): v <- v + (dt / 2) * (KCAL_TO_AMU_A2_PS2 / m) * F(x)
A (dt/2): x <- x + (dt / 2) * v
O (dt)  : v <- c1 * v + c2 * sqrt(KCAL_TO_AMU_A2_PS2 * k_B * T / m) * G_n
A (dt/2): x <- x + (dt / 2) * v
          F <- -grad V_total(x)
B (dt/2): v <- v + (dt / 2) * (KCAL_TO_AMU_A2_PS2 / m) * F(x)
```

where `c1 = exp(-gamma * dt)`, `c2 = sqrt(1 - c1^2)`, and `G_n ~ N(0, I_3)` is drawn deterministically from checkpointed counter-based `ChaCha8Rng`. Setting `gamma = 0.0 ps^-1` selects pure symplectic velocity-Verlet (`NVE`) for energy conservation verification.

**Numerical Safety Guards**:
- Atoms with `mass_amu <= 0.0` or non-finite mass remain immobilized (`v = [0, 0, 0]`) without division by zero.
- Target bath temperatures `T <= 0.0` or non-finite `T` safely clamp thermal noise amplitude to `0.0` (zero `NaN` propagation).

---

## 4. Multiplexed-Replica Exchange MD (MREMD)

Adjacent temperature rungs `(T_1, V_1)` and `(T_2, V_2)` undergo asynchronous Metropolis exchange trials with acceptance probability:

```text
beta_k   = 1 / (k_B * T_k)
P_accept = min( 1.0, exp( (beta_1 - beta_2) * (V_1 - V_2) ) )
```

- **Invalid-Input Guard**: If `T_1 <= 0`, `T_2 <= 0`, or any temperature/potential is non-finite (`NaN`/`Inf`), `metropolis_swap_probability` returns `0.0` without panicking.
- **Sugita-Okamoto (1999) Velocity Rescaling**: Whenever a replica exchanges from `T_old` to `T_new`, its atomic velocities are canonically rescaled:

```text
v_new = v_old * sqrt(T_new / T_old)
K_new = K_old * (T_new / T_old)
```

This rescaling is applied identically in:
1. `mremd::rescale_velocities_sugita_okamoto` & `try_mremd_swap_checkpoints` (`fold-physics`),
2. `FoldStore::acquire_next_lease_filtered` (`fold-storage`) when a swapped `Gen G+1` work unit is leased,
3. `integrator::run_generation_chunk_with_gpu` (`fold-physics`) whenever thermostatted integration transitions between temperatures (guaranteeing bit-exact replay in Gate 3A canary spot-checks), and
4. `WorkerSimState` (`fold-gui`) during live interactive MREMD folding.

---

## 5. Markov State Model (MSM) Pipeline (`fold-msm`)

All algorithms in `crates/fold-msm` compile under `#![forbid(unsafe_code)]`:

1. **Optimal Horn/Kabsch C-alpha RMSD (`compute_ca_rmsd`)**: Translates centroids to the origin and solves Horn's (1987) `4x4` symmetric quaternion key matrix `K` via 25 cyclic Jacobi sweeps:
   ```text
   RMSD = sqrt( max(0, (G_0 - 2 * lambda_max(K)) / N_CA) )
   ```
2. **Clustering (`cluster_conformations_kmedoids`)**: Deterministic Maximin Regular-Space seeding anchored at the global lowest-energy frame followed by Partitioning Around Medoids (`PAM`) refinement so every microstate center is a sterically valid conformation.
3. **Ergodic Pruning & Reversible MLE (`estimate_reversible_msm`)**: Counts sequential `(Run, Clone)` transitions at integer lag `tau`, restricts to cold MREMD rungs (`T <= T_ref + 25 K` when ergodic, with automatic fallback), extracts the largest **Tarjan (1972) Strongly Connected Component (SCC)**, and optimizes the Prinz-Noé-Pande (2011) reversible maximum-likelihood transition matrix `T_ij` satisfying exact detailed balance `pi_i * T_ij = pi_j * T_ji`.
4. **Spectral Kinetics, TPT & PCCA+**:
   - Symmetrizes `S_ij = x_ij / sqrt(X_i * X_j)` and computes all real eigenvalues `1 = lambda_1 >= lambda_2 >= ...` via pure-Rust Cyclic Jacobi eigendecomposition (`t_k = -lag_ps / ln(lambda_k)`).
   - Solves the Transition Path Theory (`TPT`) Dirichlet linear system `(I - T_II) q_I^+ = T_{I,F}` for forward folding committors `p_fold in [0, 1]`.
   - Applies Röblitz & Weber (2013) Inner-Simplex Algorithm (`ISA`) + crispness maximization `I(A)` (`chi = Psi * A`, `chi_{i,a} >= 0`, `sum_a chi_{i,a} = 1`) to coarse-grain microstates into `M` metastable PCCA+ macrostates.
5. **2D KDE PMF & Iso-Energy Contours (`compute_2d_pmf_grid`, `extract_pmf_iso_contours`)**:
   - Evaluates an equilibrium-reweighted `40x40` 2D Gaussian KDE on `(RMSD_best, R_g)` where each frame in active microstate `i` carries statistical weight `pi_i / count(i)`.
   - Computes adaptive 2D Silverman/Scott bandwidths per coordinate using Kish's effective sample size `N_eff = (sum w)^2 / sum(w^2)` and robust spread `min(sigma, IQR / 1.34)`:
     ```text
     h_d = 0.96 * N_eff^(-1/6) * min(sigma_d, IQR_d / 1.34)
     W(x, y) = -k_B * T_ref * ln( P_eq(x, y) / P_max )
     ```
   - Extracts smooth iso-energy contour lines at **`1, 2, 3 kcal/mol`** via Marching Squares with saddle-point disambiguation for both SVG reports and the interactive GUI canvas.

---

## 6. Primary Literature References

1. **BAOAB Langevin**: Leimkuhler, B., & Matthews, C. (2013). *Rational Construction of Stochastic Numerical Methods for Molecular Sampling*, Appl. Math. Res. Express, 2013(1), 34–56.
2. **Singularity-Free Torsions**: Blondel, A., & Karplus, M. (1996). *New Formulation for Derivatives of Torsion Angles and Improper Torsion Angles in Molecular Mechanics*, J. Comput. Chem., 17(9), 1132–1141.
3. **Still Generalized Born**: Still, W. C., Tempczyk, A., Hawley, R. C., & Hendrickson, T. (1990). *Semianalytical Treatment of Solvation for Molecular Mechanics and Dynamics*, J. Am. Chem. Soc., 112(16), 6127–6129.
4. **EEF1 Solvation**: Lazaridis, T., & Karplus, M. (1999). *Effective Energy Function for Proteins in Solution*, Proteins, 35(2), 133–152.
5. **REMD & Velocity Rescaling**: Sugita, Y., & Okamoto, Y. (1999). *Replica-Exchange Molecular Dynamics Method for Protein Folding*, Chem. Phys. Lett., 314(1–2), 141–151.
6. **Quaternion RMSD**: Horn, B. K. P. (1987). *Closed-Form Solution of Absolute Orientation Using Unit Quaternions*, J. Opt. Soc. Am. A, 4(4), 629–642.
7. **Reversible MSMs**: Prinz, J.-H., Wu, H., Sarich, M., Keller, B., Senne, M., Held, M., Chodera, J. D., Schütte, C., & Noé, F. (2011). *Markov Models of Molecular Kinetics: Generation and Validation*, J. Chem. Phys., 134(17), 174105.
8. **PCCA+**: Röblitz, S., & Weber, M. (2013). *Fuzzy Spectral Clustering by PCCA+: Application to Markov State Models and Data Classification*, Adv. Data Anal. Classif., 7(2), 147–179.
9. **Adaptive KDE**: Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*, Chapman & Hall.
