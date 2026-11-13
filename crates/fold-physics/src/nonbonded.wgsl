// BeoFold Cross-Vendor WGSL Non-Bonded Force & Energy Compute Kernel
// Evaluates C1 force-shifted 12-6 Lennard-Jones + Generalized-Born distance-screened Coulomb
// + Lazaridis-Karplus EEF1 smooth Gaussian hydrophobic solvation using 64-thread shared-memory tiles.

struct GpuUniforms {
    atom_count: u32,
    cutoff_ang: f32,
    coulomb_quarter: f32,
    _pad: u32,
}

struct GpuAtomParams {
    lj_sigma_ang: f32,
    lj_epsilon_kcal: f32,
    charge_e: f32,
    flags: u32,        // bit 0: is_backbone (1u = backbone, 0u = sidechain)
    excl_offset: u32,  // CSR start index in `csr_exclusions`
    excl_count: u32,   // CSR entry count for this atom
    _pad0: u32,
    _pad1: u32,
}

struct GpuAtomOutput {
    fx: f32,
    fy: f32,
    fz: f32,
    _pad0: f32,
    e_lj: f32,
    e_elec: f32,
    e_solv: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: GpuUniforms;
@group(0) @binding(1) var<storage, read> atom_params: array<GpuAtomParams>;
@group(0) @binding(2) var<storage, read> csr_exclusions: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputs: array<GpuAtomOutput>;

var<workgroup> tile_pos: array<vec4<f32>, 64>;
// tile_meta: x = sigma, y = epsilon, z = charge, w = f32(flags)
var<workgroup> tile_meta: array<vec4<f32>, 64>;

// Looks up covalent exclusion policy for sorted CSR row `[offset .. offset + count)`:
// Returns 0u = Full (1.0 LJ, 1.0 Elec), 1u = Excluded (1-2 / 1-3), 2u = Scaled14 (0.5 LJ, 1/1.2 Elec).
fn lookup_pair_policy(offset: u32, count: u32, partner: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = count;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        let packed = csr_exclusions[offset + mid];
        let idx = packed & 0x3FFFFFFFu;
        if (idx == partner) {
            return packed >> 30u;
        } else if (idx < partner) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return 0u;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let i = global_id.x;
    let lid = local_id.x;
    let n = uniforms.atom_count;
    let valid_i = i < n;

    var pos_i = vec3<f32>(0.0, 0.0, 0.0);
    var sigma_i: f32 = 0.0;
    var eps_i: f32 = 0.0;
    var q_i: f32 = 0.0;
    var flags_i: u32 = 0u;
    var excl_off: u32 = 0u;
    var excl_cnt: u32 = 0u;

    if (valid_i) {
        pos_i = positions[i].xyz;
        let p = atom_params[i];
        sigma_i = p.lj_sigma_ang;
        eps_i = p.lj_epsilon_kcal;
        q_i = p.charge_e;
        flags_i = p.flags;
        excl_off = p.excl_offset;
        excl_cnt = p.excl_count;
    }

    let rc = uniforms.cutoff_ang;
    let rc2 = rc * rc;
    let rc3 = rc2 * rc;

    var fi = vec3<f32>(0.0, 0.0, 0.0);
    var e_lj_acc: f32 = 0.0;
    var e_elec_acc: f32 = 0.0;
    var e_solv_acc: f32 = 0.0;

    let num_tiles = (n + 63u) / 64u;

    // All 64 invocations execute workgroupBarrier() in uniform control flow
    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let load_idx = tile * 64u + lid;
        if (load_idx < n) {
            tile_pos[lid] = positions[load_idx];
            let tp = atom_params[load_idx];
            tile_meta[lid] = vec4<f32>(
                tp.lj_sigma_ang,
                tp.lj_epsilon_kcal,
                tp.charge_e,
                f32(tp.flags),
            );
        } else {
            tile_pos[lid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            tile_meta[lid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }

        workgroupBarrier();

        if (valid_i) {
            let tile_base = tile * 64u;
            for (var k: u32 = 0u; k < 64u; k = k + 1u) {
                let j = tile_base + k;
                if (j >= n || j == i) {
                    continue;
                }

                let dvec = pos_i - tile_pos[k].xyz;
                let r2 = dot(dvec, dvec);
                if (r2 >= rc2 || r2 < 1.0e-8) {
                    continue;
                }

                let policy = lookup_pair_policy(excl_off, excl_cnt, j);
                if (policy == 1u) {
                    continue;
                }

                var lj_scale: f32 = 1.0;
                var elec_scale: f32 = 1.0;
                if (policy == 2u) {
                    lj_scale = 0.5;
                    elec_scale = 1.0 / 1.2;
                }

                let r = sqrt(r2);
                let meta_j = tile_meta[k];
                let sigma_j = meta_j.x;
                let eps_j = meta_j.y;
                let q_j = meta_j.z;
                let flags_j = u32(meta_j.w + 0.5);

                let sigma = 0.5 * (sigma_i + sigma_j);
                let eps = lj_scale * sqrt(eps_i * eps_j);

                var du_dr_total: f32 = 0.0;

                // 1. C1 force-shifted 12-6 Lennard-Jones
                if (eps > 0.0) {
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

                    du_dr_total = du_dr_total + (du_lj - duc_lj);
                    if (i < j) {
                        e_lj_acc = e_lj_acc + (u_lj - uc_lj - (r - rc) * duc_lj);
                    }
                }

                // 2. Implicit-solvent distance-screened Coulomb (eps_eff(r) = 4.0 * r)
                let q_prod = q_i * q_j;
                if (abs(q_prod) > 1.0e-7) {
                    let k_q = elec_scale * uniforms.coulomb_quarter * q_prod;
                    let u_el = k_q / r2;
                    let uc_el = k_q / rc2;
                    let du_el = -2.0 * k_q / (r2 * r);
                    let duc_el = -2.0 * k_q / rc3;

                    du_dr_total = du_dr_total + (du_el - duc_el);
                    if (i < j) {
                        e_elec_acc = e_elec_acc + (u_el - uc_el - (r - rc) * duc_el);
                    }
                }

                // 3. EEF1 Lazaridis-Karplus smooth Gaussian hydrophobic burial between non-backbone sidechains
                if ((flags_i & 1u) == 0u && (flags_j & 1u) == 0u && policy == 0u) {
                    let r_contact = 0.5 * (sigma_i + sigma_j) + 1.4;
                    let width: f32 = 1.85;
                    let well = 0.28 * sqrt(eps_i * eps_j);
                    let z = (r - r_contact) / width;
                    let z_cut = (rc - r_contact) / width;

                    let u_g = -well * exp(-z * z);
                    let uc_g = -well * exp(-z_cut * z_cut);
                    let du_g = -u_g * (2.0 * z / width);
                    let duc_g = -uc_g * (2.0 * z_cut / width);

                    du_dr_total = du_dr_total + (du_g - duc_g);
                    if (i < j) {
                        e_solv_acc = e_solv_acc + (u_g - uc_g - (r - rc) * duc_g);
                    }
                }

                let f_over_r = -du_dr_total / r;
                fi = fi + f_over_r * dvec;
            }
        }

        workgroupBarrier();
    }

    if (valid_i) {
        outputs[i] = GpuAtomOutput(
            fi.x,
            fi.y,
            fi.z,
            0.0,
            e_lj_acc,
            e_elec_acc,
            e_solv_acc,
            0.0,
        );
    }
}
