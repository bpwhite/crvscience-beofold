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

use fold_msm::{
    cluster_conformations_kmedoids, compute_2d_pmf_grid, compute_pairwise_rmsd_matrix,
    estimate_reversible_msm, extract_project_frames, ConformationalFrame, MicrostateClustering,
    PmfGrid2D, ReversibleMsmModel,
};
use fold_physics::{
    compute_radius_of_gyration, fraction_native_contacts, initialize_clone_checkpoint,
    metropolis_swap_probability, minimize_steric_clashes, native_ca_contact_pairs,
    run_generation_chunk, try_mremd_swap, verify_checkpoint_sanity, ChaCha8Rng, LangevinConfig,
    ProteinTopology, SanityThresholds, K_BOLTZMANN_KCAL,
};
use fold_proto::{
    MremdEndpoint, NodeIdentity, ProjectManifest, SignedWorkUnitResult, TrajectoryCheckpoint,
    WorkUnitId,
};
use fold_storage::{CommitOutcome, FoldStore};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

static SUMMARY_LOCK: Mutex<()> = Mutex::new(());
static SHARED_ENSEMBLE: OnceLock<MremdEnsembleTelemetry> = OnceLock::new();

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp validation directory");
        Self { path: dir }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("beofold repo root")
        .to_path_buf()
}

fn validation_dir() -> PathBuf {
    let dir = repo_root().join("validation");
    fs::create_dir_all(&dir).expect("create validation dir");
    dir
}

fn extract_existing_json_sections(json_text: &str) -> BTreeMap<String, String> {
    let mut sections = BTreeMap::new();
    let Some(tests_pos) = json_text.find("\"tests\"") else {
        return sections;
    };
    let Some(open_rel) = json_text[tests_pos..].find('{') else {
        return sections;
    };
    let bytes = json_text.as_bytes();
    let mut i = tests_pos + open_rel + 1;
    let len = bytes.len();

    while i < len {
        while i < len && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= len || bytes[i] == b'}' {
            break;
        }
        if bytes[i] != b'"' {
            break;
        }
        let key_start = i + 1;
        i += 1;
        while i < len && bytes[i] != b'"' {
            i += 1;
        }
        let key = json_text[key_start..i].to_string();
        i += 1;
        while i < len && bytes[i] != b'{' && bytes[i] != b'}' {
            i += 1;
        }
        if i >= len || bytes[i] != b'{' {
            break;
        }
        let val_start = i;
        let mut depth = 0usize;
        let mut in_str = false;
        let mut escaped = false;
        while i < len {
            let b = bytes[i];
            if in_str {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_str = false;
                }
            } else {
                match b {
                    b'"' => in_str = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            let val_end = i + 1;
                            sections.insert(key, json_text[val_start..val_end].to_string());
                            i = val_end;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
    }

    sections
}

fn write_summary_section(section_key: &str, section_json_object: String) {
    let _guard = SUMMARY_LOCK.lock().expect("lock summary mutex");
    let local_path = validation_dir().join("beofold_validation_summary.json");

    let mut sections = if local_path.exists() {
        fs::read_to_string(&local_path)
            .ok()
            .map(|s| extract_existing_json_sections(&s))
            .unwrap_or_default()
    } else {
        BTreeMap::new()
    };
    sections.insert(section_key.to_string(), section_json_object);

    let mut out = String::from(
        "{\n  \"concept_doi\": \"10.5281/zenodo.24222370\",\n  \"preprint_id\": \"APP-PREPRINT-02\",\n  \"preprint_url\": \"https://crvscience.com/apps/preprints/beofold-distributed-protein-msm\",\n  \"repo\": \"crvscience-beofold\",\n  \"schema_version\": \"1.0\",\n  \"tests\": {\n",
    );

    let total = sections.len();
    for (idx, (k, v)) in sections.iter().enumerate() {
        let comma = if idx + 1 < total { "," } else { "" };
        let _ = writeln!(out, "    \"{k}\": {v}{comma}");
    }
    out.push_str("  },\n  \"version\": \"1.4.0\"\n}\n");

    fs::write(&local_path, &out).expect("write beofold validation summary JSON");

    if let Some(hub_root) = repo_root()
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
    {
        let hub_releases = hub_root.join("app-preprints").join("releases");
        if hub_releases.is_dir() {
            let hub_path =
                hub_releases.join("beofold-distributed-protein-msm-validation-telemetry.json");
            fs::write(&hub_path, &out).expect("write hub beofold telemetry copy");
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

#[allow(dead_code)]
struct MremdEnsembleTelemetry {
    manifest: ProjectManifest,
    topology: ProteinTopology,
    frames: Vec<ConformationalFrame>,
    native_pairs: Vec<(usize, usize)>,
    frame_q_scores: Vec<f64>,
    best_frame_idx: usize,
    total_completed_wus: usize,
    mremd_trials: u64,
    mremd_swaps: u64,
    direct_pair_swap_trials: usize,
    direct_pair_swap_accepts: usize,
    mean_pair_acceptance_prob: f64,
    clustering: MicrostateClustering,
    msm: ReversibleMsmModel,
    pmf: PmfGrid2D,
}

fn build_or_get_mremd_ensemble() -> &'static MremdEnsembleTelemetry {
    SHARED_ENSEMBLE.get_or_init(|| {
        let temp = TempDirGuard::new("beofold_mremd_validation");
        let store = FoldStore::open(&temp.path).expect("open FoldStore");
        let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
        minimize_steric_clashes(&topo, &mut coords, 120);

        let temp_ladder = vec![285.0_f64, 320.0, 360.0, 405.0];
        let num_clones_per_run = 2u32;
        let max_generations = 8u32;
        let steps_per_gen = 200u64;

        let mut manifest = ProjectManifest {
            project_id: 1,
            name: "TrpCage_1L2Y_Headless_MREMD".to_string(),
            sequence: topo.sequence_one_letter.clone(),
            topology_blake3: [0u8; 32],
            initial_coords_blake3: [0u8; 32],
            atom_count: topo.atoms.len() as u32,
            residue_count: topo.residue_count,
            temp_ladder_kelvin: temp_ladder.clone(),
            dt_fs: 2.0,
            friction_per_ps: 1.5,
            steps_per_gen,
            num_clones_per_run,
            max_generations,
        };

        let mut gen0_cps: Vec<TrajectoryCheckpoint> = Vec::new();
        for (run_idx, &temp_k) in temp_ladder.iter().enumerate() {
            for clone_idx in 0..num_clones_per_run {
                let wu_id = WorkUnitId::new(manifest.project_id, run_idx as u32, clone_idx, 0);
                gen0_cps.push(initialize_clone_checkpoint(
                    &topo,
                    &coords,
                    wu_id,
                    temp_k,
                    b"beofold-headless-validation",
                ));
            }
        }

        manifest = store
            .seed_project(manifest, &topo, &gen0_cps)
            .expect("seed 4-replica Trp-cage project");

        let worker = NodeIdentity::from_cluster_secret("beofold-val-secret", "worker-01");
        let thresholds = SanityThresholds::default();
        let mut mremd_rng = ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 0, 0, 0), b"mremd-val");
        let mut now_sec = 1_700_000_000_u64;
        let mut accepted_count = 0usize;

        loop {
            let mut batch = Vec::new();
            while let Some((lease, input_cp)) = store
                .acquire_next_lease(worker.public_key_bytes(), now_sec, 600)
                .expect("acquire MREMD lease")
            {
                now_sec += 1;
                batch.push((lease, input_cp));
                if batch.len() >= 8 {
                    break;
                }
            }
            if batch.is_empty() {
                break;
            }

            let worker_ref = &worker;
            let topo_ref = &topo;
            let dt_fs = manifest.dt_fs;
            let friction_per_ps = manifest.friction_per_ps;
            let steps = manifest.steps_per_gen;

            let completed: Vec<(SignedWorkUnitResult, TrajectoryCheckpoint)> = batch
                .into_par_iter()
                .map(|(lease, input_cp)| {
                    let cfg = LangevinConfig {
                        dt_fs,
                        friction_per_ps,
                        temperature_kelvin: input_cp.temperature_kelvin,
                        steps,
                        sample_stride: 10,
                    };
                    let summary = run_generation_chunk(topo_ref, &input_cp, lease.wu_id, &cfg);
                    let out_frame = summary
                        .checkpoint
                        .to_compressed_frame()
                        .expect("encode output checkpoint");
                    let out_hash = *blake3::hash(&out_frame).as_bytes();

                    let mut signed = SignedWorkUnitResult {
                        wu_id: lease.wu_id,
                        lease_token: lease.lease_token,
                        worker_pubkey: worker_ref.public_key_bytes(),
                        input_checkpoint_blake3: lease.input_checkpoint_blake3,
                        output_checkpoint_blake3: out_hash,
                        steps_executed: steps,
                        temperature_kelvin: summary.checkpoint.temperature_kelvin,
                        mean_potential_kcal: summary.mean_potential_kcal,
                        min_potential_kcal: summary.min_potential_kcal,
                        final_potential_kcal: summary.checkpoint.potential_energy_kcal,
                        final_kinetic_kcal: summary.checkpoint.kinetic_energy_kcal,
                        final_rg_angstrom: summary.checkpoint.rg_angstrom,
                        final_sasa_angstrom2: summary.checkpoint.sasa_angstrom2,
                        wall_time_ms: 8,
                        signature: [0u8; 64],
                    };
                    worker_ref.sign_result(&mut signed);
                    (signed, summary.checkpoint)
                })
                .collect();

            for (signed, cp) in completed {
                now_sec += 1;
                let outcome = store
                    .commit_signed_result(
                        &manifest,
                        &topo,
                        &signed,
                        &cp,
                        &thresholds,
                        &mut mremd_rng,
                        now_sec,
                    )
                    .expect("commit signed MREMD chunk");
                match outcome {
                    CommitOutcome::Accepted { .. } => accepted_count += 1,
                    other => panic!("Expected CommitOutcome::Accepted, got {other:?}"),
                }
            }
        }

        let stats = store
            .collect_stats(Some(manifest.project_id))
            .expect("collect stats");

        let (_loaded_manifest, loaded_topo, frames) =
            extract_project_frames(&store, manifest.project_id, None)
                .expect("extract MREMD frames");
        assert!(
            frames.len() >= 48,
            "Expected >= 48 trajectory frames across 8 chains, got {}",
            frames.len()
        );

        // Identify lowest-energy folded reference frame and compute native contacts Q
        let mut best_frame_idx = 0usize;
        for (i, f) in frames.iter().enumerate() {
            if f.potential_energy_kcal < frames[best_frame_idx].potential_energy_kcal {
                best_frame_idx = i;
            }
        }
        let native_pairs =
            native_ca_contact_pairs(&loaded_topo, &frames[best_frame_idx].positions_angstrom, 8.5);
        let frame_q_scores: Vec<f64> = frames
            .iter()
            .map(|f| fraction_native_contacts(&native_pairs, &f.positions_angstrom, 8.8))
            .collect();

        // Direct replica-exchange Metropolis verification across adjacent rungs using real frame energies
        let mut pair_trials = 0usize;
        let mut pair_accepts = 0usize;
        let mut prob_sum = 0.0_f64;
        let mut direct_rng =
            ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 0, 1, 1), b"direct-mremd");

        for rung_idx in 0..(temp_ladder.len() - 1) {
            let t_lo = temp_ladder[rung_idx];
            let t_hi = temp_ladder[rung_idx + 1];
            let lo_frames: Vec<&ConformationalFrame> = frames
                .iter()
                .filter(|f| (f.temperature_kelvin - t_lo).abs() < 1.0)
                .collect();
            let hi_frames: Vec<&ConformationalFrame> = frames
                .iter()
                .filter(|f| (f.temperature_kelvin - t_hi).abs() < 1.0)
                .collect();
            for k in 0..lo_frames.len().min(hi_frames.len()) {
                let fa = lo_frames[k];
                let fb = hi_frames[k];
                let p_acc = metropolis_swap_probability(
                    fa.temperature_kelvin,
                    fa.potential_energy_kcal,
                    fb.temperature_kelvin,
                    fb.potential_energy_kcal,
                );
                assert!(
                    (0.0..=1.0).contains(&p_acc),
                    "Metropolis acceptance probability out of range: {p_acc}"
                );
                let ep_a = MremdEndpoint {
                    wu_id: fa.wu_id,
                    rung_index: rung_idx as u32,
                    temperature_kelvin: fa.temperature_kelvin,
                    potential_energy_kcal: fa.potential_energy_kcal,
                    output_checkpoint_blake3: fa.checkpoint_blake3,
                };
                let ep_b = MremdEndpoint {
                    wu_id: fb.wu_id,
                    rung_index: (rung_idx + 1) as u32,
                    temperature_kelvin: fb.temperature_kelvin,
                    potential_energy_kcal: fb.potential_energy_kcal,
                    output_checkpoint_blake3: fb.checkpoint_blake3,
                };
                let (swapped, _, _) = try_mremd_swap(&ep_a, &ep_b, &mut direct_rng);
                pair_trials += 1;
                if swapped {
                    pair_accepts += 1;
                }
                prob_sum += p_acc;
            }
        }

        let rmsd_mat = compute_pairwise_rmsd_matrix(&loaded_topo, &frames);
        let clustering = cluster_conformations_kmedoids(&frames, &rmsd_mat, 10, Some(0.55));
        let lag_time_ps = (manifest.steps_per_gen as f64) * manifest.dt_fs / 1000.0;
        let msm = estimate_reversible_msm(&frames, &clustering, 1, lag_time_ps, 300.0, 3)
            .expect("estimate reversible Markov State Model");
        let pmf = compute_2d_pmf_grid(&frames, &clustering, &msm, 40, 40);

        // Export combined MREMD + MSM folding funnel landscape CSV (< 384 KB)
        let mut csv = String::from(
            "frame_idx,run,clone,generation,temperature_kelvin,potential_energy_kcal,rg_angstrom,sasa_angstrom2,rmsd_to_seed_ang,rmsd_to_lowest_energy_ang,native_contacts_q,microstate_id,microstate_pi,microstate_free_energy_kcal,folding_committor\n",
        );
        for (idx, f) in frames.iter().enumerate() {
            let ms = clustering.frame_assignments[idx];
            let (act_ms, pi_val, fe_val, comm_val) =
                match msm.orig_to_active.get(ms).copied().flatten() {
                    Some(act) => (
                        act,
                        msm.stationary_pi[act],
                        msm.microstate_free_energy_kcal[act],
                        msm.folding_committor[act],
                    ),
                    None => (ms, 0.0, 6.0, 0.0),
                };
            let _ = writeln!(
                csv,
                "{},{},{},{},{:.2},{:.4},{:.4},{:.2},{:.4},{:.4},{:.4},{},{:.6},{:.4},{:.6}",
                f.frame_idx,
                f.wu_id.run,
                f.wu_id.clone,
                f.chain_step,
                f.temperature_kelvin,
                f.potential_energy_kcal,
                f.rg_angstrom,
                f.sasa_angstrom2,
                f.rmsd_to_seed_ang,
                f.rmsd_to_lowest_energy_ang,
                frame_q_scores[idx],
                act_ms,
                pi_val,
                fe_val,
                comm_val,
            );
        }
        let csv_path = validation_dir().join("mremd_msm_landscape.csv");
        fs::write(&csv_path, &csv).expect("write mremd_msm_landscape.csv");

        MremdEnsembleTelemetry {
            manifest,
            topology: loaded_topo,
            frames,
            native_pairs,
            frame_q_scores,
            best_frame_idx,
            total_completed_wus: accepted_count,
            mremd_trials: stats.mremd_trials,
            mremd_swaps: stats.mremd_swaps,
            direct_pair_swap_trials: pair_trials,
            direct_pair_swap_accepts: pair_accepts,
            mean_pair_acceptance_prob: if pair_trials > 0 {
                prob_sum / pair_trials as f64
            } else {
                0.0
            },
            clustering,
            msm,
            pmf,
        }
    })
}

#[test]
fn t1_trp_cage_baoab_stability_and_hydrophobic_collapse() {
    let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
    assert_eq!(topo.residue_count, 20);
    assert_eq!(topo.sequence_one_letter, "NLYIQWLKDGGPSSGRPPPS");

    let rg_extended_initial = compute_radius_of_gyration(&topo, &coords);
    let minimized_pot = minimize_steric_clashes(&topo, &mut coords, 120);
    let rg_post_min = compute_radius_of_gyration(&topo, &coords);

    let target_temp_k = 300.0_f64;
    let mut cp = initialize_clone_checkpoint(
        &topo,
        &coords,
        WorkUnitId::new(1, 0, 0, 0),
        target_temp_k,
        b"trp-cage-baoab-t1",
    );

    let n_dof = (3 * topo.atoms.len() - 3) as f64;
    let thresholds = SanityThresholds::default();
    let cfg = LangevinConfig {
        dt_fs: 2.0,
        friction_per_ps: 1.5,
        temperature_kelvin: target_temp_k,
        steps: 150,
        sample_stride: 10,
    };

    let mut temps = Vec::new();
    let mut rgs = Vec::new();
    let mut potentials = Vec::new();
    let mut min_potential = f64::INFINITY;

    for gen_idx in 1..=12u32 {
        let wu_id = WorkUnitId::new(1, 0, 0, gen_idx);
        let summary = run_generation_chunk(&topo, &cp, wu_id, &cfg);
        verify_checkpoint_sanity(&topo, &summary.checkpoint, &thresholds)
            .expect("BAOAB checkpoint must remain physically valid");

        cp = summary.checkpoint;
        let t_kin = 2.0 * cp.kinetic_energy_kcal / (n_dof * K_BOLTZMANN_KCAL);
        let rg = compute_radius_of_gyration(&topo, &cp.positions_angstrom);
        assert!(t_kin.is_finite() && cp.potential_energy_kcal.is_finite());

        temps.push(t_kin);
        rgs.push(rg);
        potentials.push(cp.potential_energy_kcal);
        if summary.min_potential_kcal < min_potential {
            min_potential = summary.min_potential_kcal;
        }
    }

    let mean_t_kin = temps.iter().sum::<f64>() / (temps.len() as f64);
    let t_kin_std = (temps
        .iter()
        .map(|t| (t - mean_t_kin).powi(2))
        .sum::<f64>()
        / (temps.len() as f64))
        .sqrt();
    let final_rg = *rgs.last().expect("final Rg");
    let min_rg = rgs.iter().copied().fold(f64::INFINITY, f64::min);
    let final_potential = *potentials.last().expect("final potential");

    assert!(
        (mean_t_kin - target_temp_k).abs() < 80.0,
        "Mean kinetic temperature ({mean_t_kin:.2} K) should track 300 K thermostat bath"
    );
    assert!(
        final_rg < rg_extended_initial,
        "Hydrophobic collapse should reduce Rg from extended seed ({rg_extended_initial:.3} A -> {final_rg:.3} A)"
    );

    let json = format!(
        "{{\n      \"atom_count\": {},\n      \"chunks_completed\": 12,\n      \"dt_fs\": 2.0,\n      \"final_potential_kcal\": {:.4},\n      \"final_rg_angstrom\": {:.4},\n      \"friction_per_ps\": 1.5,\n      \"mean_kinetic_temp_k\": {:.4},\n      \"min_potential_kcal\": {:.4},\n      \"min_rg_angstrom\": {:.4},\n      \"minimized_potential_kcal\": {:.4},\n      \"residue_count\": {},\n      \"rg_extended_initial_angstrom\": {:.4},\n      \"rg_post_minimization_angstrom\": {:.4},\n      \"sequence\": \"{}\",\n      \"target_temp_k\": {:.1},\n      \"temp_std_k\": {:.4},\n      \"total_steps\": 1800\n    }}",
        topo.atoms.len(),
        final_potential,
        final_rg,
        mean_t_kin,
        min_potential,
        min_rg,
        minimized_pot,
        topo.residue_count,
        rg_extended_initial,
        rg_post_min,
        topo.sequence_one_letter,
        target_temp_k,
        t_kin_std
    );
    write_summary_section("t1_trp_cage_baoab_stability", json);
}

#[test]
fn t2_mremd_four_replica_swap_and_folding_funnel() {
    let ens = build_or_get_mremd_ensemble();

    assert!(
        ens.mremd_trials > 0 || ens.direct_pair_swap_trials > 0,
        "4-replica MREMD run must evaluate replica exchange trials"
    );
    assert!(
        ens.direct_pair_swap_accepts > 0 && ens.mean_pair_acceptance_prob > 0.0,
        "Adjacent temperature ladder [285, 320, 360, 405 K] should achieve non-zero Metropolis swap acceptance"
    );
    assert!(
        !ens.native_pairs.is_empty(),
        "Folded reference structure must possess C-alpha native contacts"
    );

    let max_rmsd_best = ens
        .frames
        .iter()
        .map(|f| f.rmsd_to_lowest_energy_ang)
        .fold(0.0_f64, f64::max);
    let min_pot = ens.frames[ens.best_frame_idx].potential_energy_kcal;
    let max_pot = ens
        .frames
        .iter()
        .map(|f| f.potential_energy_kcal)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_q = ens
        .frame_q_scores
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_q = ens
        .frame_q_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    assert!(
        (ens.frame_q_scores[ens.best_frame_idx] - 1.0).abs() < 1e-9,
        "Lowest-energy native frame must satisfy Q = 1.0"
    );
    assert!(
        max_rmsd_best > 0.5 && min_q < 1.0,
        "MREMD trajectory ensemble should sample conformational diversity around native basin (max_rmsd={max_rmsd_best:.3}, min_q={min_q:.3})"
    );

    let json = format!(
        "{{\n      \"csv_artifact\": \"validation/mremd_msm_landscape.csv\",\n      \"direct_pair_swap_accepts\": {},\n      \"direct_pair_swap_trials\": {},\n      \"folded_ref_rg_angstrom\": {:.4},\n      \"lowest_potential_kcal\": {:.4},\n      \"max_potential_kcal\": {:.4},\n      \"max_rmsd_to_best_ang\": {:.4},\n      \"mean_pair_acceptance_prob\": {:.4},\n      \"native_contact_pairs\": {},\n      \"q_max\": {:.4},\n      \"q_min\": {:.4},\n      \"store_mremd_swaps\": {},\n      \"store_mremd_trials\": {},\n      \"temp_ladder_k\": [285.0, 320.0, 360.0, 405.0],\n      \"total_completed_workunits\": {},\n      \"total_frames\": {}\n    }}",
        ens.direct_pair_swap_accepts,
        ens.direct_pair_swap_trials,
        ens.frames[ens.best_frame_idx].rg_angstrom,
        min_pot,
        max_pot,
        max_rmsd_best,
        ens.mean_pair_acceptance_prob,
        ens.native_pairs.len(),
        max_q,
        min_q,
        ens.mremd_swaps,
        ens.mremd_trials,
        ens.total_completed_wus,
        ens.frames.len()
    );
    write_summary_section("t2_mremd_swap_and_funnel", json);
}

#[test]
fn t3_fold_msm_reversible_matrix_and_pcca_macrostates() {
    let ens = build_or_get_mremd_ensemble();
    let msm = &ens.msm;

    assert!(msm.num_microstates >= 2 && msm.num_microstates <= 10);
    assert!(msm.total_transitions > 0);
    assert!(
        msm.detailed_balance_max_error < 1e-6,
        "Reversible MLE transition matrix must satisfy detailed balance within 1e-6 (got {:.3e})",
        msm.detailed_balance_max_error
    );

    let mut max_row_sum_err = 0.0_f64;
    for row in &msm.transition_matrix {
        let sum: f64 = row.iter().sum();
        let err = (sum - 1.0).abs();
        if err > max_row_sum_err {
            max_row_sum_err = err;
        }
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
    assert!(
        max_row_sum_err < 1e-9,
        "All transition matrix rows must sum to 1.0 (max err = {max_row_sum_err:.3e})"
    );

    let pi_sum: f64 = msm.stationary_pi.iter().sum();
    assert!((pi_sum - 1.0).abs() < 1e-6);

    assert!(
        !msm.implied_timescales_ps.is_empty(),
        "MSM must yield positive relaxation implied timescales"
    );
    for w in msm.implied_timescales_ps.windows(2) {
        assert!(
            w[0] + 1e-9 >= w[1],
            "Implied timescales must be sorted descending"
        );
    }

    let macro_pop_sum: f64 = msm
        .macrostates
        .iter()
        .map(|m| m.stationary_population)
        .sum();
    assert!(
        (macro_pop_sum - 1.0).abs() < 1e-5,
        "PCCA+ macrostate populations must sum to 1.0 (got {macro_pop_sum:.6})"
    );

    let mut macro_json = String::from("[");
    for (idx, m) in msm.macrostates.iter().enumerate() {
        let comma = if idx + 1 < msm.macrostates.len() {
            ", "
        } else {
            ""
        };
        let _ = write!(
            macro_json,
            "{{\"macro_id\": {}, \"label\": \"{}\", \"population\": {:.6}, \"free_energy_kcal\": {:.4}, \"mean_rg_angstrom\": {:.4}, \"mean_rmsd_best_ang\": {:.4}, \"mean_committor\": {:.4}}}{}",
            m.macro_id,
            json_escape(&m.biophysical_label),
            m.stationary_population,
            m.free_energy_kcal,
            m.mean_rg_angstrom,
            m.mean_rmsd_best_ang,
            m.mean_committor,
            comma
        );
    }
    macro_json.push(']');

    let mut its_json = String::from("[");
    for (idx, &its) in msm.implied_timescales_ps.iter().take(4).enumerate() {
        let comma = if idx + 1 < msm.implied_timescales_ps.len().min(4) {
            ", "
        } else {
            ""
        };
        let _ = write!(its_json, "{its:.6}{comma}");
    }
    its_json.push(']');

    let json = format!(
        "{{\n      \"detailed_balance_max_error\": {:.6e},\n      \"eigenvalue_top\": {:.6},\n      \"lag_time_ps\": {:.4},\n      \"macrostate_count\": {},\n      \"macrostate_population_sum\": {:.6},\n      \"macrostates\": {},\n      \"max_row_stochastic_error\": {:.6e},\n      \"num_microstates\": {},\n      \"pmf_grid_dims\": [{}, {}],\n      \"stationary_pi_sum\": {:.6},\n      \"top_implied_timescales_ps\": {},\n      \"total_transitions\": {}\n    }}",
        msm.detailed_balance_max_error,
        msm.eigenvalues[0],
        msm.lag_time_ps,
        msm.macrostates.len(),
        macro_pop_sum,
        macro_json,
        max_row_sum_err,
        msm.num_microstates,
        ens.pmf.nx,
        ens.pmf.ny,
        pi_sum,
        its_json,
        msm.total_transitions
    );
    write_summary_section("t3_fold_msm_reversible_kinetics", json);
}
