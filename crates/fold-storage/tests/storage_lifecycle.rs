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

use fold_physics::{
    initialize_clone_checkpoint, minimize_steric_clashes, run_generation_chunk, ChaCha8Rng,
    LangevinConfig, ProteinTopology, SanityThresholds,
};
use fold_proto::crypto::NodeIdentity;
use fold_proto::{ProjectManifest, SignedWorkUnitResult, WorkUnitId};
use fold_storage::{to_hex, CommitOutcome, FoldStore, StorageError};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nanos));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn blob_store_detects_tampering() {
    let tmp = TempDirGuard::new("beofold_blob_test");
    let store = FoldStore::open(&tmp.0).expect("open store");

    let payload = b"BFLD\x01example-compressed-payload";
    let digest = store.put_compressed_blob(payload).expect("put blob");
    let loaded = store.get_compressed_blob(&digest).expect("get blob");
    assert_eq!(loaded, payload);

    let hex = to_hex(&digest);
    let prefix = &hex[0..2];
    let filename = format!("{}.lz4", hex);
    let raw_by_hex = store
        .read_raw_blob_by_hex(prefix, &filename)
        .expect("read_raw_blob_by_hex");
    assert_eq!(raw_by_hex, payload);

    // Tamper with one byte of the stored blob file on disk
    let blob_path = tmp.0.join("blobs").join(prefix).join(&filename);
    let mut corrupted = fs::read(&blob_path).expect("read blob file");
    corrupted[0] ^= 0x5A;
    fs::write(&blob_path, &corrupted).expect("write tampered blob");

    let err = store
        .get_compressed_blob(&digest)
        .expect_err("tampered blob must fail verification");
    assert!(matches!(err, StorageError::BlobIntegrityMismatch { .. }));
}

#[test]
fn full_prcg_lease_commit_and_quarantine_lifecycle() {
    let tmp = TempDirGuard::new("beofold_lifecycle_test");
    let store = FoldStore::open(&tmp.0).expect("open store");

    let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
    minimize_steric_clashes(&topo, &mut coords, 120);

    let wu0 = WorkUnitId::new(7, 0, 0, 0);
    let wu1 = WorkUnitId::new(7, 0, 1, 0);
    let cp0 = initialize_clone_checkpoint(&topo, &coords, wu0, 300.0, b"test-campaign");
    let cp1 = initialize_clone_checkpoint(&topo, &coords, wu1, 330.0, b"test-campaign");

    let manifest = ProjectManifest {
        project_id: 7,
        name: "TrpCage Lifecycle Test".into(),
        sequence: topo.sequence_one_letter.clone(),
        topology_blake3: [0u8; 32],
        initial_coords_blake3: [0u8; 32],
        atom_count: topo.atoms.len() as u32,
        residue_count: topo.residue_count,
        temp_ladder_kelvin: vec![300.0, 330.0],
        dt_fs: 1.0,
        friction_per_ps: 1.0,
        steps_per_gen: 10,
        num_clones_per_run: 2,
        max_generations: 3,
    };

    let seeded_manifest = store
        .seed_project(manifest, &topo, &[cp0, cp1])
        .expect("seed project");
    assert_eq!(store.list_project_ids().expect("list projects"), vec![7]);

    let worker = NodeIdentity::from_cluster_secret("test-secret", "worker-1");
    let (lease, input_cp) = store
        .acquire_next_lease(worker.public_key_bytes(), 1_000, 600)
        .expect("acquire lease 1")
        .expect("expected first queued unit");

    let cfg = LangevinConfig {
        dt_fs: seeded_manifest.dt_fs,
        friction_per_ps: seeded_manifest.friction_per_ps,
        temperature_kelvin: input_cp.temperature_kelvin,
        steps: 10,
        sample_stride: 5,
    };
    let summary = run_generation_chunk(&topo, &input_cp, lease.wu_id, &cfg);
    let out_frame = summary.checkpoint.to_compressed_frame().expect("encode cp");
    let out_hash = *blake3::hash(&out_frame).as_bytes();

    let mut signed = SignedWorkUnitResult {
        wu_id: lease.wu_id,
        lease_token: lease.lease_token,
        worker_pubkey: worker.public_key_bytes(),
        input_checkpoint_blake3: lease.input_checkpoint_blake3,
        output_checkpoint_blake3: out_hash,
        steps_executed: 10,
        temperature_kelvin: summary.checkpoint.temperature_kelvin,
        mean_potential_kcal: summary.mean_potential_kcal,
        min_potential_kcal: summary.min_potential_kcal,
        final_potential_kcal: summary.checkpoint.potential_energy_kcal,
        final_kinetic_kcal: summary.checkpoint.kinetic_energy_kcal,
        final_rg_angstrom: summary.checkpoint.rg_angstrom,
        final_sasa_angstrom2: summary.checkpoint.sasa_angstrom2,
        wall_time_ms: 12,
        signature: [0u8; 64],
    };
    worker.sign_result(&mut signed);

    let mut mremd_rng = ChaCha8Rng::from_prcg_seed(lease.wu_id, b"mremd");
    let outcome = store
        .commit_signed_result(
            &seeded_manifest,
            &topo,
            &signed,
            &summary.checkpoint,
            &SanityThresholds::default(),
            &mut mremd_rng,
            1_005,
        )
        .expect("commit valid result");

    match outcome {
        CommitOutcome::Accepted {
            completed_wu,
            spawned_next_gen,
            ..
        } => {
            assert_eq!(completed_wu, lease.wu_id);
            assert_eq!(spawned_next_gen, Some(lease.wu_id.next_generation()));
        }
        other => panic!("expected CommitOutcome::Accepted, got {:?}", other),
    }

    // Lease the second unit and submit with an invalid Ed25519 signature
    let (lease2, input_cp2) = store
        .acquire_next_lease(worker.public_key_bytes(), 1_010, 600)
        .expect("acquire lease 2")
        .expect("expected second queued unit");
    let cfg2 = LangevinConfig {
        dt_fs: seeded_manifest.dt_fs,
        friction_per_ps: seeded_manifest.friction_per_ps,
        temperature_kelvin: input_cp2.temperature_kelvin,
        steps: 10,
        sample_stride: 5,
    };
    let summary2 = run_generation_chunk(&topo, &input_cp2, lease2.wu_id, &cfg2);
    let out_frame2 = summary2.checkpoint.to_compressed_frame().expect("encode cp2");
    let out_hash2 = *blake3::hash(&out_frame2).as_bytes();

    let mut bad_signed = SignedWorkUnitResult {
        wu_id: lease2.wu_id,
        lease_token: lease2.lease_token,
        worker_pubkey: worker.public_key_bytes(),
        input_checkpoint_blake3: lease2.input_checkpoint_blake3,
        output_checkpoint_blake3: out_hash2,
        steps_executed: 10,
        temperature_kelvin: summary2.checkpoint.temperature_kelvin,
        mean_potential_kcal: summary2.mean_potential_kcal,
        min_potential_kcal: summary2.min_potential_kcal,
        final_potential_kcal: summary2.checkpoint.potential_energy_kcal,
        final_kinetic_kcal: summary2.checkpoint.kinetic_energy_kcal,
        final_rg_angstrom: summary2.checkpoint.rg_angstrom,
        final_sasa_angstrom2: summary2.checkpoint.sasa_angstrom2,
        wall_time_ms: 14,
        signature: [0u8; 64],
    };
    worker.sign_result(&mut bad_signed);
    bad_signed.signature[0] ^= 0xFF;

    let bad_outcome = store
        .commit_signed_result(
            &seeded_manifest,
            &topo,
            &bad_signed,
            &summary2.checkpoint,
            &SanityThresholds::default(),
            &mut mremd_rng,
            1_015,
        )
        .expect("quarantine commit");
    assert!(matches!(bad_outcome, CommitOutcome::Quarantined { .. }));

    let stats = store.collect_stats(Some(7)).expect("collect stats");
    assert_eq!(stats.quarantined_events, 1);
    assert_eq!(stats.completed, 1);
}
