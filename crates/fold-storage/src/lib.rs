use fold_physics::{
    run_generation_chunk, try_mremd_swap, verify_checkpoint_sanity, ChaCha8Rng, LangevinConfig,
    PhysicsError, ProteinTopology, SanityThresholds,
};
use fold_proto::crypto::verify_signed_result;
use fold_proto::{
    GlobalFederationTelemetry, LanLeasedUnit, MremdEndpoint, PendingQuorumCandidate,
    ProjectManifest, ProtoError, QuarantineEntry, RaftLogCommand, RegisteredClusterRecord,
    SignedWorkUnitResult, TrajectoryCheckpoint, WanWorkUnitSubmission, WorkUnitId, WorkUnitLease,
    WorkUnitRecord, WorkUnitStatus,
};
use redb::{Database, ReadableTable, TableDefinition};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const PROJECTS_TABLE: TableDefinition<u32, &[u8]> = TableDefinition::new("projects");
const WORKUNITS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("workunits");
const LEASES_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("leases");
const QUARANTINE_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("quarantine");
const MREMD_POOL_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("mremd_pool");
const META_U64_TABLE: TableDefinition<&str, u64> = TableDefinition::new("meta_u64");
const CLUSTERS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("clusters");
const QUORUM_STAGING_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("quorum_staging");
const WAN_UPSTREAM_LEASES_TABLE: TableDefinition<&[u8], &[u8]> =
    TableDefinition::new("wan_upstream_leases");
const WAN_OUTBOX_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("wan_outbox");

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("redb database error: {0}")]
    Database(#[from] redb::DatabaseError),
    #[error("redb transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),
    #[error("redb table error: {0}")]
    Table(#[from] redb::TableError),
    #[error("redb storage error: {0}")]
    Storage(#[from] redb::StorageError),
    #[error("redb commit error: {0}")]
    Commit(#[from] redb::CommitError),
    #[error("protocol framing error: {0}")]
    Proto(#[from] ProtoError),
    #[error("physics topology error: {0}")]
    Physics(#[from] PhysicsError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("blob BLAKE3 integrity mismatch for {expected_hex}")]
    BlobIntegrityMismatch { expected_hex: String },
    #[error("project {0} not found in ledger")]
    ProjectNotFound(u32),
    #[error("work unit {0} has no matching active lease")]
    InvalidOrExpiredLease(WorkUnitId),
}

/// Outcome of validating and committing a worker's completed PRCG chunk.
#[derive(Debug, Clone)]
pub enum CommitOutcome {
    Accepted {
        completed_wu: WorkUnitId,
        spawned_next_gen: Option<WorkUnitId>,
        next_temperature_kelvin: f64,
        mremd_swapped_with: Option<WorkUnitId>,
    },
    AwaitingSecondQuorum {
        wu_id: WorkUnitId,
    },
    Quarantined {
        wu_id: WorkUnitId,
        reason: String,
    },
}

/// Aggregate ledger summary for CLI and Supervisor GUI reporting.
#[derive(Debug, Clone, Default)]
pub struct CampaignStats {
    pub total_workunits: usize,
    pub queued: usize,
    pub leased: usize,
    pub quorum_pending: usize,
    pub completed: usize,
    pub quarantined_events: u64,
    pub canary_audits_passed: u64,
    pub mremd_trials: u64,
    pub mremd_swaps: u64,
    pub lowest_potential_kcal: Option<(WorkUnitId, f64, [u8; 32])>,
}

/// Pure-Rust ACID persistence engine combining `redb` B-tree metadata tables with an
/// Apache-compatible content-addressed `blobs/<prefix>/<blake3>.lz4` trajectory store.
pub struct FoldStore {
    root_dir: PathBuf,
    blobs_dir: PathBuf,
    db: Database,
}

impl FoldStore {
    pub fn open(root_dir: &Path) -> Result<Self, StorageError> {
        let blobs_dir = root_dir.join("blobs");
        fs::create_dir_all(&blobs_dir)?;
        let db_path = root_dir.join("ledger.redb");
        let db = Database::create(&db_path)?;

        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(PROJECTS_TABLE)?;
            let _ = write_txn.open_table(WORKUNITS_TABLE)?;
            let _ = write_txn.open_table(LEASES_TABLE)?;
            let _ = write_txn.open_table(QUARANTINE_TABLE)?;
            let _ = write_txn.open_table(MREMD_POOL_TABLE)?;
            let _ = write_txn.open_table(META_U64_TABLE)?;
            let _ = write_txn.open_table(CLUSTERS_TABLE)?;
            let _ = write_txn.open_table(QUORUM_STAGING_TABLE)?;
            let _ = write_txn.open_table(WAN_UPSTREAM_LEASES_TABLE)?;
            let _ = write_txn.open_table(WAN_OUTBOX_TABLE)?;
        }
        write_txn.commit()?;

        Ok(Self {
            root_dir: root_dir.to_path_buf(),
            blobs_dir,
            db,
        })
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn blob_rel_url(hash: &[u8; 32]) -> String {
        let hex = to_hex(hash);
        format!("/blobs/{}/{}.lz4", &hex[0..2], hex)
    }

    fn blob_path(&self, hash: &[u8; 32]) -> PathBuf {
        let hex = to_hex(hash);
        self.blobs_dir.join(&hex[0..2]).join(format!("{}.lz4", hex))
    }

    pub fn read_raw_blob_by_hex(&self, prefix: &str, filename: &str) -> Result<Vec<u8>, StorageError> {
        let hex_stem = filename.strip_suffix(".lz4").unwrap_or(filename);
        if prefix.len() != 2 || hex_stem.len() != 64 || !hex_stem.starts_with(prefix) {
            return Err(StorageError::BlobIntegrityMismatch {
                expected_hex: filename.to_string(),
            });
        }
        let expected = fold_proto::hex_to_bytes_fixed::<32>(hex_stem)?;
        self.get_compressed_blob(&expected)
    }

    /// Writes a pre-compressed wire/topology frame atomically (`temp -> fsync -> rename`)
    /// into the content-addressed blob tree and returns its 32-byte BLAKE3 digest.
    pub fn put_compressed_blob(&self, frame_bytes: &[u8]) -> Result<[u8; 32], StorageError> {
        let hash = *blake3::hash(frame_bytes).as_bytes();
        let final_path = self.blob_path(&hash);
        if final_path.exists() {
            return Ok(hash);
        }
        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp_path = final_path.with_extension(format!("tmp.{}.{}", std::process::id(), nanos));
        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(frame_bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        Ok(hash)
    }

    /// Reads a content-addressed blob and verifies its BLAKE3 cryptographic digest before returning.
    pub fn get_compressed_blob(&self, expected_hash: &[u8; 32]) -> Result<Vec<u8>, StorageError> {
        let path = self.blob_path(expected_hash);
        let bytes = fs::read(&path)?;
        let actual_hash = *blake3::hash(&bytes).as_bytes();
        if &actual_hash != expected_hash {
            return Err(StorageError::BlobIntegrityMismatch {
                expected_hex: to_hex(expected_hash),
            });
        }
        Ok(bytes)
    }

    pub fn put_checkpoint(&self, cp: &TrajectoryCheckpoint) -> Result<[u8; 32], StorageError> {
        let frame = cp.to_compressed_frame()?;
        self.put_compressed_blob(&frame)
    }

    pub fn get_checkpoint(&self, hash: &[u8; 32]) -> Result<TrajectoryCheckpoint, StorageError> {
        let frame = self.get_compressed_blob(hash)?;
        Ok(TrajectoryCheckpoint::from_compressed_frame(&frame)?)
    }

    /// Registers or refreshes a remote Beowulf Cluster Leader in `CLUSTERS_TABLE`.
    pub fn upsert_cluster_registration(
        &self,
        mut record: RegisteredClusterRecord,
        force_trusted: bool,
    ) -> Result<RegisteredClusterRecord, StorageError> {
        let write_txn = self.db.begin_write()?;
        {
            let mut tbl = write_txn.open_table(CLUSTERS_TABLE)?;
            if let Some(prev_guard) = tbl.get(&record.cluster_pubkey[..])? {
                let prev = RegisteredClusterRecord::from_rkyv_bytes(prev_guard.value())?;
                record.completed_units = prev.completed_units;
                record.quarantined_units = prev.quarantined_units;
                record.trusted = prev.trusted || force_trusted;
            } else {
                record.trusted = force_trusted;
            }
            tbl.insert(&record.cluster_pubkey[..], record.to_rkyv_bytes()?.as_slice())?;
        }
        write_txn.commit()?;
        Ok(record)
    }

    pub fn is_cluster_trusted(&self, cluster_pubkey: &[u8; 32]) -> Result<bool, StorageError> {
        let read_txn = self.db.begin_read()?;
        let tbl = read_txn.open_table(CLUSTERS_TABLE)?;
        if let Some(g) = tbl.get(&cluster_pubkey[..])? {
            let rec = RegisteredClusterRecord::from_rkyv_bytes(g.value())?;
            return Ok(rec.trusted);
        }
        Ok(false)
    }

    pub fn list_registered_clusters(&self) -> Result<Vec<RegisteredClusterRecord>, StorageError> {
        let read_txn = self.db.begin_read()?;
        let tbl = read_txn.open_table(CLUSTERS_TABLE)?;
        let mut out = Vec::new();
        for item in tbl.iter()? {
            let (_, vg) = item?;
            out.push(RegisteredClusterRecord::from_rkyv_bytes(vg.value())?);
        }
        Ok(out)
    }

    fn increment_cluster_counters(
        &self,
        cluster_pubkey: &[u8; 32],
        completed_delta: u64,
        quarantined_delta: u64,
        now_unix_sec: u64,
    ) -> Result<(), StorageError> {
        let write_txn = self.db.begin_write()?;
        {
            let mut tbl = write_txn.open_table(CLUSTERS_TABLE)?;
            let updated = if let Some(g) = tbl.get(&cluster_pubkey[..])? {
                let mut rec = RegisteredClusterRecord::from_rkyv_bytes(g.value())?;
                rec.completed_units += completed_delta;
                rec.quarantined_units += quarantined_delta;
                rec.last_seen_unix_sec = now_unix_sec;
                Some(rec.to_rkyv_bytes()?)
            } else {
                None
            };
            if let Some(bytes) = updated {
                tbl.insert(&cluster_pubkey[..], bytes.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Exports `(ProjectManifest, compressed_topology_blob)` for LAN workers or WAN sync clients.
    pub fn export_project_bundle(
        &self,
        project_id: u32,
    ) -> Result<Option<(ProjectManifest, Vec<u8>)>, StorageError> {
        let read_txn = self.db.begin_read()?;
        let proj_tbl = read_txn.open_table(PROJECTS_TABLE)?;
        let Some(guard) = proj_tbl.get(project_id)? else {
            return Ok(None);
        };
        let manifest = ProjectManifest::from_rkyv_bytes(guard.value())?;
        let topo_blob = self.get_compressed_blob(&manifest.topology_blake3)?;
        Ok(Some((manifest, topo_blob)))
    }

    pub fn export_gen0_blobs(&self, project_id: u32) -> Result<Vec<Vec<u8>>, StorageError> {
        let records = self.list_workunits(project_id)?;
        let mut blobs = Vec::new();
        for rec in records {
            if rec.wu_id.generation == 0 {
                if let Ok(b) = self.get_compressed_blob(&rec.input_checkpoint_blake3) {
                    blobs.push(b);
                }
            }
        }
        Ok(blobs)
    }

    /// Imports a project manifest + compressed topology blob pulled from the Apache Web Sync Server
    /// into a local LAN Cluster Leader's `FoldStore`.
    pub fn import_wan_project_metadata(
        &self,
        manifest: &ProjectManifest,
        compressed_topology_blob: &[u8],
    ) -> Result<(), StorageError> {
        let topo_hash = self.put_compressed_blob(compressed_topology_blob)?;
        if topo_hash != manifest.topology_blake3 {
            return Err(StorageError::BlobIntegrityMismatch {
                expected_hex: to_hex(&manifest.topology_blake3),
            });
        }
        let write_txn = self.db.begin_write()?;
        {
            let mut proj_tbl = write_txn.open_table(PROJECTS_TABLE)?;
            proj_tbl.insert(manifest.project_id, manifest.to_rkyv_bytes()?.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Stages a WAN-leased `LanLeasedUnit` into the local LAN Beowulf queue and records its
    /// upstream Apache `WorkUnitLease` token in `WAN_UPSTREAM_LEASES_TABLE`.
    pub fn stage_wan_leased_unit(&self, unit: &LanLeasedUnit) -> Result<bool, StorageError> {
        let cp_hash = self.put_compressed_blob(&unit.compressed_input_checkpoint)?;
        if cp_hash != unit.lease.input_checkpoint_blake3 {
            return Err(StorageError::BlobIntegrityMismatch {
                expected_hex: to_hex(&unit.lease.input_checkpoint_blake3),
            });
        }
        let cp = TrajectoryCheckpoint::from_compressed_frame(&unit.compressed_input_checkpoint)?;
        let key = unit.lease.wu_id.to_key_bytes();

        let write_txn = self.db.begin_write()?;
        let inserted = {
            let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
            let should_stage = match wu_tbl.get(&key[..])? {
                None => true,
                Some(existing_g) => {
                    let prev = WorkUnitRecord::from_rkyv_bytes(existing_g.value())?;
                    !matches!(prev.status, WorkUnitStatus::Completed { .. })
                }
            };
            if should_stage {
                let rec = WorkUnitRecord {
                    wu_id: unit.lease.wu_id,
                    input_checkpoint_blake3: cp_hash,
                    temperature_kelvin: cp.temperature_kelvin,
                    target_steps: unit.lease.target_steps,
                    status: WorkUnitStatus::Queued,
                    final_potential_kcal: None,
                    min_potential_kcal: None,
                    final_rg_angstrom: None,
                    quarantine_history: 0,
                };
                wu_tbl.insert(&key[..], rec.to_rkyv_bytes()?.as_slice())?;
                let mut up_tbl = write_txn.open_table(WAN_UPSTREAM_LEASES_TABLE)?;
                up_tbl.insert(&key[..], unit.lease.to_rkyv_bytes()?.as_slice())?;
            }
            should_stage
        };
        write_txn.commit()?;
        Ok(inserted)
    }

    /// Lists all active upstream WAN `(WorkUnitId, lease_token)` pairs for 120-second Apache heartbeats.
    pub fn list_upstream_wan_leases(&self) -> Result<Vec<(WorkUnitId, [u8; 16])>, StorageError> {
        let read_txn = self.db.begin_read()?;
        let up_tbl = read_txn.open_table(WAN_UPSTREAM_LEASES_TABLE)?;
        let mut out = Vec::new();
        for item in up_tbl.iter()? {
            let (_, vg) = item?;
            let lease = WorkUnitLease::from_rkyv_bytes(vg.value())?;
            out.push((lease.wu_id, lease.lease_token));
        }
        Ok(out)
    }

    /// Drains queued `WanWorkUnitSubmission` items waiting in `WAN_OUTBOX_TABLE` for upload to Apache.
    pub fn peek_wan_outbox(&self, max_items: usize) -> Result<Vec<WanWorkUnitSubmission>, StorageError> {
        let read_txn = self.db.begin_read()?;
        let out_tbl = read_txn.open_table(WAN_OUTBOX_TABLE)?;
        let mut items = Vec::new();
        for entry in out_tbl.iter()? {
            if items.len() >= max_items {
                break;
            }
            let (_, vg) = entry?;
            items.push(WanWorkUnitSubmission::from_rkyv_bytes(vg.value())?);
        }
        Ok(items)
    }

    /// Removes a `WanWorkUnitSubmission` from `WAN_OUTBOX_TABLE` and `WAN_UPSTREAM_LEASES_TABLE`
    /// once the Apache sync server acknowledges receipt.
    pub fn ack_wan_outbox_uploaded(&self, wu_id: WorkUnitId) -> Result<(), StorageError> {
        let key = wu_id.to_key_bytes();
        let write_txn = self.db.begin_write()?;
        {
            let mut out_tbl = write_txn.open_table(WAN_OUTBOX_TABLE)?;
            out_tbl.remove(&key[..])?;
            let mut up_tbl = write_txn.open_table(WAN_UPSTREAM_LEASES_TABLE)?;
            up_tbl.remove(&key[..])?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Applies a replicated `RaftLogCommand` on a LAN follower so its local `FoldStore` stays hot.
    pub fn apply_replicated_raft_command(&self, cmd: &RaftLogCommand) -> Result<(), StorageError> {
        match cmd {
            RaftLogCommand::SeedProjectBundle {
                manifest,
                compressed_topology_blob,
                gen0_compressed_checkpoints,
            } => {
                let _ = self.put_compressed_blob(compressed_topology_blob)?;
                if gen0_compressed_checkpoints.is_empty() {
                    self.import_wan_project_metadata(manifest, compressed_topology_blob)?;
                } else {
                    let mut cps = Vec::with_capacity(gen0_compressed_checkpoints.len());
                    for frame in gen0_compressed_checkpoints {
                        let _ = self.put_compressed_blob(frame)?;
                        cps.push(TrajectoryCheckpoint::from_compressed_frame(frame)?);
                    }
                    let topo = ProteinTopology::from_compressed_bytes(compressed_topology_blob)?;
                    if self.load_project(manifest.project_id).is_err() {
                        let _ = self.seed_project(manifest.clone(), &topo, &cps)?;
                    }
                }
            }

            RaftLogCommand::LeaseIssued { lease } => {
                let key = lease.wu_id.to_key_bytes();
                let write_txn = self.db.begin_write()?;
                {
                    let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
                    let updated = if let Some(g) = wu_tbl.get(&key[..])? {
                        let mut rec = WorkUnitRecord::from_rkyv_bytes(g.value())?;
                        if matches!(
                            rec.status,
                            WorkUnitStatus::Queued | WorkUnitStatus::Leased { .. }
                        ) {
                            rec.status = WorkUnitStatus::Leased {
                                lease_token: lease.lease_token,
                                expires_at_unix_sec: lease.expires_at_unix_sec,
                            };
                            Some(rec.to_rkyv_bytes()?)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(bytes) = updated {
                        wu_tbl.insert(&key[..], bytes.as_slice())?;
                    }
                    let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
                    lease_tbl.insert(&key[..], lease.to_rkyv_bytes()?.as_slice())?;
                }
                write_txn.commit()?;
            }

            RaftLogCommand::GenerationCommitted {
                receipt,
                compressed_output_checkpoint,
                next_temperature_kelvin,
                spawned_next_gen,
            } => {
                let out_hash = self.put_compressed_blob(compressed_output_checkpoint)?;
                let key = receipt.wu_id.to_key_bytes();
                let write_txn = self.db.begin_write()?;
                {
                    let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
                    let existing = wu_tbl
                        .get(&key[..])?
                        .map(|g| WorkUnitRecord::from_rkyv_bytes(g.value()))
                        .transpose()?;
                    let target_steps = if let Some(mut rec) = existing {
                        rec.status = WorkUnitStatus::Completed {
                            output_checkpoint_blake3: out_hash,
                        };
                        rec.final_potential_kcal = Some(receipt.final_potential_kcal);
                        rec.min_potential_kcal = Some(receipt.min_potential_kcal);
                        rec.final_rg_angstrom = Some(receipt.final_rg_angstrom);
                        let ts = rec.target_steps;
                        wu_tbl.insert(&key[..], rec.to_rkyv_bytes()?.as_slice())?;
                        ts
                    } else {
                        receipt.steps_executed
                    };

                    let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
                    lease_tbl.remove(&key[..])?;

                    if let Some(next_id) = spawned_next_gen {
                        let next_key = next_id.to_key_bytes();
                        if wu_tbl.get(&next_key[..])?.is_none() {
                            let next_rec = WorkUnitRecord {
                                wu_id: *next_id,
                                input_checkpoint_blake3: out_hash,
                                temperature_kelvin: *next_temperature_kelvin,
                                target_steps,
                                status: WorkUnitStatus::Queued,
                                final_potential_kcal: None,
                                min_potential_kcal: None,
                                final_rg_angstrom: None,
                                quarantine_history: 0,
                            };
                            wu_tbl.insert(&next_key[..], next_rec.to_rkyv_bytes()?.as_slice())?;
                        }
                    }
                }
                write_txn.commit()?;
            }

            RaftLogCommand::GenerationQuarantined {
                wu_id,
                worker_pubkey,
                reason,
                timestamp_unix_sec,
            } => {
                let _ = self.quarantine_and_requeue(
                    *wu_id,
                    *worker_pubkey,
                    reason.clone(),
                    None,
                    *timestamp_unix_sec,
                )?;
            }
        }
        Ok(())
    }

    pub fn seed_project(
        &self,
        mut manifest: ProjectManifest,
        topology: &ProteinTopology,
        gen0_checkpoints: &[TrajectoryCheckpoint],
    ) -> Result<ProjectManifest, StorageError> {
        let topo_bytes = topology.to_compressed_bytes()?;
        let topo_hash = self.put_compressed_blob(&topo_bytes)?;
        manifest.topology_blake3 = topo_hash;

        let mut seeded_records = Vec::with_capacity(gen0_checkpoints.len());
        for cp in gen0_checkpoints {
            let cp_hash = self.put_checkpoint(cp)?;
            if manifest.initial_coords_blake3 == [0u8; 32] {
                manifest.initial_coords_blake3 = cp_hash;
            }
            seeded_records.push(WorkUnitRecord {
                wu_id: cp.wu_id,
                input_checkpoint_blake3: cp_hash,
                temperature_kelvin: cp.temperature_kelvin,
                target_steps: manifest.steps_per_gen,
                status: WorkUnitStatus::Queued,
                final_potential_kcal: None,
                min_potential_kcal: None,
                final_rg_angstrom: None,
                quarantine_history: 0,
            });
        }

        let manifest_bytes = manifest.to_rkyv_bytes()?;
        let write_txn = self.db.begin_write()?;
        {
            let mut proj_tbl = write_txn.open_table(PROJECTS_TABLE)?;
            proj_tbl.insert(manifest.project_id, manifest_bytes.as_slice())?;

            let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
            for rec in &seeded_records {
                let key = rec.wu_id.to_key_bytes();
                let val = rec.to_rkyv_bytes()?;
                wu_tbl.insert(&key[..], val.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(manifest)
    }

    pub fn load_project(
        &self,
        project_id: u32,
    ) -> Result<(ProjectManifest, ProteinTopology), StorageError> {
        let read_txn = self.db.begin_read()?;
        let proj_tbl = read_txn.open_table(PROJECTS_TABLE)?;
        let guard = proj_tbl
            .get(project_id)?
            .ok_or(StorageError::ProjectNotFound(project_id))?;
        let manifest = ProjectManifest::from_rkyv_bytes(guard.value())?;
        let topo_frame = self.get_compressed_blob(&manifest.topology_blake3)?;
        let topology = ProteinTopology::from_compressed_bytes(&topo_frame)?;
        Ok((manifest, topology))
    }

    pub fn list_project_ids(&self) -> Result<Vec<u32>, StorageError> {
        let read_txn = self.db.begin_read()?;
        let proj_tbl = read_txn.open_table(PROJECTS_TABLE)?;
        let mut ids = Vec::new();
        for item in proj_tbl.iter()? {
            let (k, _) = item?;
            ids.push(k.value());
        }
        Ok(ids)
    }

    /// Atomically leases the next available `Queued` (or expired `Leased`) work unit, optionally
    /// excluding a specific `cluster_pubkey` so `quorum = 2` verification runs on an independent cluster.
    pub fn acquire_next_lease_filtered(
        &self,
        assignee_pubkey: [u8; 32],
        exclude_pubkey: Option<[u8; 32]>,
        now_unix_sec: u64,
        lease_ttl_sec: u64,
    ) -> Result<Option<(WorkUnitLease, TrajectoryCheckpoint)>, StorageError> {
        let write_txn = self.db.begin_write()?;
        let mut selected: Option<WorkUnitRecord> = None;

        {
            let wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
            let q_stage = write_txn.open_table(QUORUM_STAGING_TABLE)?;
            for entry in wu_tbl.iter()? {
                let (k_guard, val_guard) = entry?;
                let rec = WorkUnitRecord::from_rkyv_bytes(val_guard.value())?;
                let claimable = match &rec.status {
                    WorkUnitStatus::Queued => true,
                    WorkUnitStatus::Leased {
                        expires_at_unix_sec,
                        ..
                    } => now_unix_sec >= *expires_at_unix_sec,
                    _ => false,
                };
                if !claimable {
                    continue;
                }
                // In quorum=2 mode, ensure second verification lease goes to a distinct cluster
                if let Some(excl) = exclude_pubkey {
                    if let Some(cand_g) = q_stage.get(k_guard.value())? {
                        let cand = PendingQuorumCandidate::from_rkyv_bytes(cand_g.value())?;
                        if cand.first_cluster_pubkey == excl {
                            continue;
                        }
                    }
                }
                selected = Some(rec);
                break;
            }
        }

        let Some(mut rec) = selected else {
            return Ok(None);
        };

        let mut lease_token = [0u8; 16];
        getrandom::getrandom(&mut lease_token).unwrap_or_default();
        let expires_at_unix_sec = now_unix_sec.saturating_add(lease_ttl_sec);

        let lease = WorkUnitLease {
            wu_id: rec.wu_id,
            lease_token,
            assigned_node: assignee_pubkey,
            target_steps: rec.target_steps,
            leased_at_unix_sec: now_unix_sec,
            expires_at_unix_sec,
            input_checkpoint_blake3: rec.input_checkpoint_blake3,
        };

        rec.status = WorkUnitStatus::Leased {
            lease_token,
            expires_at_unix_sec,
        };

        {
            let key = rec.wu_id.to_key_bytes();
            let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
            wu_tbl.insert(&key[..], rec.to_rkyv_bytes()?.as_slice())?;

            let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
            lease_tbl.insert(&key[..], lease.to_rkyv_bytes()?.as_slice())?;
        }
        write_txn.commit()?;

        let input_cp = self.get_checkpoint(&lease.input_checkpoint_blake3)?;
        Ok(Some((lease, input_cp)))
    }

    pub fn acquire_next_lease(
        &self,
        worker_pubkey: [u8; 32],
        now_unix_sec: u64,
        lease_ttl_sec: u64,
    ) -> Result<Option<(WorkUnitLease, TrajectoryCheckpoint)>, StorageError> {
        self.acquire_next_lease_filtered(worker_pubkey, None, now_unix_sec, lease_ttl_sec)
    }

    /// Renews the expiration deadline on an active lease (verifying token when supplied).
    pub fn touch_lease_checked(
        &self,
        wu_id: WorkUnitId,
        expected_token: Option<[u8; 16]>,
        extend_to_unix_sec: u64,
    ) -> Result<bool, StorageError> {
        let key = wu_id.to_key_bytes();
        let write_txn = self.db.begin_write()?;
        let mut renewed = false;
        {
            let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
            let updated_lease = if let Some(lg) = lease_tbl.get(&key[..])? {
                let mut l = WorkUnitLease::from_rkyv_bytes(lg.value())?;
                if expected_token.map_or(true, |tok| tok == l.lease_token) {
                    l.expires_at_unix_sec = l.expires_at_unix_sec.max(extend_to_unix_sec);
                    renewed = true;
                    Some(l.to_rkyv_bytes()?)
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(bytes) = updated_lease {
                lease_tbl.insert(&key[..], bytes.as_slice())?;
            }

            if renewed {
                let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
                let updated_wu = if let Some(wg) = wu_tbl.get(&key[..])? {
                    let mut rec = WorkUnitRecord::from_rkyv_bytes(wg.value())?;
                    if let WorkUnitStatus::Leased {
                        lease_token,
                        expires_at_unix_sec,
                    } = rec.status
                    {
                        rec.status = WorkUnitStatus::Leased {
                            lease_token,
                            expires_at_unix_sec: expires_at_unix_sec.max(extend_to_unix_sec),
                        };
                        Some(rec.to_rkyv_bytes()?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(bytes) = updated_wu {
                    wu_tbl.insert(&key[..], bytes.as_slice())?;
                }
            }
        }
        write_txn.commit()?;
        Ok(renewed)
    }

    pub fn touch_lease(&self, wu_id: WorkUnitId, extend_to_unix_sec: u64) -> Result<(), StorageError> {
        let _ = self.touch_lease_checked(wu_id, None, extend_to_unix_sec)?;
        Ok(())
    }

    /// Commits a signed work unit in local/LAN mode (`quorum = 1`). If the completed unit was
    /// staged from an upstream WAN lease (`WAN_UPSTREAM_LEASES_TABLE`), queues a
    /// `WanWorkUnitSubmission` in `WAN_OUTBOX_TABLE` with the upstream Apache `lease_token` and
    /// skips local `Gen G+1` spawning so the global Apache server remains the single PRCG authority.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_signed_result(
        &self,
        manifest: &ProjectManifest,
        topology: &ProteinTopology,
        result: &SignedWorkUnitResult,
        output_checkpoint: &TrajectoryCheckpoint,
        sanity_limits: &SanityThresholds,
        mremd_rng: &mut ChaCha8Rng,
        now_unix_sec: u64,
    ) -> Result<CommitOutcome, StorageError> {
        self.commit_signed_result_with_policy(
            manifest,
            topology,
            result,
            output_checkpoint,
            sanity_limits,
            mremd_rng,
            now_unix_sec,
            None,
            1,
            false,
        )
    }

    /// Full 4-Gate Validation + Adaptive Quorum (`quorum = 1|2`) + Canary Spot-Check commit engine.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_signed_result_with_policy(
        &self,
        manifest: &ProjectManifest,
        topology: &ProteinTopology,
        result: &SignedWorkUnitResult,
        output_checkpoint: &TrajectoryCheckpoint,
        sanity_limits: &SanityThresholds,
        mremd_rng: &mut ChaCha8Rng,
        now_unix_sec: u64,
        wan_cluster_pubkey: Option<[u8; 32]>,
        required_quorum: u8,
        run_canary_spot_check: bool,
    ) -> Result<CommitOutcome, StorageError> {
        if let Err(sig_err) = verify_signed_result(result) {
            if let Some(cpk) = wan_cluster_pubkey {
                let _ = self.increment_cluster_counters(&cpk, 0, 1, now_unix_sec);
            }
            return self.quarantine_and_requeue(
                result.wu_id,
                result.worker_pubkey,
                format!("Gate 1 Ed25519 signature failure: {}", sig_err),
                None,
                now_unix_sec,
            );
        }

        if let Err(sanity_err) =
            verify_checkpoint_sanity(topology, output_checkpoint, sanity_limits)
        {
            if let Some(cpk) = wan_cluster_pubkey {
                let _ = self.increment_cluster_counters(&cpk, 0, 1, now_unix_sec);
            }
            let bad_hash = self.put_checkpoint(output_checkpoint).ok();
            return self.quarantine_and_requeue(
                result.wu_id,
                result.worker_pubkey,
                format!("Gate 2 biophysical sanity failure: {}", sanity_err),
                bad_hash,
                now_unix_sec,
            );
        }

        let out_hash = self.put_checkpoint(output_checkpoint)?;
        if out_hash != result.output_checkpoint_blake3 {
            if let Some(cpk) = wan_cluster_pubkey {
                let _ = self.increment_cluster_counters(&cpk, 0, 1, now_unix_sec);
            }
            return self.quarantine_and_requeue(
                result.wu_id,
                result.worker_pubkey,
                "Gate 1 BLAKE3 digest mismatch between receipt and checkpoint blob".to_string(),
                Some(out_hash),
                now_unix_sec,
            );
        }

        // Gate 3A: Deterministic CPU Canary Micro-Rollout (20-step deterministic trajectory audit)
        if run_canary_spot_check {
            if let Ok(input_cp) = self.get_checkpoint(&result.input_checkpoint_blake3) {
                let canary_steps = result.steps_executed.min(20);
                let canary_cfg = LangevinConfig {
                    dt_fs: manifest.dt_fs,
                    friction_per_ps: manifest.friction_per_ps,
                    temperature_kelvin: result.temperature_kelvin,
                    steps: canary_steps,
                    sample_stride: canary_steps.max(1),
                };
                let canary_out =
                    run_generation_chunk(topology, &input_cp, result.wu_id, &canary_cfg);
                if verify_checkpoint_sanity(topology, &canary_out.checkpoint, sanity_limits)
                    .is_err()
                {
                    return self.quarantine_and_requeue(
                        result.wu_id,
                        result.worker_pubkey,
                        "Gate 3 deterministic canary rollout detected unstable starting state".into(),
                        Some(out_hash),
                        now_unix_sec,
                    );
                }
                // When chunk length <= 20 steps, verify exact deterministic match!
                if result.steps_executed == canary_steps
                    && (canary_out.checkpoint.potential_energy_kcal - result.final_potential_kcal)
                        .abs()
                        > 0.05
                {
                    if let Some(cpk) = wan_cluster_pubkey {
                        let _ = self.increment_cluster_counters(&cpk, 0, 1, now_unix_sec);
                    }
                    return self.quarantine_and_requeue(
                        result.wu_id,
                        result.worker_pubkey,
                        "Gate 3 canary replay divergence (tampered or bit-flipped trajectory)"
                            .into(),
                        Some(out_hash),
                        now_unix_sec,
                    );
                }
            }
        }

        let key = result.wu_id.to_key_bytes();
        let write_txn = self.db.begin_write()?;

        let (mut current_rec, has_upstream_wan_lease) = {
            let lease_tbl = write_txn.open_table(LEASES_TABLE)?;
            let Some(lease_guard) = lease_tbl.get(&key[..])? else {
                return Err(StorageError::InvalidOrExpiredLease(result.wu_id));
            };
            let active_lease = WorkUnitLease::from_rkyv_bytes(lease_guard.value())?;
            if active_lease.lease_token != result.lease_token {
                return Err(StorageError::InvalidOrExpiredLease(result.wu_id));
            }

            let wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
            let wu_guard = wu_tbl
                .get(&key[..])?
                .ok_or(StorageError::InvalidOrExpiredLease(result.wu_id))?;
            let rec = WorkUnitRecord::from_rkyv_bytes(wu_guard.value())?;

            let up_tbl = write_txn.open_table(WAN_UPSTREAM_LEASES_TABLE)?;
            let upstream_opt = up_tbl
                .get(&key[..])?
                .map(|g| WorkUnitLease::from_rkyv_bytes(g.value()))
                .transpose()?;
            (rec, upstream_opt)
        };

        // Gate 3B: Untrusted `quorum >= 2` cross-cluster consensus check
        let mut winning_hash = out_hash;
        let mut winning_pot = result.final_potential_kcal;
        let mut winning_min = result.min_potential_kcal;
        let mut winning_rg = result.final_rg_angstrom;
        let mut partner_cluster_to_credit: Option<[u8; 32]> = None;

        if required_quorum >= 2 {
            let submitter_cluster = wan_cluster_pubkey.unwrap_or(result.worker_pubkey);
            let mut q_tbl = write_txn.open_table(QUORUM_STAGING_TABLE)?;
            let existing_cand = q_tbl
                .get(&key[..])?
                .map(|g| PendingQuorumCandidate::from_rkyv_bytes(g.value()))
                .transpose()?;

            match existing_cand {
                None => {
                    // Stage first candidate and re-queue work unit for an independent second cluster!
                    let cand = PendingQuorumCandidate {
                        wu_id: result.wu_id,
                        first_cluster_pubkey: submitter_cluster,
                        first_receipt: result.clone(),
                        first_checkpoint_blake3: out_hash,
                    };
                    q_tbl.insert(&key[..], cand.to_rkyv_bytes()?.as_slice())?;
                    current_rec.status = WorkUnitStatus::Queued;

                    let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
                    wu_tbl.insert(&key[..], current_rec.to_rkyv_bytes()?.as_slice())?;
                    let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
                    lease_tbl.remove(&key[..])?;
                    drop(lease_tbl);
                    drop(wu_tbl);
                    drop(q_tbl);
                    write_txn.commit()?;
                    return Ok(CommitOutcome::AwaitingSecondQuorum {
                        wu_id: result.wu_id,
                    });
                }
                Some(first) => {
                    q_tbl.remove(&key[..])?;
                    let n_atoms = (topology.atoms.len().max(1)) as f64;
                    let pot_diff_per_atom =
                        (first.first_receipt.final_potential_kcal - result.final_potential_kcal)
                            .abs()
                            / n_atoms;
                    let rg_diff =
                        (first.first_receipt.final_rg_angstrom - result.final_rg_angstrom).abs();

                    if pot_diff_per_atom > 2.5 || rg_diff > 6.0 {
                        // Disagreement -> clear staging and re-queue for fresh quorum pair
                        current_rec.status = WorkUnitStatus::Queued;
                        let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
                        wu_tbl.insert(&key[..], current_rec.to_rkyv_bytes()?.as_slice())?;
                        let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
                        lease_tbl.remove(&key[..])?;
                        drop(lease_tbl);
                        drop(wu_tbl);
                        drop(q_tbl);
                        write_txn.commit()?;
                        return Ok(CommitOutcome::Quarantined {
                            wu_id: result.wu_id,
                            reason: format!(
                                "Gate 3 quorum mismatch (dE/atom={:.2} kcal/mol, dRg={:.2} A)",
                                pot_diff_per_atom, rg_diff
                            ),
                        });
                    }

                    partner_cluster_to_credit = Some(first.first_cluster_pubkey);
                    if first.first_receipt.final_potential_kcal < result.final_potential_kcal {
                        winning_hash = first.first_checkpoint_blake3;
                        winning_pot = first.first_receipt.final_potential_kcal;
                        winning_min = first.first_receipt.min_potential_kcal;
                        winning_rg = first.first_receipt.final_rg_angstrom;
                    }
                }
            }
        }

        current_rec.status = WorkUnitStatus::Completed {
            output_checkpoint_blake3: winning_hash,
        };
        current_rec.final_potential_kcal = Some(winning_pot);
        current_rec.min_potential_kcal = Some(winning_min);
        current_rec.final_rg_angstrom = Some(winning_rg);

        let is_federated_lan_staging = has_upstream_wan_lease.is_some();

        let rung_idx = nearest_rung_index(&manifest.temp_ladder_kelvin, result.temperature_kelvin);
        let this_ep = MremdEndpoint {
            wu_id: result.wu_id,
            rung_index: rung_idx,
            temperature_kelvin: result.temperature_kelvin,
            potential_energy_kcal: winning_pot,
            output_checkpoint_blake3: winning_hash,
        };

        let mut next_temp = result.temperature_kelvin;
        let mut swapped_partner: Option<(MremdEndpoint, f64)> = None;
        let mut did_trial = false;

        if !is_federated_lan_staging && manifest.temp_ladder_kelvin.len() > 1 {
            let mremd_tbl = write_txn.open_table(MREMD_POOL_TABLE)?;
            for entry in mremd_tbl.iter()? {
                let (_, v_guard) = entry?;
                let candidate = MremdEndpoint::from_rkyv_bytes(v_guard.value())?;
                if candidate.wu_id.project == result.wu_id.project
                    && candidate.wu_id.generation == result.wu_id.generation
                    && candidate.rung_index.abs_diff(rung_idx) == 1
                {
                    did_trial = true;
                    let (accepted, new_this_t, new_cand_t) =
                        try_mremd_swap(&this_ep, &candidate, mremd_rng);
                    if accepted {
                        next_temp = new_this_t;
                        swapped_partner = Some((candidate, new_cand_t));
                    } else {
                        let cand_temp = candidate.temperature_kelvin;
                        swapped_partner = Some((candidate, cand_temp));
                    }
                    break;
                }
            }
        }

        let mut mremd_swapped_with = None;
        let spawned_next_gen = {
            let mut mremd_tbl = write_txn.open_table(MREMD_POOL_TABLE)?;
            let mut meta_tbl = write_txn.open_table(META_U64_TABLE)?;
            let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;

            if run_canary_spot_check {
                let prev_c = meta_tbl
                    .get("canary_audits_passed")?
                    .map(|g| g.value())
                    .unwrap_or(0);
                meta_tbl.insert("canary_audits_passed", prev_c + 1)?;
            }

            if did_trial {
                let prev_trials = meta_tbl
                    .get("mremd_trials")?
                    .map(|g| g.value())
                    .unwrap_or(0);
                meta_tbl.insert("mremd_trials", prev_trials + 1)?;
            }

            if let Some((partner, partner_next_t)) = swapped_partner {
                let p_key = partner.wu_id.to_key_bytes();
                mremd_tbl.remove(&p_key[..])?;

                if (partner_next_t - partner.temperature_kelvin).abs() > 1e-3 {
                    mremd_swapped_with = Some(partner.wu_id);
                    let prev_swaps = meta_tbl.get("mremd_swaps")?.map(|g| g.value()).unwrap_or(0);
                    meta_tbl.insert("mremd_swaps", prev_swaps + 1)?;

                    let partner_next_id = partner.wu_id.next_generation();
                    let pn_key = partner_next_id.to_key_bytes();
                    let updated_partner_bytes = if let Some(pn_guard) = wu_tbl.get(&pn_key[..])? {
                        let mut pn_rec = WorkUnitRecord::from_rkyv_bytes(pn_guard.value())?;
                        if matches!(pn_rec.status, WorkUnitStatus::Queued) {
                            pn_rec.temperature_kelvin = partner_next_t;
                            Some(pn_rec.to_rkyv_bytes()?)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(bytes) = updated_partner_bytes {
                        wu_tbl.insert(&pn_key[..], bytes.as_slice())?;
                    }
                }
            } else if !is_federated_lan_staging && manifest.temp_ladder_kelvin.len() > 1 {
                mremd_tbl.insert(&key[..], this_ep.to_rkyv_bytes()?.as_slice())?;
            }

            wu_tbl.insert(&key[..], current_rec.to_rkyv_bytes()?.as_slice())?;
            let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
            lease_tbl.remove(&key[..])?;

            // If this unit came from an upstream Apache WAN lease, stage it in WAN_OUTBOX_TABLE
            // using the upstream WAN lease token!
            if let Some(up_lease) = has_upstream_wan_lease {
                let mut upstream_receipt = result.clone();
                upstream_receipt.lease_token = up_lease.lease_token;
                let cp_frame = output_checkpoint.to_compressed_frame()?;
                let outbox_entry = WanWorkUnitSubmission {
                    cluster_pubkey: up_lease.assigned_node,
                    receipt: upstream_receipt,
                    compressed_checkpoint_blob: cp_frame,
                };
                let mut out_tbl = write_txn.open_table(WAN_OUTBOX_TABLE)?;
                out_tbl.insert(&key[..], outbox_entry.to_rkyv_bytes()?.as_slice())?;
                None
            } else if result.wu_id.generation + 1 < manifest.max_generations {
                let next_id = result.wu_id.next_generation();
                let next_rec = WorkUnitRecord {
                    wu_id: next_id,
                    input_checkpoint_blake3: winning_hash,
                    temperature_kelvin: next_temp,
                    target_steps: manifest.steps_per_gen,
                    status: WorkUnitStatus::Queued,
                    final_potential_kcal: None,
                    min_potential_kcal: None,
                    final_rg_angstrom: None,
                    quarantine_history: 0,
                };
                let next_key = next_id.to_key_bytes();
                wu_tbl.insert(&next_key[..], next_rec.to_rkyv_bytes()?.as_slice())?;
                Some(next_id)
            } else {
                None
            }
        };

        write_txn.commit()?;

        if let Some(cpk) = wan_cluster_pubkey {
            let _ = self.increment_cluster_counters(&cpk, 1, 0, now_unix_sec);
        }
        if let Some(p_cpk) = partner_cluster_to_credit {
            let _ = self.increment_cluster_counters(&p_cpk, 1, 0, now_unix_sec);
        }

        Ok(CommitOutcome::Accepted {
            completed_wu: result.wu_id,
            spawned_next_gen,
            next_temperature_kelvin: next_temp,
            mremd_swapped_with,
        })
    }

    pub fn quarantine_and_requeue(
        &self,
        wu_id: WorkUnitId,
        worker_pubkey: [u8; 32],
        reason: String,
        offending_blob: Option<[u8; 32]>,
        now_unix_sec: u64,
    ) -> Result<CommitOutcome, StorageError> {
        let key = wu_id.to_key_bytes();
        let entry = QuarantineEntry {
            wu_id,
            worker_pubkey,
            reason: reason.clone(),
            timestamp_unix_sec: now_unix_sec,
            offending_checkpoint_blake3: offending_blob,
        };

        let write_txn = self.db.begin_write()?;
        {
            let mut meta_tbl = write_txn.open_table(META_U64_TABLE)?;
            let seq = meta_tbl
                .get("quarantine_seq")?
                .map(|g| g.value())
                .unwrap_or(0)
                + 1;
            meta_tbl.insert("quarantine_seq", seq)?;

            let mut q_tbl = write_txn.open_table(QUARANTINE_TABLE)?;
            q_tbl.insert(seq, entry.to_rkyv_bytes()?.as_slice())?;

            let mut lease_tbl = write_txn.open_table(LEASES_TABLE)?;
            lease_tbl.remove(&key[..])?;

            let mut wu_tbl = write_txn.open_table(WORKUNITS_TABLE)?;
            let updated_bytes = if let Some(guard) = wu_tbl.get(&key[..])? {
                let mut rec = WorkUnitRecord::from_rkyv_bytes(guard.value())?;
                rec.quarantine_history += 1;
                if rec.quarantine_history >= 3 {
                    rec.status = WorkUnitStatus::Quarantined {
                        reason: reason.clone(),
                        failed_attempts: rec.quarantine_history,
                    };
                } else {
                    rec.status = WorkUnitStatus::Queued;
                }
                Some(rec.to_rkyv_bytes()?)
            } else {
                None
            };
            if let Some(bytes) = updated_bytes {
                wu_tbl.insert(&key[..], bytes.as_slice())?;
            }
        }
        write_txn.commit()?;

        Ok(CommitOutcome::Quarantined { wu_id, reason })
    }

    pub fn collect_stats(&self, project_id: Option<u32>) -> Result<CampaignStats, StorageError> {
        let read_txn = self.db.begin_read()?;
        let wu_tbl = read_txn.open_table(WORKUNITS_TABLE)?;
        let q_stage = read_txn.open_table(QUORUM_STAGING_TABLE)?;
        let meta_tbl = read_txn.open_table(META_U64_TABLE)?;

        let mut quorum_pending = 0usize;
        for entry in q_stage.iter()? {
            let _ = entry?;
            quorum_pending += 1;
        }

        let mut stats = CampaignStats {
            quorum_pending,
            quarantined_events: meta_tbl
                .get("quarantine_seq")?
                .map(|g| g.value())
                .unwrap_or(0),
            canary_audits_passed: meta_tbl
                .get("canary_audits_passed")?
                .map(|g| g.value())
                .unwrap_or(0),
            mremd_trials: meta_tbl
                .get("mremd_trials")?
                .map(|g| g.value())
                .unwrap_or(0),
            mremd_swaps: meta_tbl.get("mremd_swaps")?.map(|g| g.value())
                .unwrap_or(0),
            ..Default::default()
        };

        for item in wu_tbl.iter()? {
            let (_, v_guard) = item?;
            let rec = WorkUnitRecord::from_rkyv_bytes(v_guard.value())?;
            if let Some(pid) = project_id {
                if rec.wu_id.project != pid {
                    continue;
                }
            }
            stats.total_workunits += 1;
            match rec.status {
                WorkUnitStatus::Queued => stats.queued += 1,
                WorkUnitStatus::Leased { .. } => stats.leased += 1,
                WorkUnitStatus::Completed {
                    output_checkpoint_blake3,
                } => {
                    stats.completed += 1;
                    if let Some(e_pot) = rec.final_potential_kcal {
                        let is_better = match stats.lowest_potential_kcal {
                            None => true,
                            Some((_, best_e, _)) => e_pot < best_e,
                        };
                        if is_better {
                            stats.lowest_potential_kcal =
                                Some((rec.wu_id, e_pot, output_checkpoint_blake3));
                        }
                    }
                }
                WorkUnitStatus::Quarantined { .. } => {}
            }
        }

        Ok(stats)
    }

    pub fn build_global_telemetry(
        &self,
        quorum_mode: u8,
        now_unix_sec: u64,
    ) -> Result<GlobalFederationTelemetry, StorageError> {
        let clusters = self.list_registered_clusters()?;
        let aggregate_ns_per_day: f64 = clusters
            .iter()
            .filter(|c| now_unix_sec.saturating_sub(c.last_seen_unix_sec) <= 300)
            .map(|c| c.aggregate_ns_per_day)
            .sum();
        let active_projects = self.list_project_ids()?;
        let stats = self.collect_stats(None)?;

        Ok(GlobalFederationTelemetry {
            server_unix_sec: now_unix_sec,
            quorum_mode,
            clusters,
            aggregate_ns_per_day,
            active_projects,
            total_workunits: stats.total_workunits as u64,
            queued_workunits: stats.queued as u64,
            leased_workunits: stats.leased as u64,
            quorum_pending_workunits: stats.quorum_pending as u64,
            completed_workunits: stats.completed as u64,
            mremd_swaps: stats.mremd_swaps,
            mremd_trials: stats.mremd_trials,
            quarantined_events: stats.quarantined_events,
            canary_audits_passed: stats.canary_audits_passed,
            lowest_potential_kcal: stats.lowest_potential_kcal.map(|(wu, e, _)| (wu, e)),
        })
    }

    pub fn list_workunits(
        &self,
        project_id: u32,
    ) -> Result<Vec<WorkUnitRecord>, StorageError> {
        let read_txn = self.db.begin_read()?;
        let wu_tbl = read_txn.open_table(WORKUNITS_TABLE)?;
        let mut out = Vec::new();
        for item in wu_tbl.iter()? {
            let (_, v_guard) = item?;
            let rec = WorkUnitRecord::from_rkyv_bytes(v_guard.value())?;
            if rec.wu_id.project == project_id {
                out.push(rec);
            }
        }
        Ok(out)
    }
}

fn nearest_rung_index(ladder: &[f64], temp: f64) -> u32 {
    let mut best_idx = 0usize;
    let mut best_diff = f64::INFINITY;
    for (i, &t) in ladder.iter().enumerate() {
        let diff = (t - temp).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }
    best_idx as u32
}

pub fn to_hex(bytes: &[u8]) -> String {
    fold_proto::bytes_to_hex(bytes)
}
