use crate::prcg::{NodeCapabilityProfile, WorkUnitId};
use crate::WIRE_MAGIC;
use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProtoError {
    #[error("wire frame too short or missing BFLDv1 header")]
    InvalidHeader,
    #[error("LZ4 decompression failed: {0}")]
    Decompress(#[from] lz4_flex::block::DecompressError),
    #[error("rkyv serialization/validation error: {0}")]
    Rkyv(String),
    #[error("cryptographic signature verification failed: {0}")]
    Signature(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Static metadata for an active folding campaign.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProjectManifest {
    pub project_id: u32,
    pub name: String,
    pub sequence: String,
    pub topology_blake3: [u8; 32],
    pub initial_coords_blake3: [u8; 32],
    pub atom_count: u32,
    pub residue_count: u32,
    pub temp_ladder_kelvin: Vec<f64>,
    pub dt_fs: f64,
    pub friction_per_ps: f64,
    pub steps_per_gen: u64,
    pub num_clones_per_run: u32,
    pub max_generations: u32,
}

/// Complete, restart-exact phase-space snapshot at the boundary of a PRCG Generation.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrajectoryCheckpoint {
    pub wu_id: WorkUnitId,
    pub cumulative_steps: u64,
    pub temperature_kelvin: f64,
    pub rng_state: [u32; 16],
    pub rng_counter: u64,
    pub positions_angstrom: Vec<[f32; 3]>,
    pub velocities_ang_per_ps: Vec<[f32; 3]>,
    pub potential_energy_kcal: f64,
    pub kinetic_energy_kcal: f64,
    pub rg_angstrom: f64,
    pub sasa_angstrom2: f64,
}

/// Active work-unit lease token issued by a Cluster Leader or Web Sync Server.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WorkUnitLease {
    pub wu_id: WorkUnitId,
    pub lease_token: [u8; 16],
    pub assigned_node: [u8; 32],
    pub target_steps: u64,
    pub leased_at_unix_sec: u64,
    pub expires_at_unix_sec: u64,
    pub input_checkpoint_blake3: [u8; 32],
}

/// Worker-signed completion receipt accompanying an uploaded TrajectoryCheckpoint blob.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SignedWorkUnitResult {
    pub wu_id: WorkUnitId,
    pub lease_token: [u8; 16],
    pub worker_pubkey: [u8; 32],
    pub input_checkpoint_blake3: [u8; 32],
    pub output_checkpoint_blake3: [u8; 32],
    pub steps_executed: u64,
    pub temperature_kelvin: f64,
    pub mean_potential_kcal: f64,
    pub min_potential_kcal: f64,
    pub final_potential_kcal: f64,
    pub final_kinetic_kcal: f64,
    pub final_rg_angstrom: f64,
    pub final_sasa_angstrom2: f64,
    pub wall_time_ms: u64,
    pub signature: [u8; 64],
}

/// Lifecycle state of a single `(P, R, C, G)` work unit in the `redb` ledger.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum WorkUnitStatus {
    Queued,
    Leased {
        lease_token: [u8; 16],
        expires_at_unix_sec: u64,
    },
    Completed {
        output_checkpoint_blake3: [u8; 32],
    },
    Quarantined {
        reason: String,
        failed_attempts: u32,
    },
}

/// Persistent ledger row stored per `(P, R, C, G)` work unit.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WorkUnitRecord {
    pub wu_id: WorkUnitId,
    pub input_checkpoint_blake3: [u8; 32],
    pub temperature_kelvin: f64,
    pub target_steps: u64,
    pub status: WorkUnitStatus,
    pub final_potential_kcal: Option<f64>,
    pub min_potential_kcal: Option<f64>,
    pub final_rg_angstrom: Option<f64>,
    pub quarantine_history: u32,
}

/// Diagnostic quarantine entry retained for audit without polluting the MSM ensemble.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct QuarantineEntry {
    pub wu_id: WorkUnitId,
    pub worker_pubkey: [u8; 32],
    pub reason: String,
    pub timestamp_unix_sec: u64,
    pub offending_checkpoint_blake3: Option<[u8; 32]>,
}

/// Waiting replica endpoint at the end of a generation ready for an MREMD Metropolis swap trial.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MremdEndpoint {
    pub wu_id: WorkUnitId,
    pub rung_index: u32,
    pub temperature_kelvin: f64,
    pub potential_energy_kcal: f64,
    pub output_checkpoint_blake3: [u8; 32],
}

macro_rules! impl_wire_codec {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $ty {
                /// Serializes to zero-copy `rkyv` bytes (uncompressed).
                pub fn to_rkyv_bytes(&self) -> Result<Vec<u8>, ProtoError> {
                    let aligned = rkyv::to_bytes::<RkyvError>(self)
                        .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
                    Ok(aligned.into_vec())
                }

                /// Validates and deserializes from raw `rkyv` bytes via 16-byte aligned buffer.
                pub fn from_rkyv_bytes(raw: &[u8]) -> Result<Self, ProtoError> {
                    let mut aligned = AlignedVec::<16>::with_capacity(raw.len());
                    aligned.extend_from_slice(raw);
                    rkyv::from_bytes::<Self, RkyvError>(&aligned)
                        .map_err(|e| ProtoError::Rkyv(e.to_string()))
                }

                /// Encodes with `BFLD\x01` header + pure-Rust `lz4_flex` compression for disk/HTTP blobs.
                pub fn to_compressed_frame(&self) -> Result<Vec<u8>, ProtoError> {
                    let raw = self.to_rkyv_bytes()?;
                    let compressed = lz4_flex::compress_prepend_size(&raw);
                    let mut frame = Vec::with_capacity(WIRE_MAGIC.len() + compressed.len());
                    frame.extend_from_slice(&WIRE_MAGIC);
                    frame.extend_from_slice(&compressed);
                    Ok(frame)
                }

                /// Decodes and validates a `BFLD\x01` + `lz4_flex` compressed wire/storage frame.
                pub fn from_compressed_frame(frame: &[u8]) -> Result<Self, ProtoError> {
                    if frame.len() < WIRE_MAGIC.len() || frame[..WIRE_MAGIC.len()] != WIRE_MAGIC {
                        return Err(ProtoError::InvalidHeader);
                    }
                    let decompressed = lz4_flex::decompress_size_prepended(&frame[WIRE_MAGIC.len()..])?;
                    Self::from_rkyv_bytes(&decompressed)
                }
            }
        )+
    };
}

impl_wire_codec!(
    NodeCapabilityProfile,
    ProjectManifest,
    TrajectoryCheckpoint,
    WorkUnitLease,
    SignedWorkUnitResult,
    WorkUnitRecord,
    QuarantineEntry,
    MremdEndpoint,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prcg::ComputeTier;

    #[test]
    fn prcg_key_preserves_lexicographic_order() {
        let a = WorkUnitId::new(1, 0, 2, 9);
        let b = WorkUnitId::new(1, 0, 2, 10);
        let c = WorkUnitId::new(1, 1, 0, 0);
        assert!(a.to_key_bytes() < b.to_key_bytes());
        assert!(b.to_key_bytes() < c.to_key_bytes());
        assert_eq!(WorkUnitId::from_key_bytes(b.to_key_bytes()), b);
        assert_eq!(b.to_string().parse::<WorkUnitId>().unwrap(), b);
    }

    #[test]
    fn checkpoint_compressed_rkyv_roundtrip() {
        let cp = TrajectoryCheckpoint {
            wu_id: WorkUnitId::new(100, 2, 5, 0),
            cumulative_steps: 50_000,
            temperature_kelvin: 310.0,
            rng_state: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            rng_counter: 42,
            positions_angstrom: vec![[1.25, -3.5, 8.125], [2.75, -2.1, 7.9]],
            velocities_ang_per_ps: vec![[0.01, -0.02, 0.005], [-0.01, 0.03, -0.004]],
            potential_energy_kcal: -142.75,
            kinetic_energy_kcal: 38.25,
            rg_angstrom: 7.42,
            sasa_angstrom2: 1412.8,
        };
        let frame = cp.to_compressed_frame().expect("compress frame");
        let decoded = TrajectoryCheckpoint::from_compressed_frame(&frame).expect("decompress");
        assert_eq!(cp, decoded);
        assert_eq!(ComputeTier::TierA.recommended_steps(), 250_000);
    }
}
