use crate::prcg::{NodeCapabilityProfile, WorkUnitId};
use crate::wire::{ProjectManifest, ProtoError, SignedWorkUnitResult, WorkUnitLease};
use crate::WIRE_MAGIC;
use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};

/// Network contact descriptor for a discovered LAN Beowulf peer.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerEndpoint {
    pub node_pubkey: [u8; 32],
    /// Socket address string (`"192.168.1.42:7443"`).
    pub addr: String,
}

/// Live worker status entry surfaced in `ClusterStateSnapshot` for CLI and Supervisor GUI panels.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LivePeerStatus {
    pub capability: NodeCapabilityProfile,
    pub addr: String,
    pub active_wu: Option<WorkUnitId>,
    pub step_progress: u64,
    pub target_steps: u64,
    pub current_pot_kcal: f64,
    pub cpu_usage_pct: f32,
    pub throttled: bool,
    pub last_seen_unix_sec: u64,
}

/// Self-contained work unit leased over the LAN (includes the compressed input checkpoint blob).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LanLeasedUnit {
    pub lease: WorkUnitLease,
    pub project_id: u32,
    pub compressed_input_checkpoint: Vec<u8>,
}

/// Replicated state-machine log commands distributed across LAN Raft voters so a newly elected
/// failover Leader immediately knows active leases, completed generations, and project bundles.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RaftLogCommand {
    SeedProjectBundle {
        manifest: ProjectManifest,
        compressed_topology_blob: Vec<u8>,
        gen0_compressed_checkpoints: Vec<Vec<u8>>,
    },
    LeaseIssued {
        lease: WorkUnitLease,
    },
    GenerationCommitted {
        receipt: SignedWorkUnitResult,
        compressed_output_checkpoint: Vec<u8>,
        next_temperature_kelvin: f64,
        spawned_next_gen: Option<WorkUnitId>,
    },
    GenerationQuarantined {
        wu_id: WorkUnitId,
        worker_pubkey: [u8; 32],
        reason: String,
        timestamp_unix_sec: u64,
    },
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RaftLogEntry {
    pub index: u64,
    pub term: u64,
    pub command: RaftLogCommand,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RaftVoteRequest {
    pub term: u64,
    pub candidate_pubkey: [u8; 32],
    pub candidate_addr: String,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RaftVoteResponse {
    pub term: u64,
    pub vote_granted: bool,
    pub voter_pubkey: [u8; 32],
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RaftAppendEntriesRequest {
    pub term: u64,
    pub leader_pubkey: [u8; 32],
    pub leader_addr: String,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<RaftLogEntry>,
    pub leader_commit: u64,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RaftAppendEntriesResponse {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
    pub follower_pubkey: [u8; 32],
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ClusterStateSnapshot {
    pub raft_term: u64,
    pub leader: Option<PeerEndpoint>,
    pub active_project_id: Option<u32>,
    pub active_project_name: Option<String>,
    pub peers: Vec<LivePeerStatus>,
    pub total_workunits: u64,
    pub queued_workunits: u64,
    pub leased_workunits: u64,
    pub completed_workunits: u64,
    pub mremd_swaps: u64,
    pub mremd_trials: u64,
    pub quarantined_events: u64,
    pub lowest_potential_kcal: Option<(WorkUnitId, f64)>,
}

/// Intra-cluster Beowulf RPC requests carried over `quinn` TLS 1.3 bidirectional streams.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LanRpcRequest {
    /// Discovery & subnet-sweep liveness probe exchanging node capability profiles.
    PingHandshake {
        sender_addr: String,
        capability: NodeCapabilityProfile,
        raft_term: u64,
    },
    /// Ongaro-Ousterhout Raft leader election vote request.
    RaftRequestVote(RaftVoteRequest),
    /// Raft heartbeat and replicated PRCG lease/commit log entries.
    RaftAppendEntries(RaftAppendEntriesRequest),
    /// Downloads static project manifest + compressed `ProteinTopology` once per worker node.
    FetchProjectBundle {
        project_id: u32,
    },
    /// Requests `count` (`1` or `2` for `prefetch_depth = 1`) capability-matched PRCG chunks.
    AcquireLanLease {
        worker: NodeCapabilityProfile,
        worker_addr: String,
        max_units: u8,
    },
    /// Uploads a worker-signed Generation completion receipt + compressed checkpoint blob.
    SubmitLanResult {
        receipt: SignedWorkUnitResult,
        compressed_checkpoint_blob: Vec<u8>,
    },
    /// 5-second worker telemetry & lease renewal frame.
    WorkerHeartbeat {
        worker: NodeCapabilityProfile,
        worker_addr: String,
        active_wu: Option<WorkUnitId>,
        step_progress: u64,
        target_steps: u64,
        current_pot_kcal: f64,
        cpu_usage_pct: f32,
        throttled: bool,
    },
    /// Queries live Beowulf cluster topology and folding throughput (`fold-node cluster-status` / `fold-gui`).
    QueryClusterState,
}

/// Intra-cluster Beowulf RPC responses.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LanRpcResponse {
    PongHandshake {
        responder_addr: String,
        capability: NodeCapabilityProfile,
        raft_term: u64,
        leader: Option<PeerEndpoint>,
    },
    RaftVote(RaftVoteResponse),
    RaftAppendEntries(RaftAppendEntriesResponse),
    ProjectBundle {
        found: Option<(ProjectManifest, Vec<u8>)>,
        redirect_leader: Option<PeerEndpoint>,
    },
    LanLeaseBatch {
        units: Vec<LanLeasedUnit>,
        active_project_id: Option<u32>,
        redirect_leader: Option<PeerEndpoint>,
    },
    SubmitAck {
        accepted: bool,
        quarantined: bool,
        message: String,
        spawned_next_gen: Option<WorkUnitId>,
        redirect_leader: Option<PeerEndpoint>,
    },
    HeartbeatAck {
        raft_term: u64,
        leader: Option<PeerEndpoint>,
    },
    ClusterState(ClusterStateSnapshot),
    Error(String),
}

/// Cryptographically authenticated LAN wire envelope (`BLAKE3` keyed MAC over compressed `rkyv` frame).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedLanFrame {
    pub sender_pubkey: [u8; 32],
    pub nonce: [u8; 16],
    pub mac_blake3: [u8; 32],
    pub compressed_payload: Vec<u8>,
}

impl AuthenticatedLanFrame {
    pub fn seal_request(
        cluster_mac_key: &[u8; 32],
        sender_pubkey: [u8; 32],
        req: &LanRpcRequest,
    ) -> Result<Vec<u8>, ProtoError> {
        let raw = rkyv::to_bytes::<RkyvError>(req)
            .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
        Self::seal_inner(cluster_mac_key, sender_pubkey, b"req", raw.as_slice())
    }

    pub fn open_request(
        cluster_mac_key: &[u8; 32],
        wire_bytes: &[u8],
    ) -> Result<([u8; 32], LanRpcRequest), ProtoError> {
        let (sender, raw) = Self::open_inner(cluster_mac_key, b"req", wire_bytes)?;
        let mut aligned = AlignedVec::<16>::with_capacity(raw.len());
        aligned.extend_from_slice(&raw);
        let req = rkyv::from_bytes::<LanRpcRequest, RkyvError>(&aligned)
            .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
        Ok((sender, req))
    }

    pub fn seal_response(
        cluster_mac_key: &[u8; 32],
        sender_pubkey: [u8; 32],
        resp: &LanRpcResponse,
    ) -> Result<Vec<u8>, ProtoError> {
        let raw = rkyv::to_bytes::<RkyvError>(resp)
            .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
        Self::seal_inner(cluster_mac_key, sender_pubkey, b"rsp", raw.as_slice())
    }

    pub fn open_response(
        cluster_mac_key: &[u8; 32],
        wire_bytes: &[u8],
    ) -> Result<([u8; 32], LanRpcResponse), ProtoError> {
        let (sender, raw) = Self::open_inner(cluster_mac_key, b"rsp", wire_bytes)?;
        let mut aligned = AlignedVec::<16>::with_capacity(raw.len());
        aligned.extend_from_slice(&raw);
        let resp = rkyv::from_bytes::<LanRpcResponse, RkyvError>(&aligned)
            .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
        Ok((sender, resp))
    }

    fn seal_inner(
        cluster_mac_key: &[u8; 32],
        sender_pubkey: [u8; 32],
        dir_tag: &[u8],
        raw_rkyv: &[u8],
    ) -> Result<Vec<u8>, ProtoError> {
        let compressed_payload = lz4_flex::compress_prepend_size(raw_rkyv);
        let mut nonce = [0u8; 16];
        getrandom::getrandom(&mut nonce).unwrap_or_default();

        let mac_blake3 = Self::compute_mac(
            cluster_mac_key,
            dir_tag,
            &sender_pubkey,
            &nonce,
            &compressed_payload,
        );

        let env = AuthenticatedLanFrame {
            sender_pubkey,
            nonce,
            mac_blake3,
            compressed_payload,
        };
        let env_bytes = rkyv::to_bytes::<RkyvError>(&env)
            .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
        let mut out = Vec::with_capacity(WIRE_MAGIC.len() + env_bytes.len());
        out.extend_from_slice(&WIRE_MAGIC);
        out.extend_from_slice(env_bytes.as_slice());
        Ok(out)
    }

    fn open_inner(
        cluster_mac_key: &[u8; 32],
        dir_tag: &[u8],
        wire_bytes: &[u8],
    ) -> Result<([u8; 32], Vec<u8>), ProtoError> {
        if wire_bytes.len() < WIRE_MAGIC.len() || wire_bytes[..WIRE_MAGIC.len()] != WIRE_MAGIC {
            return Err(ProtoError::InvalidHeader);
        }
        let body = &wire_bytes[WIRE_MAGIC.len()..];
        let mut aligned = AlignedVec::<16>::with_capacity(body.len());
        aligned.extend_from_slice(body);
        let env = rkyv::from_bytes::<AuthenticatedLanFrame, RkyvError>(&aligned)
            .map_err(|e| ProtoError::Rkyv(e.to_string()))?;

        let expected = Self::compute_mac(
            cluster_mac_key,
            dir_tag,
            &env.sender_pubkey,
            &env.nonce,
            &env.compressed_payload,
        );
        // Constant-time byte comparison via `blake3::Hash` equality
        if blake3::Hash::from_bytes(expected) != blake3::Hash::from_bytes(env.mac_blake3) {
            return Err(ProtoError::Signature(
                "LAN cluster secret MAC mismatch (unauthorized peer)".into(),
            ));
        }

        let decompressed = lz4_flex::decompress_size_prepended(&env.compressed_payload)?;
        Ok((env.sender_pubkey, decompressed))
    }

    fn compute_mac(
        cluster_mac_key: &[u8; 32],
        dir_tag: &[u8],
        sender_pubkey: &[u8; 32],
        nonce: &[u8; 16],
        compressed_payload: &[u8],
    ) -> [u8; 32] {
        let mut h = blake3::Hasher::new_keyed(cluster_mac_key);
        h.update(b"beofold-lan-v2:");
        h.update(dir_tag);
        h.update(sender_pubkey);
        h.update(nonce);
        h.update(compressed_payload);
        *h.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::derive_cluster_mac_key;

    #[test]
    fn authenticated_lan_envelope_rejects_wrong_cluster_secret() {
        let key_ok = derive_cluster_mac_key("alpha-cluster");
        let key_bad = derive_cluster_mac_key("intruder-cluster");
        let sender = [9u8; 32];

        let frame = AuthenticatedLanFrame::seal_request(
            &key_ok,
            sender,
            &LanRpcRequest::QueryClusterState,
        )
        .expect("seal");

        let (pk, decoded) = AuthenticatedLanFrame::open_request(&key_ok, &frame).expect("open ok");
        assert_eq!(pk, sender);
        assert_eq!(decoded, LanRpcRequest::QueryClusterState);

        assert!(AuthenticatedLanFrame::open_request(&key_bad, &frame).is_err());
    }
}
