pub mod crypto;
pub mod lan;
pub mod prcg;
pub mod wan;
pub mod wire;

pub use crypto::{
    bytes_to_hex, canonicalize_sync_route, compute_http_request_digest,
    compute_result_signing_digest, derive_cluster_id_tag, derive_cluster_mac_key,
    hex_to_bytes_fixed, verify_http_request_headers, verify_signed_result, NodeIdentity,
};
pub use lan::{
    AuthenticatedLanFrame, ClusterStateSnapshot, LanLeasedUnit, LanRpcRequest, LanRpcResponse,
    LivePeerStatus, PeerEndpoint, RaftAppendEntriesRequest, RaftAppendEntriesResponse,
    RaftLogCommand, RaftLogEntry, RaftVoteRequest, RaftVoteResponse,
};
pub use prcg::{ComputeTier, NodeCapabilityProfile, WorkUnitId};
pub use wan::{
    GlobalFederationTelemetry, PendingQuorumCandidate, RegisteredClusterRecord,
    WanActiveProjectEntry, WanActiveProjectsResponse, WanBatchLeaseRequest, WanBatchLeaseResponse,
    WanClusterRegistration, WanClusterRegistrationAck, WanLeaseHeartbeatRequest,
    WanLeaseHeartbeatResponse, WanWorkUnitAck, WanWorkUnitSubmission, CONTENT_TYPE_RKYV,
    HEADER_BEOFOLD_PUBKEY, HEADER_BEOFOLD_SIGNATURE, HEADER_BEOFOLD_TIMESTAMP,
};
pub use wire::{
    MremdEndpoint, ProjectManifest, ProtoError, QuarantineEntry, SignedWorkUnitResult,
    TrajectoryCheckpoint, WorkUnitLease, WorkUnitRecord, WorkUnitStatus,
};

/// Wire format magic prefix (`"BFLD"` + protocol version `1`).
pub const WIRE_MAGIC: [u8; 5] = [b'B', b'F', b'L', b'D', 1];
