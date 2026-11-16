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
