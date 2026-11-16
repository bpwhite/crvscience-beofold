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

use crate::lan::LanLeasedUnit;
use crate::prcg::WorkUnitId;
use crate::wire::{ProjectManifest, ProtoError, SignedWorkUnitResult};
use crate::WIRE_MAGIC;
use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};

pub const CONTENT_TYPE_RKYV: &str = "application/x-beofold-rkyv";
pub const HEADER_BEOFOLD_PUBKEY: &str = "x-beofold-pubkey";
pub const HEADER_BEOFOLD_TIMESTAMP: &str = "x-beofold-timestamp";
pub const HEADER_BEOFOLD_SIGNATURE: &str = "x-beofold-signature";

/// `POST /api/v1/clusters/register` request payload sent by each remote Beowulf Cluster Leader.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WanClusterRegistration {
    pub cluster_pubkey: [u8; 32],
    pub cluster_name: String,
    pub total_nodes: u32,
    pub tier_s_nodes: u32,
    pub tier_a_nodes: u32,
    pub tier_b_nodes: u32,
    pub aggregate_ns_per_day: f64,
    pub os_summary: String,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WanClusterRegistrationAck {
    pub accepted: bool,
    pub trusted_cluster: bool,
    pub required_quorum: u8,
    pub heartbeat_interval_sec: u64,
    pub server_unix_sec: u64,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WanActiveProjectEntry {
    pub manifest: ProjectManifest,
    /// Relative blob download path (e.g. `"/blobs/a4/a4f8...lz4"`).
    pub topology_blob_path: String,
}

/// `GET /api/v1/projects/active` response payload.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WanActiveProjectsResponse {
    pub projects: Vec<WanActiveProjectEntry>,
}

/// `POST /api/v1/leases/batch-acquire` request sized to a home cluster's local Tier S/A/B slots.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WanBatchLeaseRequest {
    pub cluster_pubkey: [u8; 32],
    pub project_id: Option<u32>,
    pub tier_s_slots: u16,
    pub tier_a_slots: u16,
    pub tier_b_slots: u16,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WanBatchLeaseResponse {
    pub leases: Vec<LanLeasedUnit>,
}

/// `POST /api/v1/leases/heartbeat` 120-second batch renewal request.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WanLeaseHeartbeatRequest {
    pub cluster_pubkey: [u8; 32],
    pub active_leases: Vec<(WorkUnitId, [u8; 16])>,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WanLeaseHeartbeatResponse {
    pub renewed_count: u32,
    pub expired_units: Vec<WorkUnitId>,
}

/// `PUT /api/v1/workunits/{p}/{r}/{c}/{g}` trajectory chunk upload.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WanWorkUnitSubmission {
    pub cluster_pubkey: [u8; 32],
    pub receipt: SignedWorkUnitResult,
    pub compressed_checkpoint_blob: Vec<u8>,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WanWorkUnitAck {
    pub accepted: bool,
    pub awaiting_quorum: bool,
    pub quarantined: bool,
    pub spawned_next_gen: Option<WorkUnitId>,
    pub message: String,
}

/// Persistent ledger record for a registered remote Beowulf cluster on the Apache sync server.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RegisteredClusterRecord {
    pub cluster_pubkey: [u8; 32],
    pub cluster_name: String,
    pub total_nodes: u32,
    pub tier_s_nodes: u32,
    pub tier_a_nodes: u32,
    pub tier_b_nodes: u32,
    pub aggregate_ns_per_day: f64,
    pub os_summary: String,
    pub trusted: bool,
    pub completed_units: u64,
    pub quarantined_units: u64,
    pub last_seen_unix_sec: u64,
}

/// Staged first-quorum candidate stored when running in untrusted `quorum = 2` verification mode.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PendingQuorumCandidate {
    pub wu_id: WorkUnitId,
    pub first_cluster_pubkey: [u8; 32],
    pub first_receipt: SignedWorkUnitResult,
    pub first_checkpoint_blake3: [u8; 32],
}

/// `GET /api/v1/telemetry/global` multi-cluster federation snapshot.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GlobalFederationTelemetry {
    pub server_unix_sec: u64,
    pub quorum_mode: u8,
    pub clusters: Vec<RegisteredClusterRecord>,
    pub aggregate_ns_per_day: f64,
    pub active_projects: Vec<u32>,
    pub total_workunits: u64,
    pub queued_workunits: u64,
    pub leased_workunits: u64,
    pub quorum_pending_workunits: u64,
    pub completed_workunits: u64,
    pub mremd_swaps: u64,
    pub mremd_trials: u64,
    pub quarantined_events: u64,
    pub canary_audits_passed: u64,
    pub lowest_potential_kcal: Option<(WorkUnitId, f64)>,
}

impl GlobalFederationTelemetry {
    /// Formats a clean, dependency-free RFC-8259 JSON document when `Accept: application/json`
    /// is requested by a browser, dashboard, or `curl` health check.
    pub fn to_pretty_json(&self) -> String {
        let best_json = match self.lowest_potential_kcal {
            Some((wu, e)) => format!(
                "{{\"workunit\":\"{}\",\"potential_kcal\":{:.4}}}",
                wu, e
            ),
            None => "null".to_string(),
        };
        let mut cluster_entries = Vec::with_capacity(self.clusters.len());
        for c in &self.clusters {
            cluster_entries.push(format!(
                concat!(
                    "    {{\n",
                    "      \"cluster_name\": \"{}\",\n",
                    "      \"cluster_pubkey_prefix\": \"{}\",\n",
                    "      \"total_nodes\": {},\n",
                    "      \"tier_s_nodes\": {},\n",
                    "      \"tier_a_nodes\": {},\n",
                    "      \"tier_b_nodes\": {},\n",
                    "      \"aggregate_ns_per_day\": {:.2},\n",
                    "      \"os_summary\": \"{}\",\n",
                    "      \"trusted\": {},\n",
                    "      \"completed_units\": {},\n",
                    "      \"quarantined_units\": {},\n",
                    "      \"last_seen_unix_sec\": {}\n",
                    "    }}"
                ),
                escape_json(&c.cluster_name),
                &crate::crypto::bytes_to_hex(&c.cluster_pubkey)[0..16],
                c.total_nodes,
                c.tier_s_nodes,
                c.tier_a_nodes,
                c.tier_b_nodes,
                c.aggregate_ns_per_day,
                escape_json(&c.os_summary),
                c.trusted,
                c.completed_units,
                c.quarantined_units,
                c.last_seen_unix_sec
            ));
        }

        format!(
            concat!(
                "{{\n",
                "  \"server_unix_sec\": {},\n",
                "  \"quorum_mode\": {},\n",
                "  \"aggregate_ns_per_day\": {:.2},\n",
                "  \"active_projects\": {:?},\n",
                "  \"total_workunits\": {},\n",
                "  \"queued_workunits\": {},\n",
                "  \"leased_workunits\": {},\n",
                "  \"quorum_pending_workunits\": {},\n",
                "  \"completed_workunits\": {},\n",
                "  \"mremd_swaps\": {},\n",
                "  \"mremd_trials\": {},\n",
                "  \"quarantined_events\": {},\n",
                "  \"canary_audits_passed\": {},\n",
                "  \"lowest_energy_state\": {},\n",
                "  \"clusters\": [\n{}\n  ]\n",
                "}}\n"
            ),
            self.server_unix_sec,
            self.quorum_mode,
            self.aggregate_ns_per_day,
            self.active_projects,
            self.total_workunits,
            self.queued_workunits,
            self.leased_workunits,
            self.quorum_pending_workunits,
            self.completed_workunits,
            self.mremd_swaps,
            self.mremd_trials,
            self.quarantined_events,
            self.canary_audits_passed,
            best_json,
            cluster_entries.join(",\n")
        )
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

macro_rules! impl_wan_codec {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $ty {
                pub fn to_rkyv_bytes(&self) -> Result<Vec<u8>, ProtoError> {
                    let aligned = rkyv::to_bytes::<RkyvError>(self)
                        .map_err(|e| ProtoError::Rkyv(e.to_string()))?;
                    Ok(aligned.into_vec())
                }

                pub fn from_rkyv_bytes(raw: &[u8]) -> Result<Self, ProtoError> {
                    let mut aligned = AlignedVec::<16>::with_capacity(raw.len());
                    aligned.extend_from_slice(raw);
                    rkyv::from_bytes::<Self, RkyvError>(&aligned)
                        .map_err(|e| ProtoError::Rkyv(e.to_string()))
                }

                pub fn to_http_frame(&self) -> Result<Vec<u8>, ProtoError> {
                    let raw = self.to_rkyv_bytes()?;
                    let compressed = lz4_flex::compress_prepend_size(&raw);
                    let mut frame = Vec::with_capacity(WIRE_MAGIC.len() + compressed.len());
                    frame.extend_from_slice(&WIRE_MAGIC);
                    frame.extend_from_slice(&compressed);
                    Ok(frame)
                }

                pub fn from_http_frame(frame: &[u8]) -> Result<Self, ProtoError> {
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

impl_wan_codec!(
    WanClusterRegistration,
    WanClusterRegistrationAck,
    WanActiveProjectEntry,
    WanActiveProjectsResponse,
    WanBatchLeaseRequest,
    WanBatchLeaseResponse,
    WanLeaseHeartbeatRequest,
    WanLeaseHeartbeatResponse,
    WanWorkUnitSubmission,
    WanWorkUnitAck,
    RegisteredClusterRecord,
    PendingQuorumCandidate,
    GlobalFederationTelemetry,
);
