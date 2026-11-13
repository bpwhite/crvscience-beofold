use crate::dashboard::{
    build_best_conformation_json, build_best_conformation_pdb, parse_project_id_query,
    PUBLIC_DASHBOARD_HTML,
};
use fold_physics::{ChaCha8Rng, SanityThresholds};
use fold_proto::{
    canonicalize_sync_route, verify_http_request_headers, LanLeasedUnit, RegisteredClusterRecord,
    TrajectoryCheckpoint, WanActiveProjectEntry, WanActiveProjectsResponse, WanBatchLeaseRequest,
    WanBatchLeaseResponse, WanClusterRegistration, WanClusterRegistrationAck,
    WanLeaseHeartbeatRequest, WanLeaseHeartbeatResponse, WanWorkUnitAck, WanWorkUnitSubmission,
    WorkUnitId, CONTENT_TYPE_RKYV, HEADER_BEOFOLD_PUBKEY, HEADER_BEOFOLD_SIGNATURE,
    HEADER_BEOFOLD_TIMESTAMP,
};
use fold_storage::{to_hex, CommitOutcome, FoldStore};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct SyncServerPolicy {
    /// Default quorum for untrusted external clusters (`1` = immediate, `2` = redundant cross-cluster verification).
    pub default_quorum: u8,
    /// Pre-approved cluster `Ed25519` public keys that always run at `quorum = 1`.
    pub trusted_cluster_pubkeys: HashSet<[u8; 32]>,
    /// If `true`, newly registering clusters are automatically marked trusted (`quorum = 1`),
    /// suitable when connecting your own private clusters behind a shared firewall/VPN.
    pub trust_all_registered: bool,
    /// Percentage (`0..=100`) of submissions audited via Gate 3 deterministic 20-step canary replay.
    pub canary_audit_pct: u8,
    /// WAN lease TTL in seconds (`600s` default; renewed every `120s` via `/leases/heartbeat`).
    pub wan_lease_ttl_sec: u64,
}

impl Default for SyncServerPolicy {
    fn default() -> Self {
        Self {
            default_quorum: 1,
            trusted_cluster_pubkeys: HashSet::new(),
            trust_all_registered: true,
            canary_audit_pct: 20,
            wan_lease_ttl_sec: 600,
        }
    }
}

pub struct RawHttpRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

pub struct RawHttpResponse {
    pub status: u16,
    pub content_type: String,
    pub content_disposition: Option<String>,
    pub cache_control: Option<String>,
    pub body: Vec<u8>,
}

impl RawHttpResponse {
    fn rkyv_ok(frame: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type: CONTENT_TYPE_RKYV.to_string(),
            content_disposition: None,
            cache_control: None,
            body: frame,
        }
    }

    fn json_ok(json_str: String) -> Self {
        Self {
            status: 200,
            content_type: "application/json; charset=utf-8".to_string(),
            content_disposition: None,
            cache_control: None,
            body: json_str.into_bytes(),
        }
    }

    fn json_no_cache(json_str: String) -> Self {
        Self {
            status: 200,
            content_type: "application/json; charset=utf-8".to_string(),
            content_disposition: None,
            cache_control: Some("no-store".to_string()),
            body: json_str.into_bytes(),
        }
    }

    fn html_ok(html_str: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/html; charset=utf-8".to_string(),
            content_disposition: None,
            cache_control: Some("no-cache".to_string()),
            body: html_str.as_bytes().to_vec(),
        }
    }

    fn pdb_download(filename: &str, pdb_text: String) -> Self {
        Self {
            status: 200,
            content_type: "chemical/x-pdb; charset=utf-8".to_string(),
            content_disposition: Some(format!("attachment; filename=\"{}\"", filename)),
            cache_control: Some("no-store".to_string()),
            body: pdb_text.into_bytes(),
        }
    }

    fn err(status: u16, msg: impl Into<String>) -> Self {
        let m = msg.into();
        Self {
            status,
            content_type: "application/json; charset=utf-8".to_string(),
            content_disposition: None,
            cache_control: Some("no-store".to_string()),
            body: format!("{{\"error\":\"{}\"}}\n", m.replace('"', "'")).into_bytes(),
        }
    }
}

/// Transport-agnostic Web Synchronizing Server engine shared identically between
/// Mode A (`axum` daemon behind Apache `mod_proxy_http`) and Mode B (Apache `mod_cgid` binary).
pub struct SyncCore {
    pub store: FoldStore,
    pub policy: SyncServerPolicy,
    mremd_rng: Mutex<ChaCha8Rng>,
}

impl SyncCore {
    pub fn open(data_dir: &Path, policy: SyncServerPolicy) -> Result<Self, String> {
        let store = FoldStore::open(data_dir).map_err(|e| e.to_string())?;
        let rng = ChaCha8Rng::from_prcg_seed(WorkUnitId::new(0, 4242, 4242, 0), b"apache-wan-sync");
        Ok(Self {
            store,
            policy,
            mremd_rng: Mutex::new(rng),
        })
    }

    fn authenticate_cluster(&self, req: &RawHttpRequest, now: u64) -> Result<[u8; 32], RawHttpResponse> {
        let Some(pk_hex) = req.headers.get(HEADER_BEOFOLD_PUBKEY) else {
            return Err(RawHttpResponse::err(401, "missing X-BeoFold-Pubkey header"));
        };
        let Some(ts_str) = req.headers.get(HEADER_BEOFOLD_TIMESTAMP) else {
            return Err(RawHttpResponse::err(401, "missing X-BeoFold-Timestamp header"));
        };
        let Some(sig_hex) = req.headers.get(HEADER_BEOFOLD_SIGNATURE) else {
            return Err(RawHttpResponse::err(401, "missing X-BeoFold-Signature header"));
        };

        verify_http_request_headers(
            &req.method,
            &req.path,
            pk_hex,
            ts_str,
            sig_hex,
            &req.body,
            now,
            300,
        )
        .map_err(|e| RawHttpResponse::err(401, e.to_string()))
    }

    pub fn dispatch(&self, req: RawHttpRequest) -> RawHttpResponse {
        let now = unix_now();
        let method = req.method.to_ascii_uppercase();
        let route = canonicalize_sync_route(&req.path);
        let wants_json = req
            .headers
            .get("accept")
            .is_some_and(|a| a.contains("application/json"));

        // Static content-addressed blob serving (`/blobs/{prefix}/{hash}.lz4`)
        if method == "GET" && route.starts_with("/blobs/") {
            let parts: Vec<&str> = route.trim_start_matches('/').split('/').collect();
            if parts.len() == 3 {
                return match self.store.read_raw_blob_by_hex(parts[1], parts[2]) {
                    Ok(bytes) => RawHttpResponse {
                        status: 200,
                        content_type: "application/octet-stream".to_string(),
                        content_disposition: None,
                        cache_control: Some("public, max-age=31536000, immutable".to_string()),
                        body: bytes,
                    },
                    Err(e) => RawHttpResponse::err(404, e.to_string()),
                };
            }
            return RawHttpResponse::err(400, "expected /blobs/{2-hex}/{64-hex}.lz4");
        }

        match (method.as_str(), route.as_str()) {
            ("GET", "/health") => RawHttpResponse::json_ok("{\"status\":\"ok\"}\n".to_string()),

            // Read-only Public Browser Observatory (`GET /beofold/dashboard` or `GET /beofold/`)
            ("GET", "/") | ("GET", "/dashboard") => {
                if route == "/" && wants_json {
                    return RawHttpResponse::json_no_cache(
                        concat!(
                            "{\n",
                            "  \"service\": \"beofold-sync\",\n",
                            "  \"version\": \"1.4.0\",\n",
                            "  \"dashboard\": \"/beofold/dashboard\",\n",
                            "  \"telemetry\": \"/beofold/api/v1/telemetry/global\",\n",
                            "  \"best_conformation\": \"/beofold/api/v1/public/conformation/best\",\n",
                            "  \"best_pdb\": \"/beofold/api/v1/public/conformation/best.pdb\"\n",
                            "}\n"
                        )
                        .to_string(),
                    );
                }
                RawHttpResponse::html_ok(PUBLIC_DASHBOARD_HTML)
            }

            // Read-only 3D Conformation JSON (`GET /api/v1/public/conformation/best?project_id=1`)
            ("GET", "/public/conformation/best") => {
                let req_pid = parse_project_id_query(&req.path);
                match build_best_conformation_json(&self.store, req_pid) {
                    Ok(json) => RawHttpResponse::json_no_cache(json),
                    Err(e) => RawHttpResponse::err(500, e),
                }
            }

            // Read-only Crystallographic PDB Download (`GET /api/v1/public/conformation/best.pdb`)
            ("GET", "/public/conformation/best.pdb") => {
                let req_pid = parse_project_id_query(&req.path);
                match build_best_conformation_pdb(&self.store, req_pid) {
                    Ok(Some((filename, pdb_text))) => {
                        RawHttpResponse::pdb_download(&filename, pdb_text)
                    }
                    Ok(None) => RawHttpResponse::err(
                        404,
                        "no seeded or completed conformation found in ledger",
                    ),
                    Err(e) => RawHttpResponse::err(500, e),
                }
            }

            // 1. POST /api/v1/clusters/register
            ("POST", "/clusters/register") => {
                let caller_pk = match self.authenticate_cluster(&req, now) {
                    Ok(pk) => pk,
                    Err(resp) => return resp,
                };
                let reg = match WanClusterRegistration::from_http_frame(&req.body) {
                    Ok(r) => r,
                    Err(e) => return RawHttpResponse::err(400, e.to_string()),
                };
                if reg.cluster_pubkey != caller_pk {
                    return RawHttpResponse::err(403, "payload cluster_pubkey mismatches header signature");
                }

                let force_trust = self.policy.trust_all_registered
                    || self.policy.trusted_cluster_pubkeys.contains(&caller_pk);

                let record = RegisteredClusterRecord {
                    cluster_pubkey: caller_pk,
                    cluster_name: reg.cluster_name,
                    total_nodes: reg.total_nodes,
                    tier_s_nodes: reg.tier_s_nodes,
                    tier_a_nodes: reg.tier_a_nodes,
                    tier_b_nodes: reg.tier_b_nodes,
                    aggregate_ns_per_day: reg.aggregate_ns_per_day,
                    os_summary: reg.os_summary,
                    trusted: force_trust,
                    completed_units: 0,
                    quarantined_units: 0,
                    last_seen_unix_sec: now,
                };

                let saved = match self.store.upsert_cluster_registration(record, force_trust) {
                    Ok(s) => s,
                    Err(e) => return RawHttpResponse::err(500, e.to_string()),
                };

                let eff_quorum = if saved.trusted {
                    1
                } else {
                    self.policy.default_quorum.max(1)
                };
                let ack = WanClusterRegistrationAck {
                    accepted: true,
                    trusted_cluster: saved.trusted,
                    required_quorum: eff_quorum,
                    heartbeat_interval_sec: 120,
                    server_unix_sec: now,
                };
                match ack.to_http_frame() {
                    Ok(frame) => RawHttpResponse::rkyv_ok(frame),
                    Err(e) => RawHttpResponse::err(500, e.to_string()),
                }
            }

            // 2. GET /api/v1/projects/active
            ("GET", "/projects/active") => {
                let ids = match self.store.list_project_ids() {
                    Ok(v) => v,
                    Err(e) => return RawHttpResponse::err(500, e.to_string()),
                };
                let mut entries = Vec::with_capacity(ids.len());
                for pid in ids {
                    if let Ok((manifest, _topo)) = self.store.load_project(pid) {
                        let topology_blob_path = FoldStore::blob_rel_url(&manifest.topology_blake3);
                        entries.push(WanActiveProjectEntry {
                            manifest,
                            topology_blob_path,
                        });
                    }
                }
                if wants_json {
                    let items: Vec<String> = entries
                        .iter()
                        .map(|e| {
                            format!(
                                "{{\"project_id\":{},\"name\":\"{}\",\"residues\":{},\"atoms\":{},\"topology_blake3\":\"{}\",\"topology_blob_path\":\"{}\"}}",
                                e.manifest.project_id,
                                e.manifest.name,
                                e.manifest.residue_count,
                                e.manifest.atom_count,
                                to_hex(&e.manifest.topology_blake3),
                                e.topology_blob_path
                            )
                        })
                        .collect();
                    return RawHttpResponse::json_no_cache(format!(
                        "{{\"projects\":[{}]}}\n",
                        items.join(",")
                    ));
                }

                let resp = WanActiveProjectsResponse { projects: entries };
                match resp.to_http_frame() {
                    Ok(frame) => RawHttpResponse::rkyv_ok(frame),
                    Err(e) => RawHttpResponse::err(500, e.to_string()),
                }
            }

            // 3. POST /api/v1/leases/batch-acquire
            ("POST", "/leases/batch-acquire") => {
                let caller_pk = match self.authenticate_cluster(&req, now) {
                    Ok(pk) => pk,
                    Err(resp) => return resp,
                };
                let batch_req = match WanBatchLeaseRequest::from_http_frame(&req.body) {
                    Ok(r) => r,
                    Err(e) => return RawHttpResponse::err(400, e.to_string()),
                };
                if batch_req.cluster_pubkey != caller_pk {
                    return RawHttpResponse::err(403, "cluster_pubkey mismatch");
                }

                let total_requested = (batch_req.tier_s_slots as usize)
                    .saturating_add(batch_req.tier_a_slots as usize)
                    .saturating_add(batch_req.tier_b_slots as usize)
                    .clamp(1, 32);

                let trusted = self.store.is_cluster_trusted(&caller_pk).unwrap_or(false);
                let excl = if !trusted && self.policy.default_quorum >= 2 {
                    Some(caller_pk)
                } else {
                    None
                };

                let mut leases = Vec::new();
                for _ in 0..total_requested {
                    let Ok(Some((lease, input_cp))) = self.store.acquire_next_lease_filtered(
                        caller_pk,
                        excl,
                        now,
                        self.policy.wan_lease_ttl_sec,
                    ) else {
                        break;
                    };
                    if let Ok(compressed_input_checkpoint) = input_cp.to_compressed_frame() {
                        leases.push(LanLeasedUnit {
                            project_id: lease.wu_id.project,
                            lease,
                            compressed_input_checkpoint,
                        });
                    }
                }

                let resp = WanBatchLeaseResponse { leases };
                match resp.to_http_frame() {
                    Ok(frame) => RawHttpResponse::rkyv_ok(frame),
                    Err(e) => RawHttpResponse::err(500, e.to_string()),
                }
            }

            // 4. POST /api/v1/leases/heartbeat
            ("POST", "/leases/heartbeat") => {
                let caller_pk = match self.authenticate_cluster(&req, now) {
                    Ok(pk) => pk,
                    Err(resp) => return resp,
                };
                let hb = match WanLeaseHeartbeatRequest::from_http_frame(&req.body) {
                    Ok(r) => r,
                    Err(e) => return RawHttpResponse::err(400, e.to_string()),
                };
                if hb.cluster_pubkey != caller_pk {
                    return RawHttpResponse::err(403, "cluster_pubkey mismatch");
                }

                let extend_to = now + self.policy.wan_lease_ttl_sec;
                let mut renewed_count = 0u32;
                let mut expired_units = Vec::new();
                for (wu_id, tok) in hb.active_leases {
                    match self.store.touch_lease_checked(wu_id, Some(tok), extend_to) {
                        Ok(true) => renewed_count += 1,
                        _ => expired_units.push(wu_id),
                    }
                }

                let resp = WanLeaseHeartbeatResponse {
                    renewed_count,
                    expired_units,
                };
                match resp.to_http_frame() {
                    Ok(frame) => RawHttpResponse::rkyv_ok(frame),
                    Err(e) => RawHttpResponse::err(500, e.to_string()),
                }
            }

            // 5. GET /api/v1/telemetry/global
            ("GET", "/telemetry/global") => {
                let snap = match self
                    .store
                    .build_global_telemetry(self.policy.default_quorum, now)
                {
                    Ok(s) => s,
                    Err(e) => return RawHttpResponse::err(500, e.to_string()),
                };
                if wants_json {
                    return RawHttpResponse::json_no_cache(snap.to_pretty_json());
                }
                match snap.to_http_frame() {
                    Ok(frame) => RawHttpResponse::rkyv_ok(frame),
                    Err(e) => RawHttpResponse::err(500, e.to_string()),
                }
            }

            // 6. PUT /api/v1/workunits/{p}/{r}/{c}/{g}
            ("PUT", path) if path.starts_with("/workunits/") => {
                let caller_pk = match self.authenticate_cluster(&req, now) {
                    Ok(pk) => pk,
                    Err(resp) => return resp,
                };
                let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();
                if segs.len() != 5 {
                    return RawHttpResponse::err(400, "expected /workunits/{p}/{r}/{c}/{g}");
                }
                let Ok(p) = segs[1].parse::<u32>() else {
                    return RawHttpResponse::err(400, "invalid project id");
                };
                let Ok(r) = segs[2].parse::<u32>() else {
                    return RawHttpResponse::err(400, "invalid run id");
                };
                let Ok(c) = segs[3].parse::<u32>() else {
                    return RawHttpResponse::err(400, "invalid clone id");
                };
                let Ok(g) = segs[4].parse::<u32>() else {
                    return RawHttpResponse::err(400, "invalid generation id");
                };
                let path_wu = WorkUnitId::new(p, r, c, g);

                let sub = match WanWorkUnitSubmission::from_http_frame(&req.body) {
                    Ok(s) => s,
                    Err(e) => return RawHttpResponse::err(400, e.to_string()),
                };
                if sub.cluster_pubkey != caller_pk || sub.receipt.wu_id != path_wu {
                    return RawHttpResponse::err(400, "submission PRCG or cluster_pubkey mismatch");
                }

                let Ok((manifest, topo)) = self.store.load_project(p) else {
                    return RawHttpResponse::err(404, "project not found");
                };
                let Ok(output_cp) =
                    TrajectoryCheckpoint::from_compressed_frame(&sub.compressed_checkpoint_blob)
                else {
                    return RawHttpResponse::err(400, "invalid LZ4 checkpoint frame");
                };

                let trusted = self.store.is_cluster_trusted(&caller_pk).unwrap_or(false);
                let eff_quorum = if trusted {
                    1
                } else {
                    self.policy.default_quorum.max(1)
                };

                // Deterministic canary selection (`blake3(receipt_sig)[0] % 100 < canary_audit_pct`)
                let canary_roll = (sub.receipt.signature[0] as u16 * 100 / 256) as u8;
                let do_canary = canary_roll < self.policy.canary_audit_pct;

                let outcome = {
                    let mut rng = self
                        .mremd_rng
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    self.store.commit_signed_result_with_policy(
                        &manifest,
                        &topo,
                        &sub.receipt,
                        &output_cp,
                        &SanityThresholds::default(),
                        &mut rng,
                        now,
                        Some(caller_pk),
                        eff_quorum,
                        do_canary,
                    )
                };

                let ack = match outcome {
                    Ok(CommitOutcome::Accepted {
                        completed_wu,
                        spawned_next_gen,
                        mremd_swapped_with,
                        ..
                    }) => {
                        let swap_msg = mremd_swapped_with
                            .map(|w| format!(" + MREMD swap <-> {}", w))
                            .unwrap_or_default();
                        WanWorkUnitAck {
                            accepted: true,
                            awaiting_quorum: false,
                            quarantined: false,
                            spawned_next_gen,
                            message: format!("accepted {}{}", completed_wu, swap_msg),
                        }
                    }
                    Ok(CommitOutcome::AwaitingSecondQuorum { wu_id }) => WanWorkUnitAck {
                        accepted: true,
                        awaiting_quorum: true,
                        quarantined: false,
                        spawned_next_gen: None,
                        message: format!("staged {} awaiting quorum=2 verification", wu_id),
                    },
                    Ok(CommitOutcome::Quarantined { reason, .. }) => WanWorkUnitAck {
                        accepted: false,
                        awaiting_quorum: false,
                        quarantined: true,
                        spawned_next_gen: None,
                        message: reason,
                    },
                    Err(e) => return RawHttpResponse::err(409, e.to_string()),
                };

                match ack.to_http_frame() {
                    Ok(frame) => RawHttpResponse::rkyv_ok(frame),
                    Err(e) => RawHttpResponse::err(500, e.to_string()),
                }
            }

            _ => RawHttpResponse::err(404, format!("unknown BeoFold sync endpoint: {} {}", method, route)),
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fold_physics::{initialize_clone_checkpoint, minimize_steric_clashes, ProteinTopology};
    use fold_proto::{NodeIdentity, ProjectManifest};

    #[test]
    fn sync_core_handles_apache_prefixed_register_dashboard_and_pdb_export() {
        let tmp = std::env::temp_dir().join(format!("beofold_sync_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);

        let store = FoldStore::open(&tmp).unwrap();
        let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
        minimize_steric_clashes(&topo, &mut coords, 60);
        let cp0 = initialize_clone_checkpoint(
            &topo,
            &coords,
            WorkUnitId::new(1, 0, 0, 0),
            300.0,
            b"sync-test",
        );
        let manifest = ProjectManifest {
            project_id: 1,
            name: "TrpCage_SyncTest".into(),
            sequence: topo.sequence_one_letter.clone(),
            topology_blake3: [0u8; 32],
            initial_coords_blake3: [0u8; 32],
            atom_count: topo.atoms.len() as u32,
            residue_count: topo.residue_count,
            temp_ladder_kelvin: vec![300.0],
            dt_fs: 1.0,
            friction_per_ps: 1.0,
            steps_per_gen: 20,
            num_clones_per_run: 1,
            max_generations: 2,
        };
        store.seed_project(manifest, &topo, &[cp0]).unwrap();
        drop(store);

        let policy = SyncServerPolicy {
            default_quorum: 2,
            trust_all_registered: false,
            canary_audit_pct: 100,
            ..Default::default()
        };
        let core = SyncCore::open(&tmp, policy).unwrap();

        // Cluster A registers via Apache-prefixed `/beofold/api/v1/clusters/register`
        let cluster_a = NodeIdentity::from_cluster_secret("cluster-a", "ldr");
        let reg_a = WanClusterRegistration {
            cluster_pubkey: cluster_a.public_key_bytes(),
            cluster_name: "Alpha".into(),
            total_nodes: 2,
            tier_s_nodes: 0,
            tier_a_nodes: 2,
            tier_b_nodes: 0,
            aggregate_ns_per_day: 24.5,
            os_summary: "windows/x86_64".into(),
        };
        let body_a = reg_a.to_http_frame().unwrap();
        let now = unix_now();
        let (pk_h, ts_h, sig_h) = cluster_a.sign_http_request(
            "POST",
            "/beofold/api/v1/clusters/register",
            now,
            &body_a,
        );
        let mut hdrs = HashMap::new();
        hdrs.insert(HEADER_BEOFOLD_PUBKEY.into(), pk_h);
        hdrs.insert(HEADER_BEOFOLD_TIMESTAMP.into(), ts_h);
        hdrs.insert(HEADER_BEOFOLD_SIGNATURE.into(), sig_h);

        let r_reg = core.dispatch(RawHttpRequest {
            method: "POST".into(),
            path: "/api/v1/clusters/register".into(),
            headers: hdrs,
            body: body_a,
        });
        assert_eq!(r_reg.status, 200);
        let ack = WanClusterRegistrationAck::from_http_frame(&r_reg.body).unwrap();
        assert_eq!(ack.required_quorum, 2);

        // Unauthenticated write attempts remain strictly rejected (`401 Unauthorized`)
        let r_unauth = core.dispatch(RawHttpRequest {
            method: "POST".into(),
            path: "/beofold/api/v1/leases/batch-acquire".into(),
            headers: HashMap::new(),
            body: Vec::new(),
        });
        assert_eq!(r_unauth.status, 401);

        // Verify public HTML5 dashboard route (`GET /beofold/dashboard`)
        let r_dash = core.dispatch(RawHttpRequest {
            method: "GET".into(),
            path: "/beofold/dashboard".into(),
            headers: HashMap::new(),
            body: Vec::new(),
        });
        assert_eq!(r_dash.status, 200);
        assert!(r_dash.content_type.starts_with("text/html"));
        let dash_html = String::from_utf8(r_dash.body).unwrap();
        assert!(dash_html.contains("BeoFold Federation Observatory"));

        // Verify public 3D conformation JSON route (`GET /beofold/api/v1/public/conformation/best?project_id=1`)
        let r_conf = core.dispatch(RawHttpRequest {
            method: "GET".into(),
            path: "/beofold/api/v1/public/conformation/best?project_id=1".into(),
            headers: HashMap::new(),
            body: Vec::new(),
        });
        assert_eq!(r_conf.status, 200);
        let conf_json = String::from_utf8(r_conf.body).unwrap();
        assert!(conf_json.contains("\"available\": true"));
        assert!(conf_json.contains("\"TrpCage_SyncTest\""));

        // Verify public PDB download route (`GET /beofold/api/v1/public/conformation/best.pdb`)
        let r_pdb = core.dispatch(RawHttpRequest {
            method: "GET".into(),
            path: "/beofold/api/v1/public/conformation/best.pdb".into(),
            headers: HashMap::new(),
            body: Vec::new(),
        });
        assert_eq!(r_pdb.status, 200);
        assert!(r_pdb.content_disposition.is_some());
        let pdb_txt = String::from_utf8(r_pdb.body).unwrap();
        assert!(pdb_txt.contains("HEADER    BEOFOLD PROTEIN STRUCTURE"));
        assert!(pdb_txt.contains("ATOM      1"));

        // Verify JSON global telemetry endpoint
        let mut json_hdr = HashMap::new();
        json_hdr.insert("accept".into(), "application/json".into());
        let r_tel = core.dispatch(RawHttpRequest {
            method: "GET".into(),
            path: "/beofold/api/v1/telemetry/global".into(),
            headers: json_hdr,
            body: Vec::new(),
        });
        assert_eq!(r_tel.status, 200);
        let txt = String::from_utf8(r_tel.body).unwrap();
        assert!(txt.contains("\"Alpha\""));
        assert!(txt.contains("\"tier_a_nodes\": 2"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
