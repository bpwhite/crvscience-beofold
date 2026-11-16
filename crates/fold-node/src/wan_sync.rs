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

use crate::mesh::{start_mesh_node, MeshNodeConfig, SharedMeshState};
use crate::raft::RaftRole;
use fold_physics::{initialize_clone_checkpoint, minimize_steric_clashes, ProteinTopology};
use fold_proto::{
    ComputeTier, ProjectManifest, RaftLogCommand, WanActiveProjectsResponse, WanBatchLeaseRequest,
    WanBatchLeaseResponse, WanClusterRegistration, WanClusterRegistrationAck,
    WanLeaseHeartbeatRequest, WanLeaseHeartbeatResponse, WanWorkUnitAck, WorkUnitId,
    CONTENT_TYPE_RKYV, HEADER_BEOFOLD_PUBKEY, HEADER_BEOFOLD_SIGNATURE, HEADER_BEOFOLD_TIMESTAMP,
};
use fold_storage::FoldStore;
use fold_sync::{serve_axum, SyncCore, SyncServerPolicy};
use reqwest::Client;
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

/// Outbound HTTPS synchronization loop running exclusively on the elected LAN Beowulf Cluster
/// Leader when `--sync-url` is configured. Batches leases and trajectory uploads so home LAN
/// worker PCs never open inbound ports or poll the Apache server individually.
pub async fn run_wan_leader_gateway_loop(
    st: Arc<SharedMeshState>,
    sync_api_url: String,
    cluster_name: String,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let Ok(http) = Client::builder().timeout(Duration::from_secs(20)).build() else {
        eprintln!(
            "[WAN:{}] Failed to initialize rustls reqwest client",
            st.node_label
        );
        return;
    };

    let api_base = sync_api_url.trim_end_matches('/').to_string();
    let mount_root = api_base
        .strip_suffix("/api/v1")
        .unwrap_or(&api_base)
        .to_string();

    let mut known_projects: HashSet<u32> = HashSet::new();
    let mut last_register = Instant::now() - Duration::from_secs(300);
    let mut last_heartbeat = Instant::now() - Duration::from_secs(300);
    let mut poll_ticker = tokio::time::interval(Duration::from_millis(750));

    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { break; }
            }
            _ = poll_ticker.tick() => {
                let is_leader = st.raft.lock().await.role == RaftRole::Leader;
                if !is_leader {
                    continue;
                }

                if last_register.elapsed() >= Duration::from_secs(60) {
                    if register_cluster_with_apache(&http, &api_base, &st, &cluster_name).await.is_ok() {
                        last_register = Instant::now();
                    }
                }

                let _ = sync_active_projects(&http, &api_base, &mount_root, &st, &mut known_projects).await;
                let _ = flush_wan_outbox(&http, &api_base, &st).await;

                if let Ok(local_stats) = st.store.collect_stats(None) {
                    let lan_workers = st.peers.read().await.len() + 1;
                    let desired_buffer = (lan_workers * 2).clamp(2, 16);
                    let in_flight_local = local_stats.queued + local_stats.leased;
                    if in_flight_local < desired_buffer {
                        let deficit = (desired_buffer - in_flight_local) as u16;
                        let _ = acquire_wan_batch(&http, &api_base, &st, deficit).await;
                    }
                }

                if last_heartbeat.elapsed() >= Duration::from_secs(90) {
                    if send_wan_lease_heartbeat(&http, &api_base, &st).await.is_ok() {
                        last_heartbeat = Instant::now();
                    }
                }
            }
        }
    }
}

async fn signed_post_or_put(
    http: &Client,
    method: reqwest::Method,
    full_url: &str,
    route_suffix: &str,
    st: &Arc<SharedMeshState>,
    frame_body: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let ts = unix_now();
    let (pk_hex, ts_str, sig_hex) =
        st.identity
            .sign_http_request(method.as_str(), route_suffix, ts, &frame_body);

    let resp = http
        .request(method, full_url)
        .header("content-type", CONTENT_TYPE_RKYV)
        .header(HEADER_BEOFOLD_PUBKEY, pk_hex)
        .header(HEADER_BEOFOLD_TIMESTAMP, ts_str)
        .header(HEADER_BEOFOLD_SIGNATURE, sig_hex)
        .body(frame_body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let code = resp.status();
        let txt = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {} from {}: {}", code, route_suffix, txt.trim()));
    }
    resp.bytes().await.map(|b| b.to_vec()).map_err(|e| e.to_string())
}

async fn register_cluster_with_apache(
    http: &Client,
    api_base: &str,
    st: &Arc<SharedMeshState>,
    cluster_name: &str,
) -> Result<WanClusterRegistrationAck, String> {
    let peers = st.peers.read().await;
    let mut tier_s = 0u32;
    let mut tier_a = 0u32;
    let mut tier_b = 0u32;
    let mut agg_ns = st.capability.villin_ns_per_day;
    let mut os_set: HashSet<String> = HashSet::new();
    os_set.insert(format!("{}/{}", st.capability.os, st.capability.arch));

    match st.capability.tier {
        ComputeTier::TierS => tier_s += 1,
        ComputeTier::TierA => tier_a += 1,
        ComputeTier::TierB => tier_b += 1,
    }

    for p in peers.values() {
        agg_ns += p.capability.villin_ns_per_day;
        os_set.insert(format!("{}/{}", p.capability.os, p.capability.arch));
        match p.capability.tier {
            ComputeTier::TierS => tier_s += 1,
            ComputeTier::TierA => tier_a += 1,
            ComputeTier::TierB => tier_b += 1,
        }
    }
    let total_nodes = (peers.len() + 1) as u32;
    drop(peers);

    let mut os_list: Vec<String> = os_set.into_iter().collect();
    os_list.sort();

    let payload = WanClusterRegistration {
        cluster_pubkey: st.identity.public_key_bytes(),
        cluster_name: cluster_name.to_string(),
        total_nodes,
        tier_s_nodes: tier_s,
        tier_a_nodes: tier_a,
        tier_b_nodes: tier_b,
        aggregate_ns_per_day: agg_ns,
        os_summary: os_list.join(", "),
    };
    let body = payload.to_http_frame().map_err(|e| e.to_string())?;
    let url = format!("{}/clusters/register", api_base);
    let resp_bytes =
        signed_post_or_put(http, reqwest::Method::POST, &url, "/clusters/register", st, body)
            .await?;
    let ack = WanClusterRegistrationAck::from_http_frame(&resp_bytes).map_err(|e| e.to_string())?;
    Ok(ack)
}

async fn sync_active_projects(
    http: &Client,
    api_base: &str,
    mount_root: &str,
    st: &Arc<SharedMeshState>,
    known_projects: &mut HashSet<u32>,
) -> Result<(), String> {
    let url = format!("{}/projects/active", api_base);
    let resp = http.get(&url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("GET /projects/active returned {}", resp.status()));
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    let active = WanActiveProjectsResponse::from_http_frame(&bytes).map_err(|e| e.to_string())?;

    for entry in active.projects {
        let pid = entry.manifest.project_id;
        if known_projects.contains(&pid) || st.store.load_project(pid).is_ok() {
            known_projects.insert(pid);
            continue;
        }
        let blob_url = format!("{}{}", mount_root, entry.topology_blob_path);
        let blob_resp = http.get(&blob_url).send().await.map_err(|e| e.to_string())?;
        if !blob_resp.status().is_success() {
            continue;
        }
        let topo_blob = blob_resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
        if st
            .store
            .import_wan_project_metadata(&entry.manifest, &topo_blob)
            .is_ok()
        {
            let peer_count = st.peers.read().await.len();
            let mut raft = st.raft.lock().await;
            raft.append_leader_command(
                RaftLogCommand::SeedProjectBundle {
                    manifest: entry.manifest.clone(),
                    compressed_topology_blob: topo_blob,
                    gen0_compressed_checkpoints: Vec::new(),
                },
                peer_count,
            );
            known_projects.insert(pid);
            println!(
                "[WAN:{}] Imported active project P{} ('{}') from web sync server",
                st.node_label, pid, entry.manifest.name
            );
        }
    }
    Ok(())
}

async fn acquire_wan_batch(
    http: &Client,
    api_base: &str,
    st: &Arc<SharedMeshState>,
    slots_needed: u16,
) -> Result<usize, String> {
    let has_tier_s = st.capability.tier == ComputeTier::TierS
        || st
            .peers
            .read()
            .await
            .values()
            .any(|p| p.capability.tier == ComputeTier::TierS);

    let (tier_s_slots, tier_a_slots) = if has_tier_s {
        (slots_needed.div_ceil(2), slots_needed / 2)
    } else {
        (0, slots_needed)
    };

    let req = WanBatchLeaseRequest {
        cluster_pubkey: st.identity.public_key_bytes(),
        project_id: None,
        tier_s_slots,
        tier_a_slots,
        tier_b_slots: 0,
    };
    let body = req.to_http_frame().map_err(|e| e.to_string())?;
    let url = format!("{}/leases/batch-acquire", api_base);
    let resp_bytes = signed_post_or_put(
        http,
        reqwest::Method::POST,
        &url,
        "/leases/batch-acquire",
        st,
        body,
    )
    .await?;
    let batch = WanBatchLeaseResponse::from_http_frame(&resp_bytes).map_err(|e| e.to_string())?;

    let mut staged = 0usize;
    for unit in &batch.leases {
        if let Ok(true) = st.store.stage_wan_leased_unit(unit) {
            staged += 1;
        }
    }
    if staged > 0 {
        println!(
            "[WAN:{}] Leased {} PRCG chunks from web sync server for local LAN distribution",
            st.node_label, staged
        );
    }
    Ok(staged)
}

async fn flush_wan_outbox(
    http: &Client,
    api_base: &str,
    st: &Arc<SharedMeshState>,
) -> Result<usize, String> {
    let pending = st.store.peek_wan_outbox(8).map_err(|e| e.to_string())?;
    let mut uploaded = 0usize;

    for mut sub in pending {
        sub.cluster_pubkey = st.identity.public_key_bytes();
        let wu = sub.receipt.wu_id;
        let route = format!(
            "/workunits/{}/{}/{}/{}",
            wu.project, wu.run, wu.clone, wu.generation
        );
        let url = format!("{}{}", api_base, route);
        let body = sub.to_http_frame().map_err(|e| e.to_string())?;

        if let Ok(resp_bytes) =
            signed_post_or_put(http, reqwest::Method::PUT, &url, &route, st, body).await
        {
            if let Ok(ack) = WanWorkUnitAck::from_http_frame(&resp_bytes) {
                let _ = st.store.ack_wan_outbox_uploaded(wu);
                uploaded += 1;
                println!("[WAN:{}] Uploaded {} -> {}", st.node_label, wu, ack.message);
            }
        }
    }
    Ok(uploaded)
}

async fn send_wan_lease_heartbeat(
    http: &Client,
    api_base: &str,
    st: &Arc<SharedMeshState>,
) -> Result<WanLeaseHeartbeatResponse, String> {
    let active_leases = st
        .store
        .list_upstream_wan_leases()
        .map_err(|e| e.to_string())?;
    let req = WanLeaseHeartbeatRequest {
        cluster_pubkey: st.identity.public_key_bytes(),
        active_leases,
    };
    let body = req.to_http_frame().map_err(|e| e.to_string())?;
    let url = format!("{}/leases/heartbeat", api_base);
    let resp_bytes = signed_post_or_put(
        http,
        reqwest::Method::POST,
        &url,
        "/leases/heartbeat",
        st,
        body,
    )
    .await?;
    WanLeaseHeartbeatResponse::from_http_frame(&resp_bytes).map_err(|e| e.to_string())
}

pub async fn run_two_cluster_wan_federation_simulation(
    base_dir: PathBuf,
    quorum: u8,
    generations: u32,
    steps_per_gen: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir)?;

    println!("=========================================================================");
    println!(
        " BeoFold Phase 3 — Multi-Cluster Apache HTTP Federation Test (quorum={})",
        quorum.max(1)
    );
    println!("=========================================================================");

    let sync_dir = base_dir.join("web_sync_vault");
    let sync_store = FoldStore::open(&sync_dir)?;
    let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
    minimize_steric_clashes(&topo, &mut coords, 100);

    let ladder = vec![290.0, 325.0, 360.0, 400.0];
    let clones = 2u32;
    let gens = generations.max(2);
    let expected_total = (ladder.len() as u64) * (clones as u64) * (gens as u64);

    let mut gen0 = Vec::new();
    for (r, &t) in ladder.iter().enumerate() {
        for c in 0..clones {
            gen0.push(initialize_clone_checkpoint(
                &topo,
                &coords,
                WorkUnitId::new(1, r as u32, c, 0),
                t,
                b"phase3-wan-federation",
            ));
        }
    }
    let manifest = ProjectManifest {
        project_id: 1,
        name: "TrpCage_WAN_Federation".into(),
        sequence: topo.sequence_one_letter.clone(),
        topology_blake3: [0u8; 32],
        initial_coords_blake3: [0u8; 32],
        atom_count: topo.atoms.len() as u32,
        residue_count: topo.residue_count,
        temp_ladder_kelvin: ladder,
        dt_fs: 1.5,
        friction_per_ps: 1.0,
        steps_per_gen: steps_per_gen.max(40),
        num_clones_per_run: clones,
        max_generations: gens,
    };
    sync_store.seed_project(manifest, &topo, &gen0)?;
    drop(sync_store);

    let policy = SyncServerPolicy {
        default_quorum: quorum.max(1),
        trust_all_registered: quorum <= 1,
        canary_audit_pct: 50,
        wan_lease_ttl_sec: 300,
        ..Default::default()
    };
    let core = Arc::new(SyncCore::open(&sync_dir, policy)?);
    let (sync_stop_tx, sync_stop_rx) = watch::channel(false);
    let bound_http = serve_axum(
        core.clone(),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        sync_stop_rx,
    )
    .await?;
    let sync_api_url = format!("http://{}/beofold/api/v1", bound_http);
    println!("[SyncServer] Listening on {}", sync_api_url);

    let mut handles: Vec<(Arc<SharedMeshState>, watch::Sender<bool>)> = Vec::new();
    for (cluster_idx, (cname, csecret, port_base)) in [
        ("Cluster-Alpha", "alpha-home-secret", 18510u16),
        ("Cluster-Beta", "beta-lab-secret", 18520u16),
    ]
    .iter()
    .enumerate()
    {
        let peers = vec![
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), *port_base),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), *port_base + 1),
        ];
        for node_i in 0..2usize {
            let (tx, rx) = watch::channel(false);
            let cfg = MeshNodeConfig {
                data_dir: base_dir.join(format!("c{}_n{}", cluster_idx, node_i)),
                bind_addr: peers[node_i],
                advertise_ip: Some(Ipv4Addr::LOCALHOST),
                cluster_secret: (*csecret).to_string(),
                seed_peers: peers.clone(),
                enable_mdns: false,
                enable_subnet_sweep: false,
                auto_throttle: false,
                max_system_cpu_pct: 95.0,
                bench_steps: 200,
                stop_when_idle_completed: None,
                node_label: format!("{}-n{}", cname, node_i),
                wan_sync_url: Some(sync_api_url.clone()),
                wan_cluster_name: (*cname).to_string(),
                enable_gpu: false,
            };
            let st = start_mesh_node(cfg, rx).await?;
            handles.push((st, tx));
        }
    }

    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let snap = core.store.build_global_telemetry(quorum.max(1), unix_now())?;
        if snap.completed_workunits >= expected_total {
            println!(
                "\n[FederationSuccess] Both remote clusters completed all {}/{} global PRCG chunks!",
                snap.completed_workunits, expected_total
            );

            let http = Client::new();
            let json_text = http
                .get(format!("{}/telemetry/global", sync_api_url))
                .header("accept", "application/json")
                .send()
                .await?
                .text()
                .await?;
            println!("\nGlobal Telemetry JSON (`GET /beofold/api/v1/telemetry/global`):");
            println!("{}", json_text);
            break;
        }
    }

    for (st, tx) in &handles {
        let _ = tx.send(true);
        st.transport
            .endpoint()
            .close(quinn::VarInt::from_u32(0), b"sim-complete");
    }
    let _ = sync_stop_tx.send(true);
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
