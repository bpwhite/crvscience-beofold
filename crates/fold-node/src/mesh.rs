use crate::discovery::{
    detect_primary_lan_ipv4, run_unicast_discovery_sweep, spawn_mdns_discovery,
};
use crate::quic_net::{bind_beowulf_endpoint, LanTransport, MAX_LAN_RPC_BYTES};
use crate::raft::{LanRaftState, RaftRole};
use crate::throttle::AutoThrottleMonitor;
use crate::wan_sync::run_wan_leader_gateway_loop;
use fold_physics::{
    benchmark_local_node_with_gpu, initialize_clone_checkpoint, minimize_steric_clashes,
    probe_gpu_adapter, run_generation_chunk_with_gpu, ChaCha8Rng, GpuForceEngine, LangevinConfig,
    ProteinTopology, SanityThresholds,
};
use fold_proto::{
    derive_cluster_mac_key, AuthenticatedLanFrame, ClusterStateSnapshot, LanLeasedUnit,
    LanRpcRequest, LanRpcResponse, LivePeerStatus, NodeCapabilityProfile, NodeIdentity,
    PeerEndpoint, ProjectManifest, RaftLogCommand, SignedWorkUnitResult, TrajectoryCheckpoint,
    WorkUnitId,
};
use fold_storage::{to_hex, CommitOutcome, FoldStore};
use quinn::VarInt;
use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, watch, Mutex, RwLock};

#[derive(Debug, Clone)]
pub struct MeshNodeConfig {
    pub data_dir: PathBuf,
    pub bind_addr: SocketAddr,
    pub advertise_ip: Option<Ipv4Addr>,
    pub cluster_secret: String,
    pub seed_peers: Vec<SocketAddr>,
    pub enable_mdns: bool,
    pub enable_subnet_sweep: bool,
    pub auto_throttle: bool,
    pub max_system_cpu_pct: f32,
    pub bench_steps: u64,
    pub stop_when_idle_completed: Option<usize>,
    pub node_label: String,
    pub wan_sync_url: Option<String>,
    pub wan_cluster_name: String,
    pub enable_gpu: bool,
}

#[derive(Default, Clone)]
struct WorkerLiveTelemetry {
    active_wu: Option<WorkUnitId>,
    step_progress: u64,
    target_steps: u64,
    current_pot_kcal: f64,
    cpu_usage_pct: f32,
    throttled: bool,
}

pub struct SharedMeshState {
    pub node_label: String,
    pub identity: NodeIdentity,
    pub capability: NodeCapabilityProfile,
    pub self_addr: SocketAddr,
    pub store: Arc<FoldStore>,
    pub transport: LanTransport,
    pub raft: Mutex<LanRaftState>,
    pub peers: RwLock<HashMap<[u8; 32], LivePeerStatus>>,
    pub local_worker_telemetry: RwLock<WorkerLiveTelemetry>,
    pub mremd_rng: Mutex<ChaCha8Rng>,
    pub throttle: Mutex<AutoThrottleMonitor>,
    pub project_seeded_in_raft: Mutex<bool>,
    pub enable_gpu: bool,
}

/// Boots a complete Beowulf LAN node (QUIC server + mDNS/subnet discovery + Raft voter +
/// optional outbound Apache WAN Leader Gateway + `wgpu`/CPU `prefetch_depth = 1` folding worker).
pub async fn start_mesh_node(
    cfg: MeshNodeConfig,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<Arc<SharedMeshState>, Box<dyn std::error::Error + Send + Sync>> {
    let store = Arc::new(FoldStore::open(&cfg.data_dir)?);
    let key_path = cfg.data_dir.join("node_ed25519.key");
    let identity = NodeIdentity::load_or_create_keyfile(&key_path)
        .unwrap_or_else(|_| NodeIdentity::from_cluster_secret(&cfg.cluster_secret, &cfg.node_label));

    let capability =
        benchmark_local_node_with_gpu(identity.public_key_bytes(), cfg.bench_steps, cfg.enable_gpu);
    let endpoint = bind_beowulf_endpoint(cfg.bind_addr)?;
    let actual_bound = endpoint.local_addr()?;

    let adv_ip = cfg.advertise_ip.unwrap_or_else(|| {
        if actual_bound.ip().is_loopback() {
            Ipv4Addr::LOCALHOST
        } else {
            detect_primary_lan_ipv4()
        }
    });
    let self_addr = SocketAddr::new(IpAddr::V4(adv_ip), actual_bound.port());

    let cluster_mac_key = derive_cluster_mac_key(&cfg.cluster_secret);
    let transport = LanTransport::new(
        endpoint.clone(),
        cluster_mac_key,
        identity.public_key_bytes(),
    );

    let raft = LanRaftState::new(identity.public_key_bytes(), self_addr.to_string());
    let mremd_rng = ChaCha8Rng::from_prcg_seed(WorkUnitId::new(1, 777, 777, 0), b"lan-mremd");

    let state = Arc::new(SharedMeshState {
        node_label: cfg.node_label.clone(),
        identity,
        capability: capability.clone(),
        self_addr,
        store,
        transport,
        raft: Mutex::new(raft),
        peers: RwLock::new(HashMap::new()),
        local_worker_telemetry: RwLock::new(WorkerLiveTelemetry::default()),
        mremd_rng: Mutex::new(mremd_rng),
        throttle: Mutex::new(AutoThrottleMonitor::new(
            cfg.auto_throttle,
            cfg.max_system_cpu_pct,
        )),
        project_seeded_in_raft: Mutex::new(false),
        enable_gpu: cfg.enable_gpu,
    });

    let gpu_note = if cfg.enable_gpu {
        probe_gpu_adapter()
            .map(|g| format!(" | GPU: {} ({})", g.name, g.backend))
            .unwrap_or_else(|| " | GPU: none (rayon CPU)".to_string())
    } else {
        " | GPU: disabled (--no-gpu)".to_string()
    };

    println!(
        "[Mesh:{}] Bound QUIC TLS 1.3 on {} (advertising {}) | {} | {:.2} ns/day{}",
        state.node_label,
        actual_bound,
        self_addr,
        state.capability.tier,
        state.capability.villin_ns_per_day,
        gpu_note
    );

    // 1. Inbound QUIC stream RPC server loop
    {
        let st = state.clone();
        let mut stop = shutdown_rx.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = stop.changed() => {
                        if *stop.borrow() { break; }
                    }
                    incoming_opt = st.transport.endpoint().accept() => {
                        let Some(incoming) = incoming_opt else { break; };
                        let st_conn = st.clone();
                        tokio::spawn(async move {
                            if let Ok(conn) = incoming.await {
                                loop {
                                    let Ok((mut send, mut recv)) = conn.accept_bi().await else {
                                        break;
                                    };
                                    let st_req = st_conn.clone();
                                    tokio::spawn(async move {
                                        let Ok(wire_req) = recv.read_to_end(MAX_LAN_RPC_BYTES).await else {
                                            return;
                                        };
                                        let Ok((_peer_pk, req)) = AuthenticatedLanFrame::open_request(
                                            st_req.transport.cluster_mac_key(),
                                            &wire_req,
                                        ) else {
                                            return;
                                        };
                                        let resp = handle_inbound_lan_rpc(&st_req, req).await;
                                        if let Ok(wire_resp) = AuthenticatedLanFrame::seal_response(
                                            st_req.transport.cluster_mac_key(),
                                            st_req.identity.public_key_bytes(),
                                            &resp,
                                        ) {
                                            let _ = send.write_all(&wire_resp).await;
                                            let _ = send.finish();
                                        }
                                    });
                                }
                            }
                        });
                    }
                }
            }
        });
    }

    // 2. mDNS responder + unicast /24 & seed discovery task
    let (mdns_tx, mut mdns_rx) = mpsc::unbounded_channel::<SocketAddr>();
    let mdns_daemon = if cfg.enable_mdns {
        spawn_mdns_discovery(
            &cfg.cluster_secret,
            &state.identity.short_id_hex(),
            adv_ip,
            self_addr.port(),
            &state.capability,
            mdns_tx,
        )
    } else {
        None
    };

    {
        let st = state.clone();
        let seeds = cfg.seed_peers.clone();
        let enable_sweep = cfg.enable_subnet_sweep;
        let mut stop = shutdown_rx.clone();
        tokio::spawn(async move {
            let _keep_mdns_alive = mdns_daemon;
            let mut dynamic_seeds = seeds;
            let mut interval = tokio::time::interval(Duration::from_millis(1200));

            loop {
                tokio::select! {
                    _ = stop.changed() => {
                        if *stop.borrow() { break; }
                    }
                    Some(mdns_addr) = mdns_rx.recv() => {
                        if !dynamic_seeds.contains(&mdns_addr) {
                            dynamic_seeds.push(mdns_addr);
                        }
                    }
                    _ = interval.tick() => {
                        let cur_term = st.raft.lock().await.current_term;
                        let known_len = st.peers.read().await.len();
                        let hits = run_unicast_discovery_sweep(
                            &st.transport,
                            st.self_addr,
                            &st.capability,
                            cur_term,
                            &dynamic_seeds,
                            enable_sweep,
                            known_len,
                        )
                        .await;

                        let now = unix_now();
                        for (ep, cap, peer_term, peer_leader) in hits {
                            {
                                let mut peers = st.peers.write().await;
                                peers
                                    .entry(ep.node_pubkey)
                                    .and_modify(|p| {
                                        p.addr = ep.addr.clone();
                                        p.capability = cap.clone();
                                        p.last_seen_unix_sec = now;
                                    })
                                    .or_insert_with(|| {
                                        println!(
                                            "[Mesh:{}] Discovered Beowulf peer {} ({}/{}, {}) at {}",
                                            st.node_label,
                                            &to_hex(&ep.node_pubkey)[0..8],
                                            cap.os,
                                            cap.arch,
                                            cap.tier,
                                            ep.addr
                                        );
                                        LivePeerStatus {
                                            capability: cap,
                                            addr: ep.addr.clone(),
                                            active_wu: None,
                                            step_progress: 0,
                                            target_steps: 0,
                                            current_pot_kcal: 0.0,
                                            cpu_usage_pct: 0.0,
                                            throttled: false,
                                            last_seen_unix_sec: now,
                                        }
                                    });
                            }
                            let mut raft = st.raft.lock().await;
                            if peer_term > raft.current_term {
                                raft.step_down(peer_term, peer_leader);
                            }
                        }
                    }
                }
            }
        });
    }

    // 3. Raft Leader Election & 150 ms AppendEntries Replication Loop
    let has_wan_sync = cfg.wan_sync_url.is_some();
    {
        let st = state.clone();
        let mut stop = shutdown_rx.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(140));
            loop {
                tokio::select! {
                    _ = stop.changed() => {
                        if *stop.borrow() { break; }
                    }
                    _ = ticker.tick() => {
                        raft_tick(&st, has_wan_sync).await;
                    }
                }
            }
        });
    }

    // 4. Outbound Apache HTTP Multi-Cluster Gateway (active when elected Leader)
    if let Some(sync_url) = cfg.wan_sync_url.clone() {
        let st = state.clone();
        let cname = cfg.wan_cluster_name.clone();
        let stop = shutdown_rx.clone();
        tokio::spawn(async move {
            run_wan_leader_gateway_loop(st, sync_url, cname, stop).await;
        });
    }

    // 5. Prefetch-Depth-1 Folding Worker (`wgpu` WGSL + `rayon` CPU) + 5-Second Heartbeat Loop
    {
        let st = state.clone();
        let stop_target = cfg.stop_when_idle_completed;
        tokio::spawn(async move {
            worker_pipeline_loop(st, stop_target, &mut shutdown_rx).await;
        });
    }

    Ok(state)
}

async fn raft_tick(st: &Arc<SharedMeshState>, has_wan_sync: bool) {
    let now = unix_now();
    let active_peers: Vec<(PeerEndpoint, SocketAddr)> = {
        let mut peers = st.peers.write().await;
        peers.retain(|_, p| now.saturating_sub(p.last_seen_unix_sec) <= 12);
        peers
            .iter()
            .filter_map(|(&pk, p)| {
                p.addr.parse::<SocketAddr>().ok().map(|sa| {
                    (
                        PeerEndpoint {
                            node_pubkey: pk,
                            addr: p.addr.clone(),
                        },
                        sa,
                    )
                })
            })
            .collect()
    };
    let peer_pks: Vec<[u8; 32]> = active_peers.iter().map(|(ep, _)| ep.node_pubkey).collect();

    let maybe_vote_req = {
        let mut raft = st.raft.lock().await;
        if raft.election_timed_out() {
            let total_voters = peer_pks.len() + 1;
            let vreq = raft.start_election(total_voters);
            if raft.role == RaftRole::Leader {
                println!(
                    "[Raft:{}] Elected Beowulf Cluster Leader for term {} (single-node quorum)",
                    st.node_label, raft.current_term
                );
            }
            vreq
        } else {
            None
        }
    };

    if let Some(vote_req) = maybe_vote_req {
        for (_ep, sock_addr) in &active_peers {
            let net = st.transport.clone();
            let req = LanRpcRequest::RaftRequestVote(vote_req.clone());
            let st_inner = st.clone();
            let pks_clone = peer_pks.clone();
            let target = *sock_addr;
            tokio::spawn(async move {
                if let Ok((_, LanRpcResponse::RaftVote(vresp))) =
                    net.rpc(target, &req, Duration::from_millis(250)).await
                {
                    let mut raft = st_inner.raft.lock().await;
                    if raft.record_vote_response(&vresp, &pks_clone) {
                        println!(
                            "[Raft:{}] Elected Beowulf Cluster Leader for term {} ({} peers)",
                            st_inner.node_label,
                            raft.current_term,
                            pks_clone.len()
                        );
                    }
                }
            });
        }
    }

    let is_leader = st.raft.lock().await.role == RaftRole::Leader;
    if is_leader {
        if !has_wan_sync {
            let mut seeded_flag = st.project_seeded_in_raft.lock().await;
            if !*seeded_flag {
                if let Ok(ids) = st.store.list_project_ids() {
                    if let Some(&pid) = ids.first() {
                        if let (Ok(Some((manifest, topo_blob))), Ok(gen0_blobs)) = (
                            st.store.export_project_bundle(pid),
                            st.store.export_gen0_blobs(pid),
                        ) {
                            let mut raft = st.raft.lock().await;
                            raft.append_leader_command(
                                RaftLogCommand::SeedProjectBundle {
                                    manifest,
                                    compressed_topology_blob: topo_blob,
                                    gen0_compressed_checkpoints: gen0_blobs,
                                },
                                peer_pks.len(),
                            );
                            *seeded_flag = true;
                        }
                    }
                }
            }
        }

        for (ep, sock_addr) in &active_peers {
            let ae_req = {
                let raft = st.raft.lock().await;
                if raft.role != RaftRole::Leader {
                    break;
                }
                raft.build_append_entries_for(&ep.node_pubkey)
            };
            let net = st.transport.clone();
            let st_inner = st.clone();
            let pks_clone = peer_pks.clone();
            let target = *sock_addr;
            let target_pk = ep.node_pubkey;

            tokio::spawn(async move {
                let rpc = LanRpcRequest::RaftAppendEntries(ae_req);
                if let Ok((_, LanRpcResponse::RaftAppendEntries(aresp))) =
                    net.rpc(target, &rpc, Duration::from_millis(350)).await
                {
                    st_inner.peers.write().await.entry(target_pk).and_modify(|p| {
                        p.last_seen_unix_sec = unix_now();
                    });
                    let mut raft = st_inner.raft.lock().await;
                    raft.record_append_response(&aresp, &pks_clone);
                }
            });
        }
    }
}

async fn handle_inbound_lan_rpc(st: &Arc<SharedMeshState>, req: LanRpcRequest) -> LanRpcResponse {
    let now = unix_now();

    match req {
        LanRpcRequest::PingHandshake {
            sender_addr,
            capability,
            raft_term,
        } => {
            if capability.node_pubkey != st.identity.public_key_bytes() {
                st.peers
                    .write()
                    .await
                    .entry(capability.node_pubkey)
                    .and_modify(|p| {
                        p.addr = sender_addr.clone();
                        p.capability = capability.clone();
                        p.last_seen_unix_sec = now;
                    })
                    .or_insert(LivePeerStatus {
                        capability: capability.clone(),
                        addr: sender_addr,
                        active_wu: None,
                        step_progress: 0,
                        target_steps: 0,
                        current_pot_kcal: 0.0,
                        cpu_usage_pct: 0.0,
                        throttled: false,
                        last_seen_unix_sec: now,
                    });
            }

            let mut raft = st.raft.lock().await;
            if raft_term > raft.current_term {
                raft.step_down(raft_term, None);
            }
            LanRpcResponse::PongHandshake {
                responder_addr: st.self_addr.to_string(),
                capability: st.capability.clone(),
                raft_term: raft.current_term,
                leader: raft.current_leader.clone(),
            }
        }

        LanRpcRequest::RaftRequestVote(vreq) => {
            let mut raft = st.raft.lock().await;
            LanRpcResponse::RaftVote(raft.handle_vote_request(&vreq))
        }

        LanRpcRequest::RaftAppendEntries(aereq) => {
            let leader_pk = aereq.leader_pubkey;
            let (resp, newly_committed) = {
                let mut raft = st.raft.lock().await;
                raft.handle_append_entries(aereq)
            };
            st.peers.write().await.entry(leader_pk).and_modify(|p| {
                p.last_seen_unix_sec = now;
            });
            for cmd in newly_committed {
                let _ = st.store.apply_replicated_raft_command(&cmd);
            }
            LanRpcResponse::RaftAppendEntries(resp)
        }

        LanRpcRequest::FetchProjectBundle { project_id } => {
            let (is_leader, leader_ep) = {
                let raft = st.raft.lock().await;
                (raft.role == RaftRole::Leader, raft.current_leader.clone())
            };
            if !is_leader {
                return LanRpcResponse::ProjectBundle {
                    found: None,
                    redirect_leader: leader_ep,
                };
            }
            let found = st.store.export_project_bundle(project_id).ok().flatten();
            LanRpcResponse::ProjectBundle {
                found,
                redirect_leader: None,
            }
        }

        LanRpcRequest::AcquireLanLease {
            worker,
            worker_addr,
            max_units,
        } => {
            if worker.node_pubkey != st.identity.public_key_bytes() {
                st.peers
                    .write()
                    .await
                    .entry(worker.node_pubkey)
                    .and_modify(|p| {
                        p.last_seen_unix_sec = now;
                        p.addr = worker_addr.clone();
                    })
                    .or_insert(LivePeerStatus {
                        capability: worker.clone(),
                        addr: worker_addr,
                        active_wu: None,
                        step_progress: 0,
                        target_steps: 0,
                        current_pot_kcal: 0.0,
                        cpu_usage_pct: 0.0,
                        throttled: false,
                        last_seen_unix_sec: now,
                    });
            }

            let (is_leader, leader_ep) = {
                let raft = st.raft.lock().await;
                (raft.role == RaftRole::Leader, raft.current_leader.clone())
            };
            if !is_leader {
                return LanRpcResponse::LanLeaseBatch {
                    units: Vec::new(),
                    active_project_id: None,
                    redirect_leader: leader_ep,
                };
            }

            let active_pid = st
                .store
                .list_project_ids()
                .ok()
                .and_then(|v| v.into_iter().next());
            let mut units = Vec::new();
            let count = max_units.clamp(1, 2) as usize;
            let peer_count = st.peers.read().await.len();

            for _ in 0..count {
                let Ok(Some((lease, cp))) =
                    st.store.acquire_next_lease(worker.node_pubkey, now, 30)
                else {
                    break;
                };
                let Ok(compressed_input_checkpoint) = cp.to_compressed_frame() else {
                    break;
                };
                {
                    let mut raft = st.raft.lock().await;
                    raft.append_leader_command(
                        RaftLogCommand::LeaseIssued {
                            lease: lease.clone(),
                        },
                        peer_count,
                    );
                }
                units.push(LanLeasedUnit {
                    project_id: lease.wu_id.project,
                    lease,
                    compressed_input_checkpoint,
                });
            }

            LanRpcResponse::LanLeaseBatch {
                units,
                active_project_id: active_pid,
                redirect_leader: None,
            }
        }

        LanRpcRequest::SubmitLanResult {
            receipt,
            compressed_checkpoint_blob,
        } => {
            let (is_leader, leader_ep) = {
                let raft = st.raft.lock().await;
                (raft.role == RaftRole::Leader, raft.current_leader.clone())
            };
            if !is_leader {
                return LanRpcResponse::SubmitAck {
                    accepted: false,
                    quarantined: false,
                    message: "not active Raft leader".into(),
                    spawned_next_gen: None,
                    redirect_leader: leader_ep,
                };
            }

            let Ok((manifest, topo)) = st.store.load_project(receipt.wu_id.project) else {
                return LanRpcResponse::Error("unknown project ID".into());
            };
            let Ok(output_cp) =
                TrajectoryCheckpoint::from_compressed_frame(&compressed_checkpoint_blob)
            else {
                return LanRpcResponse::Error("malformed LZ4 checkpoint frame".into());
            };

            let outcome = {
                let mut rng = st.mremd_rng.lock().await;
                st.store.commit_signed_result(
                    &manifest,
                    &topo,
                    &receipt,
                    &output_cp,
                    &SanityThresholds::default(),
                    &mut rng,
                    now,
                )
            };

            let peer_count = st.peers.read().await.len();
            match outcome {
                Ok(CommitOutcome::Accepted {
                    completed_wu,
                    spawned_next_gen,
                    next_temperature_kelvin,
                    mremd_swapped_with,
                }) => {
                    {
                        let mut raft = st.raft.lock().await;
                        raft.append_leader_command(
                            RaftLogCommand::GenerationCommitted {
                                receipt: receipt.clone(),
                                compressed_output_checkpoint: compressed_checkpoint_blob,
                                next_temperature_kelvin,
                                spawned_next_gen,
                            },
                            peer_count,
                        );
                    }
                    let swap_str = mremd_swapped_with
                        .map(|p| format!(" (MREMD swap <-> {})", p))
                        .unwrap_or_default();
                    println!(
                        "[Leader:{}] Committed {} from worker {}.. | V_pot={:.2} kcal/mol{}",
                        st.node_label,
                        completed_wu,
                        &to_hex(&receipt.worker_pubkey)[0..8],
                        receipt.final_potential_kcal,
                        swap_str
                    );
                    LanRpcResponse::SubmitAck {
                        accepted: true,
                        quarantined: false,
                        message: format!("accepted {}{}", completed_wu, swap_str),
                        spawned_next_gen,
                        redirect_leader: None,
                    }
                }
                Ok(CommitOutcome::AwaitingSecondQuorum { wu_id }) => LanRpcResponse::SubmitAck {
                    accepted: true,
                    quarantined: false,
                    message: format!("staged {} for quorum", wu_id),
                    spawned_next_gen: None,
                    redirect_leader: None,
                },
                Ok(CommitOutcome::Quarantined { wu_id, reason }) => {
                    {
                        let mut raft = st.raft.lock().await;
                        raft.append_leader_command(
                            RaftLogCommand::GenerationQuarantined {
                                wu_id,
                                worker_pubkey: receipt.worker_pubkey,
                                reason: reason.clone(),
                                timestamp_unix_sec: now,
                            },
                            peer_count,
                        );
                    }
                    LanRpcResponse::SubmitAck {
                        accepted: false,
                        quarantined: true,
                        message: reason,
                        spawned_next_gen: None,
                        redirect_leader: None,
                    }
                }
                Err(e) => LanRpcResponse::Error(e.to_string()),
            }
        }

        LanRpcRequest::WorkerHeartbeat {
            worker,
            worker_addr,
            active_wu,
            step_progress,
            target_steps,
            current_pot_kcal,
            cpu_usage_pct,
            throttled,
        } => {
            if let Some(wu) = active_wu {
                let _ = st.store.touch_lease(wu, now + 30);
            }
            st.peers.write().await.insert(
                worker.node_pubkey,
                LivePeerStatus {
                    capability: worker,
                    addr: worker_addr,
                    active_wu,
                    step_progress,
                    target_steps,
                    current_pot_kcal,
                    cpu_usage_pct,
                    throttled,
                    last_seen_unix_sec: now,
                },
            );
            let raft = st.raft.lock().await;
            LanRpcResponse::HeartbeatAck {
                raft_term: raft.current_term,
                leader: raft.current_leader.clone(),
            }
        }

        LanRpcRequest::QueryClusterState => {
            let (term, leader) = {
                let raft = st.raft.lock().await;
                (raft.current_term, raft.current_leader.clone())
            };
            let pid = st
                .store
                .list_project_ids()
                .ok()
                .and_then(|v| v.into_iter().next());
            let pname = pid
                .and_then(|id| st.store.load_project(id).ok())
                .map(|(m, _)| m.name);
            let stats = st.store.collect_stats(pid).unwrap_or_default();

            let mut all_peers: Vec<LivePeerStatus> =
                st.peers.read().await.values().cloned().collect();
            let self_tel = st.local_worker_telemetry.read().await.clone();
            all_peers.push(LivePeerStatus {
                capability: st.capability.clone(),
                addr: st.self_addr.to_string(),
                active_wu: self_tel.active_wu,
                step_progress: self_tel.step_progress,
                target_steps: self_tel.target_steps,
                current_pot_kcal: self_tel.current_pot_kcal,
                cpu_usage_pct: self_tel.cpu_usage_pct,
                throttled: self_tel.throttled,
                last_seen_unix_sec: now,
            });

            LanRpcResponse::ClusterState(ClusterStateSnapshot {
                raft_term: term,
                leader,
                active_project_id: pid,
                active_project_name: pname,
                peers: all_peers,
                total_workunits: stats.total_workunits as u64,
                queued_workunits: stats.queued as u64,
                leased_workunits: stats.leased as u64,
                completed_workunits: stats.completed as u64,
                mremd_swaps: stats.mremd_swaps,
                mremd_trials: stats.mremd_trials,
                quarantined_events: stats.quarantined_events,
                lowest_potential_kcal: stats
                    .lowest_potential_kcal
                    .map(|(wu, e, _)| (wu, e)),
            })
        }
    }
}

type CachedProjectEntry = (
    ProjectManifest,
    Arc<ProteinTopology>,
    Option<Arc<GpuForceEngine>>,
);

async fn worker_pipeline_loop(
    st: Arc<SharedMeshState>,
    stop_when_completed: Option<usize>,
    shutdown_rx: &mut watch::Receiver<bool>,
) {
    let mut staged_queue: VecDeque<LanLeasedUnit> = VecDeque::new();
    let mut cached_projects: HashMap<u32, CachedProjectEntry> = HashMap::new();

    {
        let st_hb = st.clone();
        let mut stop_hb = shutdown_rx.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    _ = stop_hb.changed() => {
                        if *stop_hb.borrow() { break; }
                    }
                    _ = ticker.tick() => {
                        let leader_ep = st_hb.raft.lock().await.current_leader.clone();
                        let Some(leader) = leader_ep else { continue; };
                        if leader.node_pubkey == st_hb.identity.public_key_bytes() {
                            if let Some(wu) = st_hb.local_worker_telemetry.read().await.active_wu {
                                let _ = st_hb.store.touch_lease(wu, unix_now() + 30);
                            }
                            continue;
                        }
                        if let Ok(target_sock) = leader.addr.parse::<SocketAddr>() {
                            let tel = st_hb.local_worker_telemetry.read().await.clone();
                            let hb_req = LanRpcRequest::WorkerHeartbeat {
                                worker: st_hb.capability.clone(),
                                worker_addr: st_hb.self_addr.to_string(),
                                active_wu: tel.active_wu,
                                step_progress: tel.step_progress,
                                target_steps: tel.target_steps,
                                current_pot_kcal: tel.current_pot_kcal,
                                cpu_usage_pct: tel.cpu_usage_pct,
                                throttled: tel.throttled,
                            };
                            let _ = st_hb.transport.rpc(target_sock, &hb_req, Duration::from_secs(2)).await;
                        }
                    }
                }
            }
        });
    }

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        if let Some(target_done) = stop_when_completed {
            if let Ok(stats) = st.store.collect_stats(None) {
                if stats.completed >= target_done && stats.queued == 0 && stats.leased == 0 {
                    break;
                }
            }
        }

        {
            let (cpu_pct, should_pause) = st.throttle.lock().await.sample(false);
            let mut tel = st.local_worker_telemetry.write().await;
            tel.cpu_usage_pct = cpu_pct;
            tel.throttled = should_pause;
            if should_pause {
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        }

        if staged_queue.is_empty() {
            if let Some(leader) = st.raft.lock().await.current_leader.clone() {
                let batch = request_leases_from_leader(&st, &leader, 2).await;
                for u in batch {
                    staged_queue.push_back(u);
                }
            }
        }

        let Some(unit) = staged_queue.pop_front() else {
            tokio::time::sleep(Duration::from_millis(250)).await;
            continue;
        };

        if !cached_projects.contains_key(&unit.project_id) {
            if let Ok((m, t)) = st.store.load_project(unit.project_id) {
                let gpu_opt = if st.enable_gpu {
                    GpuForceEngine::try_new(&t).ok().map(Arc::new)
                } else {
                    None
                };
                cached_projects.insert(unit.project_id, (m, Arc::new(t), gpu_opt));
            } else if let Some(leader) = st.raft.lock().await.current_leader.clone() {
                if let Some((m, topo_blob)) =
                    fetch_project_from_leader(&st, &leader, unit.project_id).await
                {
                    if let Ok(topo) = ProteinTopology::from_compressed_bytes(&topo_blob) {
                        let gpu_opt = if st.enable_gpu {
                            GpuForceEngine::try_new(&topo).ok().map(Arc::new)
                        } else {
                            None
                        };
                        cached_projects.insert(unit.project_id, (m, Arc::new(topo), gpu_opt));
                    }
                }
            }
        }

        let Some((manifest, topo_arc, gpu_opt)) = cached_projects.get(&unit.project_id).cloned()
        else {
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        };

        let Ok(input_cp) =
            TrajectoryCheckpoint::from_compressed_frame(&unit.compressed_input_checkpoint)
        else {
            continue;
        };

        {
            let mut tel = st.local_worker_telemetry.write().await;
            tel.active_wu = Some(unit.lease.wu_id);
            tel.step_progress = 0;
            tel.target_steps = unit.lease.target_steps;
            tel.current_pot_kcal = input_cp.potential_energy_kcal;
        }

        if staged_queue.is_empty() {
            if let Some(leader) = st.raft.lock().await.current_leader.clone() {
                let st_pf = st.clone();
                let prefetch_handle =
                    tokio::spawn(
                        async move { request_leases_from_leader(&st_pf, &leader, 1).await },
                    );

                let cfg = LangevinConfig {
                    dt_fs: manifest.dt_fs,
                    friction_per_ps: manifest.friction_per_ps,
                    temperature_kelvin: input_cp.temperature_kelvin,
                    steps: unit.lease.target_steps,
                    sample_stride: (unit.lease.target_steps / 20).max(1),
                };
                let wu_id = unit.lease.wu_id;
                let t0 = Instant::now();
                let fold_res = tokio::task::spawn_blocking(move || {
                    run_generation_chunk_with_gpu(
                        &topo_arc,
                        gpu_opt.as_deref(),
                        &input_cp,
                        wu_id,
                        &cfg,
                    )
                })
                .await;

                if let Ok(prefetched) = prefetch_handle.await {
                    for pf_u in prefetched {
                        staged_queue.push_back(pf_u);
                    }
                }

                if let Ok(summary) = fold_res {
                    let wall_ms = t0.elapsed().as_millis() as u64;
                    submit_completed_unit(&st, &unit, summary, wall_ms).await;
                }
            }
        } else {
            let cfg = LangevinConfig {
                dt_fs: manifest.dt_fs,
                friction_per_ps: manifest.friction_per_ps,
                temperature_kelvin: input_cp.temperature_kelvin,
                steps: unit.lease.target_steps,
                sample_stride: (unit.lease.target_steps / 20).max(1),
            };
            let wu_id = unit.lease.wu_id;
            let t0 = Instant::now();
            if let Ok(summary) = tokio::task::spawn_blocking(move || {
                run_generation_chunk_with_gpu(
                    &topo_arc,
                    gpu_opt.as_deref(),
                    &input_cp,
                    wu_id,
                    &cfg,
                )
            })
            .await
            {
                let wall_ms = t0.elapsed().as_millis() as u64;
                submit_completed_unit(&st, &unit, summary, wall_ms).await;
            }
        }

        st.local_worker_telemetry.write().await.active_wu = None;
    }
}

async fn request_leases_from_leader(
    st: &Arc<SharedMeshState>,
    leader: &PeerEndpoint,
    max_units: u8,
) -> Vec<LanLeasedUnit> {
    let req = LanRpcRequest::AcquireLanLease {
        worker: st.capability.clone(),
        worker_addr: st.self_addr.to_string(),
        max_units,
    };

    if leader.node_pubkey == st.identity.public_key_bytes() {
        if let LanRpcResponse::LanLeaseBatch { units, .. } = handle_inbound_lan_rpc(st, req).await {
            return units;
        }
        return Vec::new();
    }

    let Ok(sock) = leader.addr.parse::<SocketAddr>() else {
        return Vec::new();
    };
    match st.transport.rpc(sock, &req, Duration::from_secs(3)).await {
        Ok((
            _,
            LanRpcResponse::LanLeaseBatch {
                units,
                redirect_leader,
                ..
            },
        )) => {
            if let Some(new_ldr) = redirect_leader {
                st.raft.lock().await.current_leader = Some(new_ldr);
            }
            units
        }
        _ => Vec::new(),
    }
}

async fn fetch_project_from_leader(
    st: &Arc<SharedMeshState>,
    leader: &PeerEndpoint,
    project_id: u32,
) -> Option<(ProjectManifest, Vec<u8>)> {
    let req = LanRpcRequest::FetchProjectBundle { project_id };
    if leader.node_pubkey == st.identity.public_key_bytes() {
        if let LanRpcResponse::ProjectBundle { found, .. } = handle_inbound_lan_rpc(st, req).await {
            return found;
        }
        return None;
    }
    let sock = leader.addr.parse::<SocketAddr>().ok()?;
    if let Ok((_, LanRpcResponse::ProjectBundle { found, .. })) =
        st.transport.rpc(sock, &req, Duration::from_secs(4)).await
    {
        return found;
    }
    None
}

async fn submit_completed_unit(
    st: &Arc<SharedMeshState>,
    unit: &LanLeasedUnit,
    summary: fold_physics::GenerationSummary,
    wall_ms: u64,
) {
    let Ok(out_blob) = summary.checkpoint.to_compressed_frame() else {
        return;
    };
    let out_hash = *blake3::hash(&out_blob).as_bytes();

    let mut receipt = SignedWorkUnitResult {
        wu_id: unit.lease.wu_id,
        lease_token: unit.lease.lease_token,
        worker_pubkey: [0u8; 32],
        input_checkpoint_blake3: unit.lease.input_checkpoint_blake3,
        output_checkpoint_blake3: out_hash,
        steps_executed: unit.lease.target_steps,
        temperature_kelvin: summary.checkpoint.temperature_kelvin,
        mean_potential_kcal: summary.mean_potential_kcal,
        min_potential_kcal: summary.min_potential_kcal,
        final_potential_kcal: summary.checkpoint.potential_energy_kcal,
        final_kinetic_kcal: summary.checkpoint.kinetic_energy_kcal,
        final_rg_angstrom: summary.checkpoint.rg_angstrom,
        final_sasa_angstrom2: summary.checkpoint.sasa_angstrom2,
        wall_time_ms: wall_ms,
        signature: [0u8; 64],
    };
    st.identity.sign_result(&mut receipt);

    let submit_req = LanRpcRequest::SubmitLanResult {
        receipt,
        compressed_checkpoint_blob: out_blob,
    };

    for _attempt in 0..6 {
        let Some(leader) = st.raft.lock().await.current_leader.clone() else {
            tokio::time::sleep(Duration::from_millis(350)).await;
            continue;
        };

        if leader.node_pubkey == st.identity.public_key_bytes() {
            if let LanRpcResponse::SubmitAck { accepted: true, .. } =
                handle_inbound_lan_rpc(st, submit_req.clone()).await
            {
                return;
            }
        } else if let Ok(sock) = leader.addr.parse::<SocketAddr>() {
            if let Ok((
                _,
                LanRpcResponse::SubmitAck {
                    accepted,
                    quarantined,
                    redirect_leader,
                    ..
                },
            )) = st
                .transport
                .rpc(sock, &submit_req, Duration::from_secs(4))
                .await
            {
                if accepted || quarantined {
                    return;
                }
                if let Some(new_ldr) = redirect_leader {
                    st.raft.lock().await.current_leader = Some(new_ldr);
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
}

pub async fn query_remote_cluster_status(
    peer_addr: SocketAddr,
    cluster_secret: &str,
) -> Result<ClusterStateSnapshot, Box<dyn std::error::Error + Send + Sync>> {
    let bind: SocketAddr = "0.0.0.0:0".parse()?;
    let ep = bind_beowulf_endpoint(bind)?;
    let id = NodeIdentity::generate_random()?;
    let mac_key = derive_cluster_mac_key(cluster_secret);
    let net = LanTransport::new(ep, mac_key, id.public_key_bytes());

    let (_, resp) = net
        .rpc(
            peer_addr,
            &LanRpcRequest::QueryClusterState,
            Duration::from_secs(3),
        )
        .await?;
    match resp {
        LanRpcResponse::ClusterState(mut snap) => {
            if let Some(ref ldr) = snap.leader {
                if let Ok(ldr_sock) = ldr.addr.parse::<SocketAddr>() {
                    if ldr_sock != peer_addr {
                        if let Ok((_, LanRpcResponse::ClusterState(ldr_snap))) = net
                            .rpc(
                                ldr_sock,
                                &LanRpcRequest::QueryClusterState,
                                Duration::from_secs(3),
                            )
                            .await
                        {
                            snap = ldr_snap;
                        }
                    }
                }
            }
            Ok(snap)
        }
        other => Err(format!("unexpected cluster status response: {:?}", other).into()),
    }
}

pub async fn run_multi_node_failover_simulation(
    base_dir: PathBuf,
    num_nodes: usize,
    generations: u32,
    steps_per_gen: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let n = num_nodes.max(3);
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir)?;

    println!("=========================================================================");
    println!(
        " BeoFold Phase 2 — {}-Node QUIC + Raft Beowulf Cluster & Failover Test",
        n
    );
    println!("=========================================================================");

    let base_port = 17443u16;
    let addrs: Vec<SocketAddr> = (0..n)
        .map(|i| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), base_port + (i as u16)))
        .collect();

    let node0_dir = base_dir.join("node_0");
    let store0 = FoldStore::open(&node0_dir)?;
    let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
    minimize_steric_clashes(&topo, &mut coords, 100);

    let ladder = vec![290.0, 320.0, 355.0, 395.0];
    let clones = 2u32;
    let total_expected_chunks = (ladder.len()) * (clones as usize) * (generations as usize);
    let mut gen0 = Vec::new();
    for (r, &t) in ladder.iter().enumerate() {
        for c in 0..clones {
            gen0.push(initialize_clone_checkpoint(
                &topo,
                &coords,
                WorkUnitId::new(1, r as u32, c, 0),
                t,
                b"phase2-failover-sim",
            ));
        }
    }
    let manifest = ProjectManifest {
        project_id: 1,
        name: "TrpCage_1L2Y_Cluster".into(),
        sequence: topo.sequence_one_letter.clone(),
        topology_blake3: [0u8; 32],
        initial_coords_blake3: [0u8; 32],
        atom_count: topo.atoms.len() as u32,
        residue_count: topo.residue_count,
        temp_ladder_kelvin: ladder,
        dt_fs: 1.5,
        friction_per_ps: 1.0,
        steps_per_gen: steps_per_gen.max(50),
        num_clones_per_run: clones,
        max_generations: generations.max(2),
    };
    store0.seed_project(manifest, &topo, &gen0)?;
    drop(store0);

    let mut nodes: Vec<(Arc<SharedMeshState>, watch::Sender<bool>)> = Vec::new();
    for i in 0..n {
        let (tx, rx) = watch::channel(false);
        let cfg = MeshNodeConfig {
            data_dir: base_dir.join(format!("node_{}", i)),
            bind_addr: addrs[i],
            advertise_ip: Some(Ipv4Addr::LOCALHOST),
            cluster_secret: "beofold-sim-secret".into(),
            seed_peers: addrs.clone(),
            enable_mdns: false,
            enable_subnet_sweep: false,
            auto_throttle: false,
            max_system_cpu_pct: 95.0,
            bench_steps: 250,
            stop_when_idle_completed: None,
            node_label: format!("node-{}", i),
            wan_sync_url: None,
            wan_cluster_name: "local-sim".into(),
            enable_gpu: false,
        };
        let st = start_mesh_node(cfg, rx).await?;
        nodes.push((st, tx));
    }

    tokio::time::sleep(Duration::from_millis(1500)).await;

    let mut killed_first_leader = false;
    let deadline = Instant::now() + Duration::from_secs(45);

    while Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(400)).await;

        let mut leader_idx: Option<usize> = None;
        for (idx, (st, tx)) in nodes.iter().enumerate() {
            if *tx.borrow() {
                continue;
            }
            if st.raft.lock().await.role == RaftRole::Leader {
                leader_idx = Some(idx);
                break;
            }
        }

        let Some(l_idx) = leader_idx else {
            continue;
        };
        let stats = nodes[l_idx].0.store.collect_stats(Some(1))?;

        if !killed_first_leader && stats.completed >= (total_expected_chunks / 4).max(2) {
            tokio::time::sleep(Duration::from_millis(250)).await;
            println!(
                "\n[Chaos] >>> Killing active Cluster Leader '{}' (completed {}/{} chunks) to test Raft failover! <<<\n",
                nodes[l_idx].0.node_label, stats.completed, total_expected_chunks
            );
            let _ = nodes[l_idx].1.send(true);
            nodes[l_idx]
                .0
                .transport
                .endpoint()
                .close(VarInt::from_u32(0), b"simulated-crash");
            killed_first_leader = true;
            continue;
        }

        if stats.completed >= total_expected_chunks {
            println!(
                "\n[Success] Surviving Beowulf cluster Leader '{}' completed all {}/{} PRCG chunks (MREMD swaps: {}/{})!",
                nodes[l_idx].0.node_label,
                stats.completed,
                total_expected_chunks,
                stats.mremd_swaps,
                stats.mremd_trials
            );
            break;
        }
    }

    for (st, tx) in &nodes {
        let _ = tx.send(true);
        st.transport.endpoint().close(VarInt::from_u32(0), b"shutdown");
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
