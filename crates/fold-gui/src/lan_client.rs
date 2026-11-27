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

use fold_proto::{
    derive_cluster_id_tag, derive_cluster_mac_key, AuthenticatedLanFrame, ClusterStateSnapshot,
    GlobalFederationTelemetry, LanRpcRequest, LanRpcResponse, NodeIdentity, CONTENT_TYPE_RKYV,
};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{
    ClientConfig, Connection, Endpoint, IdleTimeout, ServerConfig, TransportConfig, VarInt,
};
use reqwest::Client;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{
    ring::default_provider, verify_tls12_signature, verify_tls13_signature, CryptoProvider,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;

const MAX_GUI_RPC_BYTES: usize = 32 * 1024 * 1024;
const ALPN_BEOFOLD_LAN: &[u8] = b"beofold-lan-v2";
pub const MDNS_SERVICE_TYPE: &str = "_beofold._udp.local.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredLanPeer {
    pub addr: SocketAddr,
    pub instance: String,
    pub cluster_tag: String,
    pub node_short: String,
    pub tier: String,
    pub os: String,
    pub cores: String,
    pub matches_secret: bool,
}

/// Live network snapshot shared lock-free / low-contention with the 60 FPS `egui` render thread.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct NetworkTelemetryState {
    pub lan_target: String,
    pub lan_secret: String,
    pub lan_connected: bool,
    pub lan_last_error: Option<String>,
    pub lan_snapshot: Option<ClusterStateSnapshot>,
    pub lan_discovery_scanning: bool,
    pub lan_discovery_status: Option<String>,
    pub discovered_lan_peers: Vec<DiscoveredLanPeer>,

    pub wan_sync_url: String,
    pub wan_connected: bool,
    pub wan_last_error: Option<String>,
    pub wan_snapshot: Option<GlobalFederationTelemetry>,
}

#[derive(Debug, Clone)]
pub struct PollerTargets {
    pub lan_peer: Option<SocketAddr>,
    pub cluster_secret: String,
    pub wan_sync_url: Option<String>,
}

#[derive(Debug)]
struct LanSelfSignedVerifier(Arc<CryptoProvider>);

impl ServerCertVerifier for LanSelfSignedVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn build_gui_quic_endpoint() -> Result<Endpoint, String> {
    let provider = Arc::new(default_provider());
    let _ = default_provider().install_default();

    let cert = rcgen::generate_simple_self_signed(vec!["beofold-gui.local".to_string()])
        .map_err(|e| e.to_string())?;
    let key_der = PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
    let cert_der: CertificateDer<'static> = cert.cert.into();

    let mut server_crypto = rustls::ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], PrivateKeyDer::Pkcs8(key_der))
        .map_err(|e| e.to_string())?;
    server_crypto.alpn_protocols = vec![ALPN_BEOFOLD_LAN.to_vec()];

    let quic_server = QuicServerConfig::try_from(server_crypto).map_err(|e| e.to_string())?;
    let mut server_cfg = ServerConfig::with_crypto(Arc::new(quic_server));

    let mut transport = TransportConfig::default();
    if let Ok(idle) = IdleTimeout::try_from(Duration::from_secs(20)) {
        transport.max_idle_timeout(Some(idle));
    }
    transport.keep_alive_interval(Some(Duration::from_secs(6)));
    transport.stream_receive_window(VarInt::from_u32(32 * 1024 * 1024));
    let transport_arc = Arc::new(transport);
    server_cfg.transport_config(transport_arc.clone());

    let mut client_crypto = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(LanSelfSignedVerifier(provider)))
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![ALPN_BEOFOLD_LAN.to_vec()];

    let quic_client = QuicClientConfig::try_from(client_crypto).map_err(|e| e.to_string())?;
    let mut client_cfg = ClientConfig::new(Arc::new(quic_client));
    client_cfg.transport_config(transport_arc);

    let bind_addr: SocketAddr = "0.0.0.0:0".parse().map_err(|e: std::net::AddrParseError| e.to_string())?;
    let mut ep = Endpoint::server(server_cfg, bind_addr).map_err(|e| e.to_string())?;
    ep.set_default_client_config(client_cfg);
    Ok(ep)
}

async fn query_lan_peer_once(
    ep: &Endpoint,
    cached_conn: &mut Option<(SocketAddr, Connection)>,
    peer_addr: SocketAddr,
    cluster_secret: &str,
    gui_pubkey: [u8; 32],
) -> Result<ClusterStateSnapshot, String> {
    let mac_key = derive_cluster_mac_key(cluster_secret);
    let snap = rpc_query_state(ep, cached_conn, peer_addr, &mac_key, gui_pubkey).await?;

    if let Some(ref ldr) = snap.leader {
        if let Ok(ldr_sock) = ldr.addr.parse::<SocketAddr>() {
            if ldr_sock != peer_addr {
                if let Ok(ldr_snap) =
                    rpc_query_state(ep, cached_conn, ldr_sock, &mac_key, gui_pubkey).await
                {
                    return Ok(ldr_snap);
                }
            }
        }
    }
    Ok(snap)
}

async fn rpc_query_state(
    ep: &Endpoint,
    cached_conn: &mut Option<(SocketAddr, Connection)>,
    addr: SocketAddr,
    mac_key: &[u8; 32],
    gui_pubkey: [u8; 32],
) -> Result<ClusterStateSnapshot, String> {
    let wire_req =
        AuthenticatedLanFrame::seal_request(mac_key, gui_pubkey, &LanRpcRequest::QueryClusterState)
            .map_err(|e| e.to_string())?;

    let conn = if let Some((prev_addr, ref conn)) = *cached_conn {
        if prev_addr == addr && conn.close_reason().is_none() {
            Some(conn.clone())
        } else {
            None
        }
    } else {
        None
    };

    let conn = match conn {
        Some(c) => c,
        None => {
            let connecting = ep
                .connect(addr, "beofold.local")
                .map_err(|e| e.to_string())?;
            let c = tokio::time::timeout(Duration::from_millis(1800), connecting)
                .await
                .map_err(|_| format!("QUIC connect timeout to {}", addr))?
                .map_err(|e| e.to_string())?;
            *cached_conn = Some((addr, c.clone()));
            c
        }
    };
    let (mut send, mut recv) = tokio::time::timeout(Duration::from_millis(1500), conn.open_bi())
        .await
        .map_err(|_| "QUIC stream open timeout".to_string())?
        .map_err(|e| {
            *cached_conn = None;
            e.to_string()
        })?;

    send.write_all(&wire_req)
        .await
        .map_err(|e| e.to_string())?;
    send.finish().map_err(|e| e.to_string())?;

    let wire_resp = tokio::time::timeout(
        Duration::from_millis(2000),
        recv.read_to_end(MAX_GUI_RPC_BYTES),
    )
    .await
    .map_err(|_| "QUIC read timeout".to_string())?
    .map_err(|e| e.to_string())?;

    let (_, resp) =
        AuthenticatedLanFrame::open_response(mac_key, &wire_resp).map_err(|e| e.to_string())?;
    match resp {
        LanRpcResponse::ClusterState(s) => Ok(s),
        LanRpcResponse::Error(msg) => Err(msg),
        other => Err(format!("unexpected RPC reply: {:?}", other)),
    }
}

async fn query_wan_telemetry_once(
    http: &Client,
    sync_url: &str,
) -> Result<GlobalFederationTelemetry, String> {
    let base = sync_url.trim_end_matches('/');
    let url = if base.ends_with("/telemetry/global") {
        base.to_string()
    } else if base.ends_with("/api/v1") {
        format!("{}/telemetry/global", base)
    } else {
        format!("{}/api/v1/telemetry/global", base)
    };

    let resp = http
        .get(&url)
        .header("accept", CONTENT_TYPE_RKYV)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} from {}", resp.status(), url));
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    GlobalFederationTelemetry::from_http_frame(&bytes).map_err(|e| e.to_string())
}

/// Spawns a dedicated background `tokio` runtime thread that polls both the LAN Beowulf Cluster
/// Leader (over TLS 1.3 QUIC) and the Apache Web Sync Server (over HTTPS `reqwest`) every 1.25s.
pub fn spawn_network_telemetry_thread(
    initial: PollerTargets,
    shared: Arc<Mutex<NetworkTelemetryState>>,
) -> watch::Sender<PollerTargets> {
    let (tx, mut rx) = watch::channel(initial);

    std::thread::Builder::new()
        .name("beofold-gui-net-poller".into())
        .spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            else {
                return;
            };

            rt.block_on(async move {
                let Ok(ep) = build_gui_quic_endpoint() else {
                    return;
                };
                let gui_id = NodeIdentity::from_cluster_secret("gui-supervisor", "workstation");
                let gui_pk = gui_id.public_key_bytes();
                let http = Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build()
                    .unwrap_or_default();

                let mut cached_conn: Option<(SocketAddr, Connection)> = None;
                let mut ticker = tokio::time::interval(Duration::from_millis(1250));

                loop {
                    tokio::select! {
                        changed = rx.changed() => {
                            if changed.is_err() {
                                break;
                            }
                            cached_conn = None;
                        }
                        _ = ticker.tick() => {}
                    }

                    let targets = rx.borrow().clone();

                    if let Some(peer_sock) = targets.lan_peer {
                        match query_lan_peer_once(
                            &ep,
                            &mut cached_conn,
                            peer_sock,
                            &targets.cluster_secret,
                            gui_pk,
                        )
                        .await
                        {
                            Ok(snap) => {
                                if let Ok(mut guard) = shared.lock() {
                                    guard.lan_connected = true;
                                    guard.lan_last_error = None;
                                    guard.lan_snapshot = Some(snap);
                                }
                            }
                            Err(err) => {
                                if let Ok(mut guard) = shared.lock() {
                                    guard.lan_connected = false;
                                    guard.lan_last_error = Some(err);
                                }
                            }
                        }
                    }

                    if let Some(ref url) = targets.wan_sync_url {
                        if !url.trim().is_empty() {
                            match query_wan_telemetry_once(&http, url).await {
                                Ok(wsnap) => {
                                    if let Ok(mut guard) = shared.lock() {
                                        guard.wan_connected = true;
                                        guard.wan_last_error = None;
                                        guard.wan_snapshot = Some(wsnap);
                                    }
                                }
                                Err(err) => {
                                    if let Ok(mut guard) = shared.lock() {
                                        guard.wan_connected = false;
                                        guard.wan_last_error = Some(err);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        })
        .ok();

    tx
}

/// Parses a resolved `_beofold._udp.local.` TXT record & IPv4 addresses into `DiscoveredLanPeer` entries.
pub fn parse_mdns_peer_record(
    instance: &str,
    addrs_v4: &[Ipv4Addr],
    port: u16,
    txt_props: &[(&str, &str)],
    expected_cluster_tag: &str,
) -> Vec<DiscoveredLanPeer> {
    if port == 0 {
        return Vec::new();
    }
    let find_prop = |key: &str| -> String {
        txt_props
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| (*v).to_string())
            .unwrap_or_default()
    };
    let cluster_tag = find_prop("cluster");
    let node_short = find_prop("node");
    let tier = find_prop("tier");
    let os = find_prop("os");
    let cores = find_prop("cores");
    let matches_secret = !cluster_tag.is_empty() && cluster_tag == expected_cluster_tag;

    addrs_v4
        .iter()
        .filter(|ip| !ip.is_unspecified())
        .map(|ip| DiscoveredLanPeer {
            addr: SocketAddr::new(IpAddr::V4(*ip), port),
            instance: instance.to_string(),
            cluster_tag: cluster_tag.clone(),
            node_short: node_short.clone(),
            tier: if tier.is_empty() { "?".into() } else { tier.clone() },
            os: if os.is_empty() { "unknown".into() } else { os.clone() },
            cores: if cores.is_empty() { "?".into() } else { cores.clone() },
            matches_secret,
        })
        .collect()
}

/// Performs a bounded non-blocking background mDNS browse (`_beofold._udp.local.`) paired with a
/// fast localhost QUIC probe (`127.0.0.1:7443..=7446`) so 1-click discovery finds local and LAN
/// Beowulf nodes without blocking the `egui` render loop or hanging offline.
pub fn spawn_one_click_mdns_scan(
    cluster_secret: String,
    wan_sync_url: String,
    shared: Arc<Mutex<NetworkTelemetryState>>,
    targets_tx: watch::Sender<PollerTargets>,
    auto_connect_if_idle: bool,
) {
    if let Ok(mut guard) = shared.lock() {
        if guard.lan_discovery_scanning {
            return;
        }
        guard.lan_discovery_scanning = true;
        guard.lan_discovery_status = Some("Scanning _beofold._udp.local. (mDNS) + localhost:7443..7446...".into());
    }

    std::thread::Builder::new()
        .name("beofold-gui-mdns-scan".into())
        .spawn(move || {
            let expected_tag = derive_cluster_id_tag(&cluster_secret);
            let mut peers: Vec<DiscoveredLanPeer> = Vec::new();
            let mut seen_addrs: HashSet<SocketAddr> = HashSet::new();
            let mut mdns_note: Option<String> = None;

            match ServiceDaemon::new() {
                Ok(mdns) => {
                    match mdns.browse(MDNS_SERVICE_TYPE) {
                        Ok(rx) => {
                            let deadline = Instant::now() + Duration::from_millis(1400);
                            while Instant::now() < deadline {
                                let rem = deadline.saturating_duration_since(Instant::now());
                                match rx.recv_timeout(rem.min(Duration::from_millis(150))) {
                                    Ok(ServiceEvent::ServiceResolved(resolved)) => {
                                        let v4s: Vec<Ipv4Addr> =
                                            resolved.get_addresses_v4().into_iter().collect();
                                        let cluster_v = resolved
                                            .get_property_val_str("cluster")
                                            .unwrap_or("");
                                        let node_v = resolved
                                            .get_property_val_str("node")
                                            .unwrap_or("");
                                        let tier_v = resolved
                                            .get_property_val_str("tier")
                                            .unwrap_or("");
                                        let os_v = resolved
                                            .get_property_val_str("os")
                                            .unwrap_or("");
                                        let cores_v = resolved
                                            .get_property_val_str("cores")
                                            .unwrap_or("");
                                        let props = [
                                            ("cluster", cluster_v),
                                            ("node", node_v),
                                            ("tier", tier_v),
                                            ("os", os_v),
                                            ("cores", cores_v),
                                        ];
                                        for p in parse_mdns_peer_record(
                                            resolved.get_fullname(),
                                            &v4s,
                                            resolved.get_port(),
                                            &props,
                                            &expected_tag,
                                        ) {
                                            if seen_addrs.insert(p.addr) {
                                                peers.push(p);
                                            }
                                        }
                                    }
                                    Ok(_) => {}
                                    Err(_) => break,
                                }
                            }
                        }
                        Err(err) => {
                            mdns_note = Some(format!("mDNS browse unavailable: {err}"));
                        }
                    }
                    let _ = mdns.shutdown();
                }
                Err(err) => {
                    mdns_note = Some(format!("mDNS daemon unavailable: {err}"));
                }
            }

            // Fast unicast QUIC fallback for localhost ports 7443..=7446 (Windows Firewall resilience)
            if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                rt.block_on(async {
                    if let Ok(ep) = build_gui_quic_endpoint() {
                        let gui_id =
                            NodeIdentity::from_cluster_secret("gui-supervisor", "discovery");
                        let gui_pk = gui_id.public_key_bytes();
                        let mac_key = derive_cluster_mac_key(&cluster_secret);
                        for port in 7443_u16..=7446_u16 {
                            let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
                            if seen_addrs.contains(&addr) {
                                continue;
                            }
                            let mut cached = None;
                            if let Ok(Ok(snap)) = tokio::time::timeout(
                                Duration::from_millis(260),
                                rpc_query_state(&ep, &mut cached, addr, &mac_key, gui_pk),
                            )
                            .await
                            {
                                let (tier_s, os_s, cores_s) = snap
                                    .peers
                                    .first()
                                    .map(|p| {
                                        (
                                            format!("{}", p.capability.tier),
                                            p.capability.os.clone(),
                                            p.capability.logical_cores.to_string(),
                                        )
                                    })
                                    .unwrap_or_else(|| ("?".into(), "local".into(), "?".into()));
                                if seen_addrs.insert(addr) {
                                    peers.push(DiscoveredLanPeer {
                                        addr,
                                        instance: format!("localhost:{port}"),
                                        cluster_tag: expected_tag.clone(),
                                        node_short: "local".into(),
                                        tier: tier_s,
                                        os: os_s,
                                        cores: cores_s,
                                        matches_secret: true,
                                    });
                                }
                            }
                        }
                    }
                });
            }

            // Sort matching-secret peers first, then by SocketAddr
            peers.sort_by(|a, b| {
                b.matches_secret
                    .cmp(&a.matches_secret)
                    .then_with(|| a.addr.to_string().cmp(&b.addr.to_string()))
            });

            let best_match = peers.iter().find(|p| p.matches_secret).map(|p| p.addr);
            let status_str = if peers.is_empty() {
                mdns_note.unwrap_or_else(|| {
                    "No BeoFold peers discovered on _beofold._udp.local. or localhost:7443..7446."
                        .into()
                })
            } else {
                let matching = peers.iter().filter(|p| p.matches_secret).count();
                format!(
                    "Discovered {} LAN peer(s) ({} matching cluster secret).",
                    peers.len(),
                    matching
                )
            };

            let mut should_send_target: Option<SocketAddr> = None;
            if let Ok(mut guard) = shared.lock() {
                guard.lan_discovery_scanning = false;
                guard.lan_discovery_status = Some(status_str);
                guard.discovered_lan_peers = peers;
                if auto_connect_if_idle && !guard.lan_connected {
                    if let Some(addr) = best_match {
                        guard.lan_target = addr.to_string();
                        should_send_target = Some(addr);
                    }
                }
            }

            if let Some(addr) = should_send_target {
                let wan_opt = if wan_sync_url.trim().is_empty() {
                    None
                } else {
                    Some(wan_sync_url.trim().to_string())
                };
                let _ = targets_tx.send(PollerTargets {
                    lan_peer: Some(addr),
                    cluster_secret,
                    wan_sync_url: wan_opt,
                });
            }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdns_peer_record_parsing_filters_cluster_secret_and_handles_offline_empty() {
        let tag = derive_cluster_id_tag("beofold-home-cluster");
        let addrs = vec![Ipv4Addr::new(192, 168, 1, 42), Ipv4Addr::UNSPECIFIED];
        let props = [
            ("cluster", tag.as_str()),
            ("node", "a1b2c3d4"),
            ("tier", "S"),
            ("os", "windows"),
            ("cores", "24"),
        ];
        let peers = parse_mdns_peer_record("beofold-node._beofold._udp.local.", &addrs, 7443, &props, &tag);
        assert_eq!(peers.len(), 1);
        assert_eq!(
            peers[0].addr,
            "192.168.1.42:7443".parse::<SocketAddr>().unwrap()
        );
        assert!(peers[0].matches_secret);
        assert_eq!(peers[0].tier, "S");
        assert_eq!(peers[0].cores, "24");

        let mismatched = parse_mdns_peer_record(
            "other-node._beofold._udp.local.",
            &[Ipv4Addr::new(192, 168, 1, 99)],
            7443,
            &props,
            "different-tag",
        );
        assert_eq!(mismatched.len(), 1);
        assert!(!mismatched[0].matches_secret);

        let empty_port = parse_mdns_peer_record("bad", &addrs, 0, &props, &tag);
        assert!(empty_port.is_empty());
    }
}
