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
    derive_cluster_mac_key, AuthenticatedLanFrame, ClusterStateSnapshot, GlobalFederationTelemetry,
    LanRpcRequest, LanRpcResponse, NodeIdentity, CONTENT_TYPE_RKYV,
};
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
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

const MAX_GUI_RPC_BYTES: usize = 32 * 1024 * 1024;
const ALPN_BEOFOLD_LAN: &[u8] = b"beofold-lan-v2";

/// Live network snapshot shared lock-free / low-contention with the 60 FPS `egui` render thread.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct NetworkTelemetryState {
    pub lan_target: String,
    pub lan_secret: String,
    pub lan_connected: bool,
    pub lan_last_error: Option<String>,
    pub lan_snapshot: Option<ClusterStateSnapshot>,

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

    let need_reconnect = match cached_conn {
        Some((prev_addr, ref conn)) => *prev_addr != addr || conn.close_reason().is_some(),
        None => true,
    };

    if need_reconnect {
        let connecting = ep
            .connect(addr, "beofold.local")
            .map_err(|e| e.to_string())?;
        let conn = tokio::time::timeout(Duration::from_millis(1800), connecting)
            .await
            .map_err(|_| format!("QUIC connect timeout to {}", addr))?
            .map_err(|e| e.to_string())?;
        *cached_conn = Some((addr, conn));
    }

    let conn = cached_conn.as_ref().unwrap().1.clone();
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
