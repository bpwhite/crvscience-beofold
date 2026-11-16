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

use fold_proto::{AuthenticatedLanFrame, LanRpcRequest, LanRpcResponse, ProtoError};
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{ClientConfig, Connection, Endpoint, IdleTimeout, ServerConfig, TransportConfig, VarInt};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{
    ring::default_provider, verify_tls12_signature, verify_tls13_signature, CryptoProvider,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;

/// Maximum wire RPC frame size (`64 MiB`) to accommodate multi-chain protein trajectory blobs.
pub const MAX_LAN_RPC_BYTES: usize = 64 * 1024 * 1024;
const ALPN_BEOFOLD_LAN: &[u8] = b"beofold-lan-v2";

#[derive(Error, Debug)]
pub enum QuicNetError {
    #[error("TLS/certificate setup error: {0}")]
    Tls(String),
    #[error("QUIC endpoint bind error: {0}")]
    Bind(String),
    #[error("QUIC connect error to {addr}: {reason}")]
    Connect { addr: SocketAddr, reason: String },
    #[error("QUIC stream I/O error: {0}")]
    Stream(String),
    #[error("RPC timed out after {0:?}")]
    Timeout(Duration),
    #[error("protocol authentication/serialization error: {0}")]
    Proto(#[from] ProtoError),
}

/// Custom TLS 1.3 verifier for zero-config home Beowulf peer certificates.
/// Transport confidentiality & handshake integrity are enforced by TLS 1.3 (`ring`), while
/// peer authorization is enforced on every frame via `AuthenticatedLanFrame` (`BLAKE3` keyed MAC
/// derived from `--cluster-secret`) plus `Ed25519` work-unit signatures.
#[derive(Debug)]
struct BeowulfLanCertVerifier(Arc<CryptoProvider>);

impl BeowulfLanCertVerifier {
    fn new(provider: Arc<CryptoProvider>) -> Arc<Self> {
        Arc::new(Self(provider))
    }
}

impl ServerCertVerifier for BeowulfLanCertVerifier {
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

fn build_transport_config() -> Result<Arc<TransportConfig>, QuicNetError> {
    let mut transport = TransportConfig::default();
    let idle = IdleTimeout::try_from(Duration::from_secs(30))
        .map_err(|e| QuicNetError::Tls(format!("invalid idle timeout: {}", e)))?;
    transport.max_idle_timeout(Some(idle));
    transport.keep_alive_interval(Some(Duration::from_secs(8)));
    transport.stream_receive_window(VarInt::from_u32(64 * 1024 * 1024));
    transport.receive_window(VarInt::from_u32(128 * 1024 * 1024));
    Ok(Arc::new(transport))
}

/// Binds a dual-role (server + client) pure-Rust `quinn` QUIC endpoint on `bind_addr`.
pub fn bind_beowulf_endpoint(bind_addr: SocketAddr) -> Result<Endpoint, QuicNetError> {
    let provider = Arc::new(default_provider());
    let _ = default_provider().install_default();

    let cert = rcgen::generate_simple_self_signed(vec!["beofold.local".to_string()])
        .map_err(|e| QuicNetError::Tls(e.to_string()))?;
    let key_der = PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
    let cert_der: CertificateDer<'static> = cert.cert.into();

    let mut server_crypto = rustls::ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| QuicNetError::Tls(e.to_string()))?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], PrivateKeyDer::Pkcs8(key_der))
        .map_err(|e| QuicNetError::Tls(e.to_string()))?;
    server_crypto.alpn_protocols = vec![ALPN_BEOFOLD_LAN.to_vec()];

    let quic_server_crypto = QuicServerConfig::try_from(server_crypto)
        .map_err(|e| QuicNetError::Tls(e.to_string()))?;
    let mut server_config = ServerConfig::with_crypto(Arc::new(quic_server_crypto));
    let transport = build_transport_config()?;
    server_config.transport_config(transport.clone());

    let mut client_crypto = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| QuicNetError::Tls(e.to_string()))?
        .dangerous()
        .with_custom_certificate_verifier(BeowulfLanCertVerifier::new(provider))
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![ALPN_BEOFOLD_LAN.to_vec()];

    let quic_client_crypto = QuicClientConfig::try_from(client_crypto)
        .map_err(|e| QuicNetError::Tls(e.to_string()))?;
    let mut client_config = ClientConfig::new(Arc::new(quic_client_crypto));
    client_config.transport_config(transport);

    let mut endpoint = Endpoint::server(server_config, bind_addr)
        .map_err(|e| QuicNetError::Bind(format!("{}: {}", bind_addr, e)))?;
    endpoint.set_default_client_config(client_config);
    Ok(endpoint)
}

/// Thread-safe QUIC connection cache avoiding redundant TLS 1.3 handshakes between LAN peers.
#[derive(Clone)]
pub struct LanTransport {
    endpoint: Endpoint,
    cluster_mac_key: [u8; 32],
    node_pubkey: [u8; 32],
    pool: Arc<RwLock<HashMap<SocketAddr, Connection>>>,
}

impl LanTransport {
    pub fn new(
        endpoint: Endpoint,
        cluster_mac_key: [u8; 32],
        node_pubkey: [u8; 32],
    ) -> Self {
        Self {
            endpoint,
            cluster_mac_key,
            node_pubkey,
            pool: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    pub fn local_addr(&self) -> Result<SocketAddr, QuicNetError> {
        self.endpoint
            .local_addr()
            .map_err(|e| QuicNetError::Bind(e.to_string()))
    }

    pub fn cluster_mac_key(&self) -> &[u8; 32] {
        &self.cluster_mac_key
    }

    pub fn node_pubkey(&self) -> [u8; 32] {
        self.node_pubkey
    }

    async fn get_or_connect(
        &self,
        addr: SocketAddr,
        connect_timeout: Duration,
    ) -> Result<Connection, QuicNetError> {
        {
            let guard = self.pool.read().await;
            if let Some(existing) = guard.get(&addr) {
                if existing.close_reason().is_none() {
                    return Ok(existing.clone());
                }
            }
        }

        let connecting = self
            .endpoint
            .connect(addr, "beofold.local")
            .map_err(|e| QuicNetError::Connect {
                addr,
                reason: e.to_string(),
            })?;

        let conn = tokio::time::timeout(connect_timeout, connecting)
            .await
            .map_err(|_| QuicNetError::Timeout(connect_timeout))?
            .map_err(|e| QuicNetError::Connect {
                addr,
                reason: e.to_string(),
            })?;

        let mut guard = self.pool.write().await;
        guard.insert(addr, conn.clone());
        Ok(conn)
    }

    /// Sends an authenticated `LanRpcRequest` over a multiplexed QUIC bidirectional stream.
    pub async fn rpc(
        &self,
        target: SocketAddr,
        req: &LanRpcRequest,
        timeout: Duration,
    ) -> Result<([u8; 32], LanRpcResponse), QuicNetError> {
        let wire_req =
            AuthenticatedLanFrame::seal_request(&self.cluster_mac_key, self.node_pubkey, req)?;

        let fut = async {
            let conn = self
                .get_or_connect(target, timeout.min(Duration::from_secs(3)))
                .await?;
            let (mut send, mut recv) = match conn.open_bi().await {
                Ok(pair) => pair,
                Err(_) => {
                    // Evict stale connection and retry once cleanly
                    self.pool.write().await.remove(&target);
                    let fresh = self
                        .get_or_connect(target, timeout.min(Duration::from_secs(3)))
                        .await?;
                    fresh
                        .open_bi()
                        .await
                        .map_err(|e| QuicNetError::Stream(e.to_string()))?
                }
            };

            send.write_all(&wire_req)
                .await
                .map_err(|e| QuicNetError::Stream(e.to_string()))?;
            send.finish()
                .map_err(|e| QuicNetError::Stream(e.to_string()))?;

            let wire_resp = recv
                .read_to_end(MAX_LAN_RPC_BYTES)
                .await
                .map_err(|e| QuicNetError::Stream(e.to_string()))?;

            let opened = AuthenticatedLanFrame::open_response(&self.cluster_mac_key, &wire_resp)?;
            Ok(opened)
        };

        tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_| QuicNetError::Timeout(timeout))?
    }
}
