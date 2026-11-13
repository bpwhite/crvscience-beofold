use crate::prcg::WorkUnitId;
use crate::wire::{ProtoError, SignedWorkUnitResult};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use std::fs;
use std::path::Path;

const RESULT_DOMAIN_TAG: &[u8] = b"beofold-v1-wu-result";
const HTTP_DOMAIN_TAG: &[u8] = b"beofold-v3-http-wan-auth";

/// Derives a 32-byte symmetric BLAKE3 MAC key from a human-readable `--cluster-secret`.
pub fn derive_cluster_mac_key(cluster_secret: &str) -> [u8; 32] {
    blake3::derive_key("beofold-v2-lan-cluster-mac-key", cluster_secret.as_bytes())
}

/// Derives an 8-hex-character public cluster discriminator advertised in mDNS TXT records.
pub fn derive_cluster_id_tag(cluster_secret: &str) -> String {
    let raw = blake3::derive_key("beofold-v2-mdns-cluster-tag", cluster_secret.as_bytes());
    bytes_to_hex(&raw[0..4])
}

pub fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

pub fn hex_to_bytes_fixed<const N: usize>(hex_str: &str) -> Result<[u8; N], ProtoError> {
    let clean = hex_str.trim();
    if clean.len() != N * 2 {
        return Err(ProtoError::Signature(format!(
            "expected {}-char hex string, got {}",
            N * 2,
            clean.len()
        )));
    }
    let mut out = [0u8; N];
    for (i, slot) in out.iter_mut().enumerate() {
        let byte_str = &clean[(i * 2)..(i * 2 + 2)];
        *slot = u8::from_str_radix(byte_str, 16)
            .map_err(|e| ProtoError::Signature(format!("invalid hex '{}': {}", byte_str, e)))?;
    }
    Ok(out)
}

/// Canonicalizes an Apache reverse-proxy or CGI endpoint path (stripping leading `/beofold`
/// and `/api/v1` prefixes) so `Ed25519` HTTP request signatures remain valid regardless of
/// Apache `ProxyPass` or `ScriptAlias` mount points.
pub fn canonicalize_sync_route(raw_path: &str) -> String {
    let mut p = raw_path.split('?').next().unwrap_or("/").trim();
    if let Some(rest) = p.strip_prefix("/beofold") {
        p = if rest.is_empty() { "/" } else { rest };
    }
    if let Some(rest) = p.strip_prefix("/api/v1") {
        p = if rest.is_empty() { "/" } else { rest };
    }
    if !p.starts_with('/') {
        format!("/{}", p)
    } else if p.len() > 1 && p.ends_with('/') {
        p.trim_end_matches('/').to_string()
    } else {
        p.to_string()
    }
}

/// Computes the canonical 32-byte BLAKE3 digest signed in `X-BeoFold-Signature` HTTP headers.
pub fn compute_http_request_digest(
    method: &str,
    raw_or_canonical_path: &str,
    timestamp_unix_sec: u64,
    body_bytes: &[u8],
) -> [u8; 32] {
    let route = canonicalize_sync_route(raw_or_canonical_path);
    let body_hash = blake3::hash(body_bytes);
    let mut hasher = blake3::Hasher::new();
    hasher.update(HTTP_DOMAIN_TAG);
    hasher.update(method.trim().to_ascii_uppercase().as_bytes());
    hasher.update(b":");
    hasher.update(route.as_bytes());
    hasher.update(b":");
    hasher.update(&timestamp_unix_sec.to_le_bytes());
    hasher.update(body_hash.as_bytes());
    *hasher.finalize().as_bytes()
}

/// Verifies `X-BeoFold-Pubkey`, `X-BeoFold-Timestamp`, and `X-BeoFold-Signature` on an inbound
/// Apache HTTP / CGI / `axum` request within `max_skew_sec` (`300s` replay protection).
pub fn verify_http_request_headers(
    method: &str,
    path: &str,
    pubkey_hex: &str,
    timestamp_str: &str,
    signature_hex: &str,
    body_bytes: &[u8],
    now_unix_sec: u64,
    max_skew_sec: u64,
) -> Result<[u8; 32], ProtoError> {
    let pubkey_bytes = hex_to_bytes_fixed::<32>(pubkey_hex)?;
    let sig_bytes = hex_to_bytes_fixed::<64>(signature_hex)?;
    let ts = timestamp_str
        .trim()
        .parse::<u64>()
        .map_err(|e| ProtoError::Signature(format!("invalid timestamp header: {}", e)))?;

    if now_unix_sec.abs_diff(ts) > max_skew_sec {
        return Err(ProtoError::Signature(format!(
            "HTTP signature timestamp skew exceeded ({} vs server {})",
            ts, now_unix_sec
        )));
    }

    let vk = VerifyingKey::from_bytes(&pubkey_bytes)
        .map_err(|e| ProtoError::Signature(format!("invalid Ed25519 pubkey: {}", e)))?;
    let sig = Signature::from_bytes(&sig_bytes);
    let digest = compute_http_request_digest(method, path, ts, body_bytes);
    vk.verify(&digest, &sig)
        .map_err(|e| ProtoError::Signature(format!("HTTP Ed25519 signature invalid: {}", e)))?;

    Ok(pubkey_bytes)
}

/// Pure-Rust Ed25519 node or cluster identity used to sign completed work units and HTTP leases.
#[derive(Clone)]
pub struct NodeIdentity {
    signing_key: SigningKey,
}

impl NodeIdentity {
    /// Generates a fresh cryptographic identity using OS entropy (`getrandom`).
    pub fn generate_random() -> Result<Self, ProtoError> {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed)
            .map_err(|e| ProtoError::Signature(format!("OS RNG failure: {}", e)))?;
        Ok(Self {
            signing_key: SigningKey::from_bytes(&seed),
        })
    }

    /// Derives a deterministic node or cluster identity from a secret phrase and tag via BLAKE3 KDF.
    pub fn from_cluster_secret(cluster_secret: &str, node_tag: &str) -> Self {
        let mut material = Vec::with_capacity(cluster_secret.len() + 1 + node_tag.len());
        material.extend_from_slice(cluster_secret.as_bytes());
        material.push(b':');
        material.extend_from_slice(node_tag.as_bytes());
        let seed = blake3::derive_key("beofold-v1-deterministic-node-key", &material);
        Self {
            signing_key: SigningKey::from_bytes(&seed),
        }
    }

    /// Loads a 32-byte secret key file from disk, or generates and saves a new one atomically.
    pub fn load_or_create_keyfile(path: &Path) -> Result<Self, ProtoError> {
        if path.exists() {
            let raw = fs::read(path)?;
            if raw.len() != 32 {
                return Err(ProtoError::Signature(format!(
                    "expected 32-byte Ed25519 seed in {}, found {} bytes",
                    path.display(),
                    raw.len()
                )));
            }
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&raw);
            Ok(Self {
                signing_key: SigningKey::from_bytes(&seed),
            })
        } else {
            let id = Self::generate_random()?;
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            fs::write(path, id.signing_key.to_bytes())?;
            Ok(id)
        }
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    pub fn public_key_hex(&self) -> String {
        bytes_to_hex(&self.public_key_bytes())
    }

    pub fn short_id_hex(&self) -> String {
        bytes_to_hex(&self.public_key_bytes()[0..6])
    }

    pub fn sign_digest(&self, digest: &[u8; 32]) -> [u8; 64] {
        let sig: Signature = self.signing_key.sign(digest);
        sig.to_bytes()
    }

    /// Produces `(pubkey_hex, timestamp_str, signature_hex)` headers for outbound Apache HTTPS requests.
    pub fn sign_http_request(
        &self,
        method: &str,
        path: &str,
        timestamp_unix_sec: u64,
        body_bytes: &[u8],
    ) -> (String, String, String) {
        let digest = compute_http_request_digest(method, path, timestamp_unix_sec, body_bytes);
        let sig = self.sign_digest(&digest);
        (
            self.public_key_hex(),
            timestamp_unix_sec.to_string(),
            bytes_to_hex(&sig),
        )
    }

    /// Signs a `SignedWorkUnitResult` in-place over all canonical execution fields.
    pub fn sign_result(&self, result: &mut SignedWorkUnitResult) {
        result.worker_pubkey = self.public_key_bytes();
        let digest = compute_result_signing_digest(
            result.wu_id,
            &result.lease_token,
            &result.worker_pubkey,
            &result.input_checkpoint_blake3,
            &result.output_checkpoint_blake3,
            result.steps_executed,
            result.temperature_kelvin,
            result.final_potential_kcal,
        );
        result.signature = self.sign_digest(&digest);
    }
}

/// Computes the canonical domain-separated BLAKE3 digest signed by worker nodes.
#[allow(clippy::too_many_arguments)]
pub fn compute_result_signing_digest(
    wu_id: WorkUnitId,
    lease_token: &[u8; 16],
    worker_pubkey: &[u8; 32],
    input_checkpoint_blake3: &[u8; 32],
    output_checkpoint_blake3: &[u8; 32],
    steps_executed: u64,
    temperature_kelvin: f64,
    final_potential_kcal: f64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RESULT_DOMAIN_TAG);
    hasher.update(&wu_id.to_key_bytes());
    hasher.update(lease_token);
    hasher.update(worker_pubkey);
    hasher.update(input_checkpoint_blake3);
    hasher.update(output_checkpoint_blake3);
    hasher.update(&steps_executed.to_le_bytes());
    hasher.update(&temperature_kelvin.to_bits().to_le_bytes());
    hasher.update(&final_potential_kcal.to_bits().to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Verifies the Ed25519 provenance signature on a completed work-unit receipt.
pub fn verify_signed_result(result: &SignedWorkUnitResult) -> Result<(), ProtoError> {
    let vk = VerifyingKey::from_bytes(&result.worker_pubkey)
        .map_err(|e| ProtoError::Signature(format!("invalid public key: {}", e)))?;
    let sig = Signature::from_bytes(&result.signature);
    let digest = compute_result_signing_digest(
        result.wu_id,
        &result.lease_token,
        &result.worker_pubkey,
        &result.input_checkpoint_blake3,
        &result.output_checkpoint_blake3,
        result.steps_executed,
        result.temperature_kelvin,
        result.final_potential_kcal,
    );
    vk.verify(&digest, &sig)
        .map_err(|e| ProtoError::Signature(format!("signature mismatch: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_request_signature_survives_apache_prefix_rewrite() {
        let cluster_id = NodeIdentity::from_cluster_secret("oregon-cluster", "leader-key");
        let body = b"sample-binary-rkyv-payload";
        let (pk_hex, ts_str, sig_hex) = cluster_id.sign_http_request(
            "PUT",
            "/beofold/api/v1/workunits/1/0/2/3",
            1_762_980_000,
            body,
        );

        // Verify when Apache mod_proxy strips `/beofold` or `mod_cgid` passes `/workunits/1/0/2/3`
        let verified_pk = verify_http_request_headers(
            "PUT",
            "/workunits/1/0/2/3",
            &pk_hex,
            &ts_str,
            &sig_hex,
            body,
            1_762_980_015,
            300,
        )
        .expect("valid signature across Apache rewrite");
        assert_eq!(verified_pk, cluster_id.public_key_bytes());
    }
}
