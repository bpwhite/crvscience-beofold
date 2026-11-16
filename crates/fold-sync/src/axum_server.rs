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

use crate::core::{RawHttpRequest, SyncCore};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Router;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::watch;

/// Maximum upload size (`64 MiB`) matching `LimitRequestBody` in the Apache `mod_proxy` config.
const MAX_HTTP_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Runs Mode A: Persistent async `axum` HTTP synchronizing server designed to run behind
/// Apache `mod_proxy_http` (`ProxyPass "/beofold/api/v1/" "http://127.0.0.1:8780/api/v1/"`).
pub async fn serve_axum(
    core: Arc<SyncCore>,
    bind_addr: SocketAddr,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<SocketAddr, std::io::Error> {
    let app = Router::new()
        .fallback(handle_axum_http)
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .with_state(core);

    let listener = TcpListener::bind(bind_addr).await?;
    let bound = listener.local_addr()?;

    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                while !*shutdown_rx.borrow() {
                    if shutdown_rx.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await;
    });

    Ok(bound)
}

async fn handle_axum_http(
    State(core): State<Arc<SyncCore>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut hdr_map = HashMap::with_capacity(headers.len());
    for (name, val) in &headers {
        if let Ok(v_str) = val.to_str() {
            hdr_map.insert(name.as_str().to_ascii_lowercase(), v_str.to_string());
        }
    }

    let full_path = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| uri.path().to_string());

    let raw_req = RawHttpRequest {
        method: method.as_str().to_string(),
        path: full_path,
        headers: hdr_map,
        body: body.to_vec(),
    };

    let core_ref = core.clone();
    let raw_resp = tokio::task::spawn_blocking(move || core_ref.dispatch(raw_req))
        .await
        .unwrap_or_else(|e| crate::core::RawHttpResponse {
            status: 500,
            content_type: "application/json".to_string(),
            content_disposition: None,
            cache_control: Some("no-store".to_string()),
            body: format!("{{\"error\":\"worker panic: {}\"}}\n", e).into_bytes(),
        });

    let status = StatusCode::from_u16(raw_resp.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut resp_headers = HeaderMap::new();
    if let Ok(ct) = HeaderValue::from_str(&raw_resp.content_type) {
        resp_headers.insert(header::CONTENT_TYPE, ct);
    }
    if let Some(cd) = &raw_resp.content_disposition {
        if let Ok(hv) = HeaderValue::from_str(cd) {
            resp_headers.insert(header::CONTENT_DISPOSITION, hv);
        }
    }
    if let Some(cc) = &raw_resp.cache_control {
        if let Ok(hv) = HeaderValue::from_str(cc) {
            resp_headers.insert(header::CACHE_CONTROL, hv);
        }
    }

    (status, resp_headers, raw_resp.body).into_response()
}
