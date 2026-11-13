use crate::core::{RawHttpRequest, SyncCore};
use std::collections::HashMap;
use std::env;
use std::io::{self, Read, Write};

const MAX_CGI_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Runs Mode B: RFC-3875 Apache `mod_cgid` / `mod_cgi` single-request handler.
/// Drop the compiled binary into `/var/www/cgi-bin/fold-sync` with `BEOFOLD_DATA_DIR` set in Apache
/// (`SetEnv BEOFOLD_DATA_DIR /var/lib/beofold`) for zero-daemon shared Apache hosting!
pub fn run_cgi_once(core: &SyncCore) -> io::Result<()> {
    let method = env::var("REQUEST_METHOD").unwrap_or_else(|_| "GET".to_string());
    let mut path = env::var("PATH_INFO")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("REQUEST_URI").ok())
        .unwrap_or_else(|| "/".to_string());

    if !path.contains('?') {
        if let Ok(qs) = env::var("QUERY_STRING") {
            let trimmed = qs.trim();
            if !trimmed.is_empty() {
                path.push('?');
                path.push_str(trimmed);
            }
        }
    }

    let content_len = env::var("CONTENT_LENGTH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0)
        .min(MAX_CGI_BODY_BYTES);

    let mut body = Vec::with_capacity(content_len);
    if content_len > 0 {
        io::stdin()
            .take(content_len as u64)
            .read_to_end(&mut body)?;
    }

    let mut headers = HashMap::new();
    for (k, v) in env::vars() {
        if let Some(rest) = k.strip_prefix("HTTP_") {
            let header_name = rest.to_ascii_lowercase().replace('_', "-");
            headers.insert(header_name, v);
        } else if k == "CONTENT_TYPE" {
            headers.insert("content-type".to_string(), v);
        }
    }

    let resp = core.dispatch(RawHttpRequest {
        method,
        path,
        headers,
        body,
    });

    let reason = match resp.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Internal Server Error",
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    write!(out, "Status: {} {}\r\n", resp.status, reason)?;
    write!(out, "Content-Type: {}\r\n", resp.content_type)?;
    if let Some(cd) = &resp.content_disposition {
        write!(out, "Content-Disposition: {}\r\n", cd)?;
    }
    if let Some(cc) = &resp.cache_control {
        write!(out, "Cache-Control: {}\r\n", cc)?;
    }
    write!(out, "Content-Length: {}\r\n\r\n", resp.body.len())?;
    out.write_all(&resp.body)?;
    out.flush()?;
    Ok(())
}
