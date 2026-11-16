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

pub mod axum_server;
pub mod cgi;
pub mod core;
pub mod dashboard;

pub use axum_server::serve_axum;
pub use cgi::run_cgi_once;
pub use core::{RawHttpRequest, RawHttpResponse, SyncCore, SyncServerPolicy};
pub use dashboard::{
    build_best_conformation_json, build_best_conformation_pdb, PUBLIC_DASHBOARD_HTML,
};
