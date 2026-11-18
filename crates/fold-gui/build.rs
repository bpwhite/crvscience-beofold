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

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/beofold.ico");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let ico_path = manifest_dir.join("assets").join("beofold.ico");
    if !ico_path.is_file() {
        println!(
            "cargo:warning=Missing icon asset at {}, skipping Windows resource embedding",
            ico_path.display()
        );
        return;
    }

    let ico_rc = ico_path.to_string_lossy().replace('\\', "/");
    let rc_content = format!(
        r#"1 ICON "{ico_rc}"

1 VERSIONINFO
FILEVERSION 1,4,0,0
PRODUCTVERSION 1,4,0,0
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
            VALUE "CompanyName", "CRV Science Network"
            VALUE "FileDescription", "BeoFold Distributed Protein Folding & MSM Supervisor"
            VALUE "FileVersion", "1.4.0"
            VALUE "InternalName", "fold-gui"
            VALUE "LegalCopyright", "Copyright CRV Science Network (GPL-3.0-or-later)"
            VALUE "OriginalFilename", "3_BeoFold.exe"
            VALUE "ProductName", "BeoFold"
            VALUE "ProductVersion", "1.4.0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END
"#
    );

    let rc_path = out_dir.join("beofold.rc");
    if let Err(err) = fs::write(&rc_path, rc_content) {
        println!(
            "cargo:warning=Failed to write Windows resource script {}: {}",
            rc_path.display(),
            err
        );
        return;
    }

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "gnu" {
        let obj_path = out_dir.join("beofold_res.o");
        for tool in ["windres", "x86_64-w64-mingw32-windres"] {
            if let Ok(status) = Command::new(tool)
                .arg(&rc_path)
                .arg("-O")
                .arg("coff")
                .arg("-o")
                .arg(&obj_path)
                .status()
            {
                if status.success() {
                    println!("cargo:rustc-link-arg={}", obj_path.display());
                    return;
                }
            }
        }
        println!("cargo:warning=windres not found on gnu target; skipping Windows resource embedding");
    } else if target_env == "msvc" {
        let res_path = out_dir.join("beofold_res.res");
        if let Some(rc_exe) = find_msvc_rc_exe() {
            if let Ok(status) = Command::new(rc_exe)
                .arg("/nologo")
                .arg("/fo")
                .arg(&res_path)
                .arg(&rc_path)
                .status()
            {
                if status.success() {
                    println!("cargo:rustc-link-arg={}", res_path.display());
                    return;
                }
            }
        }
        println!("cargo:warning=rc.exe not found on msvc target; skipping Windows resource embedding");
    }
}

fn find_msvc_rc_exe() -> Option<PathBuf> {
    if Command::new("rc.exe").arg("/?").output().is_ok() {
        return Some(PathBuf::from("rc.exe"));
    }
    let kits_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if !kits_root.is_dir() {
        return None;
    }
    let mut versions: Vec<PathBuf> = fs::read_dir(kits_root)
        .ok()?
        .filter_map(|e| e.ok().map(|entry| entry.path()))
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    versions.reverse();
    for ver in versions {
        for arch in ["x64", "x86"] {
            let candidate = ver.join(arch).join("rc.exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}
