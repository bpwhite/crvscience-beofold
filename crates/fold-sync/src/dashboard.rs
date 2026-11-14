use fold_physics::ProteinTopology;
use fold_proto::{ProjectManifest, TrajectoryCheckpoint, WorkUnitId};
use fold_storage::{to_hex, FoldStore};

/// Compile-time embedded read-only HTML5 + Vanilla JS + Canvas 3D Observatory UI.
/// Keeping the entire dashboard embedded inside the `fold-sync` binary preserves single-file
/// deployment under both Mode A (`axum` daemon) and Mode B (`/var/www/cgi-bin/fold-sync`).
pub const PUBLIC_DASHBOARD_HTML: &str = include_str!("dashboard.html");

/// Extracts an optional `?project_id=N` query parameter from a raw HTTP request URI.
pub fn parse_project_id_query(raw_path: &str) -> Option<u32> {
    let (_, query) = raw_path.split_once('?')?;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k.trim() == "project_id" {
                if let Ok(pid) = v.trim().parse::<u32>() {
                    return Some(pid);
                }
            }
        }
    }
    None
}

struct ResolvedBestSnapshot {
    source_stage: &'static str,
    manifest: ProjectManifest,
    topology: ProteinTopology,
    wu_id: WorkUnitId,
    checkpoint_blake3: [u8; 32],
    checkpoint: TrajectoryCheckpoint,
}

fn resolve_best_snapshot(
    store: &FoldStore,
    requested_project: Option<u32>,
) -> Result<Option<ResolvedBestSnapshot>, String> {
    let stats = store
        .collect_stats(requested_project)
        .map_err(|e| e.to_string())?;

    if let Some((best_wu, _best_e, best_hash)) = stats.lowest_potential_kcal {
        let (manifest, topology) = store
            .load_project(best_wu.project)
            .map_err(|e| e.to_string())?;
        let checkpoint = store
            .get_checkpoint(&best_hash)
            .map_err(|e| e.to_string())?;
        return Ok(Some(ResolvedBestSnapshot {
            source_stage: "completed_lowest_energy",
            manifest,
            topology,
            wu_id: best_wu,
            checkpoint_blake3: best_hash,
            checkpoint,
        }));
    }

    // Fallback when campaign is newly initialized (`0` completed units yet): serve the first
    // seeded `Gen 0` conformation so visitors immediately see the starting 3D polypeptide!
    let target_pid = match requested_project {
        Some(pid) => Some(pid),
        None => store
            .list_project_ids()
            .map_err(|e| e.to_string())?
            .into_iter()
            .next(),
    };
    let Some(pid) = target_pid else {
        return Ok(None);
    };

    let (manifest, topology) = store.load_project(pid).map_err(|e| e.to_string())?;
    let mut records = store.list_workunits(pid).map_err(|e| e.to_string())?;
    records.sort_by_key(|r| r.wu_id.to_key_bytes());

    let Some(first_rec) = records.first() else {
        return Ok(None);
    };
    let checkpoint = store
        .get_checkpoint(&first_rec.input_checkpoint_blake3)
        .map_err(|e| e.to_string())?;

    Ok(Some(ResolvedBestSnapshot {
        source_stage: "seeded_gen0_conformation",
        manifest,
        topology,
        wu_id: first_rec.wu_id,
        checkpoint_blake3: first_rec.input_checkpoint_blake3,
        checkpoint,
    }))
}

/// Builds the JSON payload for `GET /api/v1/public/conformation/best` containing full 3D heavy-atom
/// coordinates, covalent connectivity, energetics, and BLAKE3 provenance digest.
pub fn build_best_conformation_json(
    store: &FoldStore,
    requested_project: Option<u32>,
) -> Result<String, String> {
    let Some(snap) = resolve_best_snapshot(store, requested_project)? else {
        return Ok("{\"available\":false}\n".to_string());
    };

    let n = snap
        .topology
        .atoms
        .len()
        .min(snap.checkpoint.positions_angstrom.len());
    let mut atom_entries = Vec::with_capacity(n);
    for idx in 0..n {
        let a = &snap.topology.atoms[idx];
        let pos = snap.checkpoint.positions_angstrom[idx];
        atom_entries.push(format!(
            "{{\"i\":{},\"name\":\"{}\",\"res\":\"{}\",\"seq\":{},\"elem\":\"{}\",\"q\":{:.4},\"bb\":{},\"ca\":{},\"x\":{:.3},\"y\":{:.3},\"z\":{:.3}}}",
            idx,
            escape_json(&a.name),
            escape_json(&a.res_name),
            a.res_seq,
            escape_json(&a.element),
            a.charge_e,
            a.is_backbone,
            a.is_ca,
            pos[0],
            pos[1],
            pos[2]
        ));
    }

    let mut bond_entries = Vec::with_capacity(snap.topology.bonds.len());
    for b in &snap.topology.bonds {
        if (b.i as usize) < n && (b.j as usize) < n {
            bond_entries.push(format!("[{},{}]", b.i, b.j));
        }
    }

    let simulated_ps =
        (snap.checkpoint.cumulative_steps as f64) * snap.manifest.dt_fs * 1.0e-3;

    Ok(format!(
        concat!(
            "{{\n",
            "  \"available\": true,\n",
            "  \"source_stage\": \"{}\",\n",
            "  \"project_id\": {},\n",
            "  \"project_name\": \"{}\",\n",
            "  \"sequence\": \"{}\",\n",
            "  \"residue_count\": {},\n",
            "  \"atom_count\": {},\n",
            "  \"workunit\": \"{}\",\n",
            "  \"cumulative_steps\": {},\n",
            "  \"simulated_ps\": {:.4},\n",
            "  \"temperature_kelvin\": {:.2},\n",
            "  \"potential_energy_kcal\": {:.4},\n",
            "  \"kinetic_energy_kcal\": {:.4},\n",
            "  \"rg_angstrom\": {:.4},\n",
            "  \"sasa_angstrom2\": {:.4},\n",
            "  \"checkpoint_blake3\": \"{}\",\n",
            "  \"atoms\": [{}],\n",
            "  \"bonds\": [{}]\n",
            "}}\n"
        ),
        snap.source_stage,
        snap.manifest.project_id,
        escape_json(&snap.manifest.name),
        escape_json(&snap.manifest.sequence),
        snap.manifest.residue_count,
        n,
        snap.wu_id,
        snap.checkpoint.cumulative_steps,
        simulated_ps,
        snap.checkpoint.temperature_kelvin,
        snap.checkpoint.potential_energy_kcal,
        snap.checkpoint.kinetic_energy_kcal,
        snap.checkpoint.rg_angstrom,
        snap.checkpoint.sasa_angstrom2,
        to_hex(&snap.checkpoint_blake3),
        atom_entries.join(","),
        bond_entries.join(",")
    ))
}

/// Builds a downloadable crystallographic `.pdb` snapshot (`(filename, pdb_text)`) for
/// `GET /api/v1/public/conformation/best.pdb`.
pub fn build_best_conformation_pdb(
    store: &FoldStore,
    requested_project: Option<u32>,
) -> Result<Option<(String, String)>, String> {
    let Some(snap) = resolve_best_snapshot(store, requested_project)? else {
        return Ok(None);
    };

    let safe_name: String = snap
        .manifest
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let filename = format!(
        "beofold_P{}_{}_{}.pdb",
        snap.manifest.project_id, safe_name, snap.wu_id
    );
    let title = format!("{} {}", snap.source_stage, snap.wu_id);
    let pdb_text = snap
        .topology
        .format_pdb_snapshot(
            &snap.checkpoint.positions_angstrom,
            &title,
            snap.checkpoint.potential_energy_kcal,
        )
        .map_err(|e| e.to_string())?;

    Ok(Some((filename, pdb_text)))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fold_physics::{initialize_clone_checkpoint, minimize_steric_clashes};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDirGuard(PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn query_parser_and_embedded_dashboard_html() {
        assert_eq!(
            parse_project_id_query("/api/v1/public/conformation/best?project_id=42"),
            Some(42)
        );
        assert_eq!(
            parse_project_id_query("/api?foo=1&project_id=7&bar=2"),
            Some(7)
        );
        assert_eq!(parse_project_id_query("/api"), None);
        assert_eq!(parse_project_id_query("/api?project_id=bad"), None);

        assert!(!PUBLIC_DASHBOARD_HTML.is_empty());
        assert!(PUBLIC_DASHBOARD_HTML.contains("BeoFold"));
    }

    #[test]
    fn best_conformation_json_and_pdb_on_empty_and_seeded_store() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "beofold_dash_test_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        let _guard = TempDirGuard(dir.clone());

        let store = FoldStore::open(&dir).expect("open store");
        let empty_json = build_best_conformation_json(&store, None).expect("empty json");
        assert_eq!(empty_json, "{\"available\":false}\n");
        assert!(build_best_conformation_pdb(&store, None).expect("empty pdb").is_none());

        let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
        minimize_steric_clashes(&topo, &mut coords, 10);
        let wu = WorkUnitId::new(11, 0, 0, 0);
        let cp = initialize_clone_checkpoint(&topo, &coords, wu, 300.0, b"dash-test");
        let manifest = ProjectManifest {
            project_id: 11,
            name: "TrpCage Dashboard Test".into(),
            sequence: topo.sequence_one_letter.clone(),
            topology_blake3: [0u8; 32],
            initial_coords_blake3: [0u8; 32],
            atom_count: topo.atoms.len() as u32,
            residue_count: topo.residue_count,
            temp_ladder_kelvin: vec![300.0],
            dt_fs: 1.0,
            friction_per_ps: 1.0,
            steps_per_gen: 100,
            num_clones_per_run: 1,
            max_generations: 2,
        };
        store
            .seed_project(manifest, &topo, &[cp])
            .expect("seed project");

        let seeded_json = build_best_conformation_json(&store, Some(11)).expect("seeded json");
        assert!(seeded_json.contains("\"available\": true"));
        assert!(seeded_json.contains("\"source_stage\": \"seeded_gen0_conformation\""));

        let (filename, pdb_text) = build_best_conformation_pdb(&store, Some(11))
            .expect("seeded pdb")
            .expect("expected Some pdb");
        assert!(filename.ends_with(".pdb"));
        assert!(pdb_text.contains("ATOM"));
    }
}
