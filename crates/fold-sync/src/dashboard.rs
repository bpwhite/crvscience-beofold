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
