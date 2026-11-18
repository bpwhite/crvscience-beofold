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

use fold_proto::WIRE_MAGIC;
use pdbtbx::Element;
use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PhysicsError {
    #[error("PDB/mmCIF parse error: {0}")]
    PdbParse(String),
    #[error("topology contains zero protein atoms")]
    EmptyTopology,
    #[error("coordinate count mismatch: topology expects {expected}, got {actual}")]
    AtomCountMismatch { expected: usize, actual: usize },
    #[error("serialization error: {0}")]
    Codec(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AtomRecord {
    pub serial: u32,
    pub name: String,
    pub res_name: String,
    pub res_seq: i32,
    pub chain_id: char,
    pub element: String,
    pub mass_amu: f32,
    pub charge_e: f32,
    pub lj_sigma_ang: f32,
    pub lj_epsilon_kcal: f32,
    pub gb_radius_ang: f32,
    pub is_backbone: bool,
    pub is_ca: bool,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct BondTerm {
    pub i: u32,
    pub j: u32,
    pub r0_ang: f32,
    pub kb_kcal_ang2: f32,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct AngleTerm {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub theta0_rad: f32,
    pub ktheta_kcal_rad2: f32,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct TorsionTerm {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub l: u32,
    pub periodicity: u8,
    pub phase_rad: f32,
    pub vn_kcal: f32,
}

/// Non-bonded interaction policy for covalently connected atom pairs (`AMBER ff14SB` convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusionPolicy {
    /// Full 1.0 Lennard-Jones and 1.0 electrostatics.
    Full,
    /// 1-4 dihedral terminal atoms scaled by 0.5 LJ and 1/1.2 (0.8333) electrostatics.
    Scaled14,
    /// Excluded 1-2 (bonded) or 1-3 (angle) pair.
    Excluded,
}

/// Complete, serializable molecular mechanics topology for a target protein.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProteinTopology {
    pub name: String,
    pub sequence_one_letter: String,
    pub residue_count: u32,
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<BondTerm>,
    pub angles: Vec<AngleTerm>,
    pub torsions: Vec<TorsionTerm>,
    /// Packed `(min_idx << 32) | max_idx` keys for excluded 1-2 and 1-3 pairs.
    pub excluded_12_13: Vec<u64>,
    /// Packed `(min_idx << 32) | max_idx` keys for scaled 1-4 pairs.
    pub scaled_14: Vec<u64>,
}

impl ProteinTopology {
    #[inline]
    pub fn pack_pair(a: usize, b: usize) -> u64 {
        let (lo, hi) = if a < b { (a as u64, b as u64) } else { (b as u64, a as u64) };
        (lo << 32) | hi
    }

    #[inline]
    pub fn pair_policy(&self, a: usize, b: usize) -> ExclusionPolicy {
        if a == b {
            return ExclusionPolicy::Excluded;
        }
        let key = Self::pack_pair(a, b);
        if self.excluded_12_13.binary_search(&key).is_ok() {
            ExclusionPolicy::Excluded
        } else if self.scaled_14.binary_search(&key).is_ok() {
            ExclusionPolicy::Scaled14
        } else {
            ExclusionPolicy::Full
        }
    }

    pub fn to_compressed_bytes(&self) -> Result<Vec<u8>, PhysicsError> {
        let aligned = rkyv::to_bytes::<RkyvError>(self)
            .map_err(|e| PhysicsError::Codec(e.to_string()))?;
        let compressed = lz4_flex::compress_prepend_size(aligned.as_slice());
        let mut out = Vec::with_capacity(WIRE_MAGIC.len() + compressed.len());
        out.extend_from_slice(&WIRE_MAGIC);
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    pub fn from_compressed_bytes(frame: &[u8]) -> Result<Self, PhysicsError> {
        if frame.len() < WIRE_MAGIC.len() || frame[..WIRE_MAGIC.len()] != WIRE_MAGIC {
            return Err(PhysicsError::Codec("missing BFLDv1 topology header".into()));
        }
        let raw = lz4_flex::decompress_size_prepended(&frame[WIRE_MAGIC.len()..])
            .map_err(|e| PhysicsError::Codec(e.to_string()))?;
        let mut aligned = AlignedVec::<16>::with_capacity(raw.len());
        aligned.extend_from_slice(&raw);
        rkyv::from_bytes::<Self, RkyvError>(&aligned)
            .map_err(|e| PhysicsError::Codec(e.to_string()))
    }

    /// Parses a crystallographic PDB or mmCIF file via `pdbtbx` and builds a bonded + non-bonded force field.
    pub fn from_pdb_file(path: &Path) -> Result<(Self, Vec<[f32; 3]>), PhysicsError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| PhysicsError::PdbParse("non-UTF8 path".into()))?;
        let (pdb, _warnings) = pdbtbx::open(path_str)
            .map_err(|errs| PhysicsError::PdbParse(format!("{:?}", errs)))?;

        let mut atoms = Vec::new();
        let mut positions = Vec::new();
        let mut seq_chars = Vec::new();
        let mut res_counter = 0u32;

        for chain in pdb.chains() {
            let chain_char = chain.id().chars().next().unwrap_or('A');
            for residue in chain.residues() {
                let res_name = residue.name().unwrap_or("UNK").to_uppercase();
                if !is_standard_amino_acid(&res_name) {
                    continue;
                }
                res_counter += 1;
                seq_chars.push(three_to_one_letter(&res_name));
                let res_seq = residue.serial_number() as i32;

                if let Some(conformer) = residue.conformers().next() {
                    for atom in conformer.atoms() {
                        let atom_name = atom.name().trim().to_uppercase();
                        let elem_str = match atom.element() {
                            Some(Element::H) => "H",
                            Some(Element::C) => "C",
                            Some(Element::N) => "N",
                            Some(Element::O) => "O",
                            Some(Element::S) => "S",
                            _ => infer_element_from_name(&atom_name),
                        };
                        let (mass, sigma, eps, gb_rad) = element_vdw_params(elem_str, &atom_name);
                        let charge = partial_charge_heuristic(&res_name, &atom_name);
                        let is_backbone = matches!(atom_name.as_str(), "N" | "CA" | "C" | "O" | "OXT");
                        let is_ca = atom_name == "CA";

                        atoms.push(AtomRecord {
                            serial: (atoms.len() + 1) as u32,
                            name: atom_name,
                            res_name: res_name.clone(),
                            res_seq,
                            chain_id: chain_char,
                            element: elem_str.to_string(),
                            mass_amu: mass,
                            charge_e: charge,
                            lj_sigma_ang: sigma,
                            lj_epsilon_kcal: eps,
                            gb_radius_ang: gb_rad,
                            is_backbone,
                            is_ca,
                        });
                        positions.push([atom.x() as f32, atom.y() as f32, atom.z() as f32]);
                    }
                }
            }
        }

        if atoms.is_empty() {
            return Err(PhysicsError::EmptyTopology);
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pdb_target")
            .to_string();
        let sequence_one_letter: String = seq_chars.into_iter().collect();
        let topo = Self::infer_connectivity(name, sequence_one_letter, res_counter, atoms, &positions);
        Ok((topo, positions))
    }

    /// Builds a built-in 20-residue Trp-cage (`1L2Y`: `NLYIQWLKDGGPSSGRPPPS`) topology in an
    /// extended spiral-coil seed conformation (`N`, `CA`, `C`, `O`, plus `CB` sidechain beads)
    /// so Phase 1 folds a real miniprotein out of the box without external downloads.
    pub fn build_trp_cage_seed() -> (Self, Vec<[f32; 3]>) {
        Self::build_polypeptide_from_sequence("TrpCage_1L2Y", "NLYIQWLKDGGPSSGRPPPS")
    }

    /// Builds the 35-residue Villin Headpiece (`1YRF`) topology used for the startup capability benchmark.
    pub fn build_villin_benchmark_seed() -> (Self, Vec<[f32; 3]>) {
        Self::build_polypeptide_from_sequence(
            "Villin_HP35",
            "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        )
    }

    /// Constructs a clean, clash-free polypeptide (`N`, `CA`, `C`, `O`, and residue-specific `CB` centroid)
    /// from any 1-letter amino acid sequence with realistic covalent geometry.
    pub fn build_polypeptide_from_sequence(
        name: &str,
        sequence: &str,
    ) -> (Self, Vec<[f32; 3]>) {
        let mut atoms = Vec::new();
        let mut positions = Vec::new();
        let mut bonds = Vec::new();
        let mut prev_c_idx: Option<u32> = None;

        // Gentle open coil pitch so starting state is unfolded (`R_g` large) and sterically valid
        for (res_idx, aa) in sequence.chars().enumerate() {
            let aa_up = aa.to_ascii_uppercase();
            let res_seq = (res_idx + 1) as i32;
            let res_name = one_to_three_letter(aa_up);
            let s = res_idx as f32;

            // Place backbone along a wide, gently curving open polypeptide chain (~3.80 A CA-CA step)
            let base_angle = s * 0.42;
            let radius = 6.5 + 0.18 * s;
            let ca_pos = [
                radius * base_angle.cos(),
                radius * base_angle.sin(),
                s * 2.85,
            ];
            let tangent = [-base_angle.sin(), base_angle.cos(), 0.55];
            let t_norm = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
            let dir = [tangent[0] / t_norm, tangent[1] / t_norm, tangent[2] / t_norm];
            let radial = [base_angle.cos(), base_angle.sin(), 0.0];

            let n_pos = [
                ca_pos[0] - 1.46 * dir[0] - 0.15 * radial[0],
                ca_pos[1] - 1.46 * dir[1] - 0.15 * radial[1],
                ca_pos[2] - 1.46 * dir[2],
            ];
            let c_pos = [
                ca_pos[0] + 1.52 * dir[0] - 0.12 * radial[0],
                ca_pos[1] + 1.52 * dir[1] - 0.12 * radial[1],
                ca_pos[2] + 1.52 * dir[2],
            ];
            let o_pos = [
                c_pos[0] - 0.55 * radial[0] + 0.35 * dir[1],
                c_pos[1] - 0.55 * radial[1] - 0.35 * dir[0],
                c_pos[2] + 1.05,
            ];

            let n_idx = atoms.len() as u32;
            atoms.push(AtomRecord {
                serial: n_idx + 1,
                name: "N".into(),
                res_name: res_name.into(),
                res_seq,
                chain_id: 'A',
                element: "N".into(),
                mass_amu: 14.007,
                charge_e: -0.4157,
                lj_sigma_ang: 3.25,
                lj_epsilon_kcal: 0.17,
                gb_radius_ang: 1.55,
                is_backbone: true,
                is_ca: false,
            });
            positions.push(n_pos);

            let ca_idx = atoms.len() as u32;
            atoms.push(AtomRecord {
                serial: ca_idx + 1,
                name: "CA".into(),
                res_name: res_name.into(),
                res_seq,
                chain_id: 'A',
                element: "C".into(),
                mass_amu: 12.011,
                charge_e: 0.0337,
                lj_sigma_ang: 3.40,
                lj_epsilon_kcal: 0.1094,
                gb_radius_ang: 1.70,
                is_backbone: true,
                is_ca: true,
            });
            positions.push(ca_pos);

            let c_idx = atoms.len() as u32;
            atoms.push(AtomRecord {
                serial: c_idx + 1,
                name: "C".into(),
                res_name: res_name.into(),
                res_seq,
                chain_id: 'A',
                element: "C".into(),
                mass_amu: 12.011,
                charge_e: 0.5973,
                lj_sigma_ang: 3.40,
                lj_epsilon_kcal: 0.086,
                gb_radius_ang: 1.70,
                is_backbone: true,
                is_ca: false,
            });
            positions.push(c_pos);

            let o_idx = atoms.len() as u32;
            atoms.push(AtomRecord {
                serial: o_idx + 1,
                name: "O".into(),
                res_name: res_name.into(),
                res_seq,
                chain_id: 'A',
                element: "O".into(),
                mass_amu: 15.999,
                charge_e: -0.5679,
                lj_sigma_ang: 2.96,
                lj_epsilon_kcal: 0.21,
                gb_radius_ang: 1.50,
                is_backbone: true,
                is_ca: false,
            });
            positions.push(o_pos);

            // Actual geometric bond distances of initial placement so Gen 0 starts near equilibrium
            bonds.push(BondTerm {
                i: n_idx,
                j: ca_idx,
                r0_ang: dist3(n_pos, ca_pos),
                kb_kcal_ang2: 320.0,
            });
            bonds.push(BondTerm {
                i: ca_idx,
                j: c_idx,
                r0_ang: dist3(ca_pos, c_pos),
                kb_kcal_ang2: 310.0,
            });
            bonds.push(BondTerm {
                i: c_idx,
                j: o_idx,
                r0_ang: dist3(c_pos, o_pos),
                kb_kcal_ang2: 520.0,
            });

            if let Some(pc) = prev_c_idx {
                let pc_pos = positions[pc as usize];
                bonds.push(BondTerm {
                    i: pc,
                    j: n_idx,
                    r0_ang: dist3(pc_pos, n_pos),
                    kb_kcal_ang2: 420.0,
                });
            }

            // Sidechain C-beta / united sidechain bead for all non-glycine residues
            if aa_up != 'G' {
                let (sc_mass, sc_charge, sc_sigma, sc_eps) = sidechain_bead_params(aa_up);
                let cb_pos = [
                    ca_pos[0] + 1.54 * radial[0],
                    ca_pos[1] + 1.54 * radial[1],
                    ca_pos[2] - 0.15,
                ];
                let cb_idx = atoms.len() as u32;
                atoms.push(AtomRecord {
                    serial: cb_idx + 1,
                    name: "CB".into(),
                    res_name: res_name.into(),
                    res_seq,
                    chain_id: 'A',
                    element: "C".into(),
                    mass_amu: sc_mass,
                    charge_e: sc_charge + 0.3526, // Neutralizes backbone dipole except charged residues
                    lj_sigma_ang: sc_sigma,
                    lj_epsilon_kcal: sc_eps,
                    gb_radius_ang: sc_sigma * 0.5,
                    is_backbone: false,
                    is_ca: false,
                });
                positions.push(cb_pos);
                bonds.push(BondTerm {
                    i: ca_idx,
                    j: cb_idx,
                    r0_ang: dist3(ca_pos, cb_pos),
                    kb_kcal_ang2: 280.0,
                });
            }

            prev_c_idx = Some(c_idx);
        }

        let topo = Self::build_angles_torsions_exclusions(
            name.to_string(),
            sequence.to_ascii_uppercase(),
            sequence.chars().count() as u32,
            atoms,
            bonds,
            &positions,
        );
        (topo, positions)
    }

    fn infer_connectivity(
        name: String,
        sequence_one_letter: String,
        residue_count: u32,
        atoms: Vec<AtomRecord>,
        positions: &[[f32; 3]],
    ) -> Self {
        let mut bonds = Vec::new();
        let n = atoms.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // Only check atoms in same or adjacent residue for covalent peptide/sidechain bonds
                if (atoms[i].res_seq - atoms[j].res_seq).abs() > 1
                    || atoms[i].chain_id != atoms[j].chain_id
                {
                    continue;
                }
                let d = dist3(positions[i], positions[j]);
                let max_cov = if atoms[i].element == "H" || atoms[j].element == "H" {
                    1.22
                } else if atoms[i].element == "S" || atoms[j].element == "S" {
                    2.10
                } else {
                    1.68
                };
                if d > 0.4 && d <= max_cov {
                    bonds.push(BondTerm {
                        i: i as u32,
                        j: j as u32,
                        r0_ang: d,
                        kb_kcal_ang2: 310.0,
                    });
                }
            }
        }
        Self::build_angles_torsions_exclusions(
            name,
            sequence_one_letter,
            residue_count,
            atoms,
            bonds,
            positions,
        )
    }

    fn build_angles_torsions_exclusions(
        name: String,
        sequence_one_letter: String,
        residue_count: u32,
        atoms: Vec<AtomRecord>,
        bonds: Vec<BondTerm>,
        positions: &[[f32; 3]],
    ) -> Self {
        let mut adj: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        let mut excl_set: HashSet<u64> = HashSet::new();

        for b in &bonds {
            adj.entry(b.i).or_default().push(b.j);
            adj.entry(b.j).or_default().push(b.i);
            excl_set.insert(Self::pack_pair(b.i as usize, b.j as usize));
        }

        let mut angles = Vec::new();
        for (&center, neighbors) in &adj {
            for a_idx in 0..neighbors.len() {
                for b_idx in (a_idx + 1)..neighbors.len() {
                    let i = neighbors[a_idx];
                    let k = neighbors[b_idx];
                    excl_set.insert(Self::pack_pair(i as usize, k as usize));
                    let theta0 = angle3(
                        positions[i as usize],
                        positions[center as usize],
                        positions[k as usize],
                    );
                    angles.push(AngleTerm {
                        i,
                        j: center,
                        k,
                        theta0_rad: theta0,
                        ktheta_kcal_rad2: 55.0,
                    });
                }
            }
        }

        let mut torsions = Vec::new();
        let mut scaled_set: HashSet<u64> = HashSet::new();
        for b in &bonds {
            let j = b.i;
            let k = b.j;
            let Some(j_nb) = adj.get(&j) else { continue };
            let Some(k_nb) = adj.get(&k) else { continue };
            for &i in j_nb {
                if i == k {
                    continue;
                }
                for &l in k_nb {
                    if l == j || l == i {
                        continue;
                    }
                    let pair14 = Self::pack_pair(i as usize, l as usize);
                    if !excl_set.contains(&pair14) {
                        scaled_set.insert(pair14);
                    }
                    // Peptide bond C-N gets strong planar omega barrier (periodicity 2, phase pi)
                    let is_omega = (atoms[j as usize].name == "C" && atoms[k as usize].name == "N")
                        || (atoms[j as usize].name == "N" && atoms[k as usize].name == "C"
                            && atoms[j as usize].res_seq != atoms[k as usize].res_seq);
                    if is_omega {
                        torsions.push(TorsionTerm {
                            i,
                            j,
                            k,
                            l,
                            periodicity: 2,
                            phase_rad: std::f32::consts::PI,
                            vn_kcal: 5.0,
                        });
                    } else {
                        torsions.push(TorsionTerm {
                            i,
                            j,
                            k,
                            l,
                            periodicity: 3,
                            phase_rad: 0.0,
                            vn_kcal: 0.65,
                        });
                    }
                }
            }
        }

        let mut excluded_12_13: Vec<u64> = excl_set.into_iter().collect();
        excluded_12_13.sort_unstable();
        let mut scaled_14: Vec<u64> = scaled_set.into_iter().collect();
        scaled_14.sort_unstable();

        Self {
            name,
            sequence_one_letter,
            residue_count,
            atoms,
            bonds,
            angles,
            torsions,
            excluded_12_13,
            scaled_14,
        }
    }

    /// Formats atomic coordinates as an in-memory crystallographic PDB string for HTTP downloads and disk export.
    pub fn format_pdb_snapshot(
        &self,
        positions: &[[f32; 3]],
        remark_title: &str,
        potential_energy_kcal: f64,
    ) -> Result<String, PhysicsError> {
        use std::fmt::Write as _;
        if positions.len() != self.atoms.len() {
            return Err(PhysicsError::AtomCountMismatch {
                expected: self.atoms.len(),
                actual: positions.len(),
            });
        }
        let mut out = String::with_capacity(self.atoms.len() * 84 + 256);
        let _ = writeln!(out, "HEADER    BEOFOLD PROTEIN STRUCTURE               01-JAN-26   BFLD");
        let _ = writeln!(out, "TITLE     {} ({})", self.name, remark_title);
        let _ = writeln!(
            out,
            "REMARK   1 POTENTIAL ENERGY (KCAL/MOL): {:.4}",
            potential_energy_kcal
        );
        let _ = writeln!(out, "REMARK   2 SEQUENCE: {}", self.sequence_one_letter);

        for (atom, pos) in self.atoms.iter().zip(positions.iter()) {
            let _ = writeln!(
                out,
                "ATOM  {:>5} {:<4} {:>3} {:1}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}",
                atom.serial % 100_000,
                atom.name,
                atom.res_name,
                atom.chain_id,
                atom.res_seq,
                pos[0],
                pos[1],
                pos[2],
                atom.element
            );
        }
        let _ = writeln!(out, "TER");
        let _ = writeln!(out, "END");
        Ok(out)
    }

    /// Exports atomic coordinates to a standard crystallographic PDB file readable by `pdbtbx`, PyMOL, or ChimeraX.
    pub fn write_pdb_snapshot(
        &self,
        positions: &[[f32; 3]],
        path: &Path,
        remark_title: &str,
        potential_energy_kcal: f64,
    ) -> Result<(), PhysicsError> {
        let pdb_text = self.format_pdb_snapshot(positions, remark_title, potential_energy_kcal)?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        w.write_all(pdb_text.as_bytes())?;
        w.flush()?;
        Ok(())
    }
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn angle3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-6);
    let vn = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    let cos_t = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (un * vn)).clamp(-0.999999, 0.999999);
    cos_t.acos()
}

fn is_standard_amino_acid(res: &str) -> bool {
    matches!(
        res,
        "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLN" | "GLU" | "GLY" | "HIS" | "ILE" | "LEU"
            | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL"
    )
}

fn three_to_one_letter(res: &str) -> char {
    match res {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        _ => 'X',
    }
}

/// Sanitizes a raw 1-letter amino acid or FASTA string by stripping `>` header lines and
/// whitespace, uppercasing, and validating that every residue is one of the 20 standard amino acids.
pub fn sanitize_amino_acid_sequence(raw: &str) -> Result<String, PhysicsError> {
    let mut seq = String::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') {
            continue;
        }
        for ch in trimmed.chars() {
            if ch.is_ascii_whitespace() {
                continue;
            }
            let up = ch.to_ascii_uppercase();
            if matches!(
                up,
                'A' | 'C'
                    | 'D'
                    | 'E'
                    | 'F'
                    | 'G'
                    | 'H'
                    | 'I'
                    | 'K'
                    | 'L'
                    | 'M'
                    | 'N'
                    | 'P'
                    | 'Q'
                    | 'R'
                    | 'S'
                    | 'T'
                    | 'V'
                    | 'W'
                    | 'Y'
            ) {
                seq.push(up);
            } else {
                return Err(PhysicsError::PdbParse(format!(
                    "Unsupported amino-acid character '{}' in sequence (expected standard 20 AA codes)",
                    ch
                )));
            }
        }
    }
    if seq.len() < 2 {
        return Err(PhysicsError::PdbParse(
            "Polypeptide sequence must contain at least 2 amino-acid residues".into(),
        ));
    }
    Ok(seq)
}

fn one_to_three_letter(aa: char) -> &'static str {
    match aa.to_ascii_uppercase() {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "ALA",
    }
}

fn infer_element_from_name(atom_name: &str) -> &'static str {
    match atom_name.chars().next().unwrap_or('C') {
        'N' => "N",
        'O' => "O",
        'S' => "S",
        'H' => "H",
        _ => "C",
    }
}

fn element_vdw_params(elem: &str, atom_name: &str) -> (f32, f32, f32, f32) {
    match elem {
        "H" => (1.008, 2.47, 0.0157, 1.20),
        "N" => (14.007, 3.25, 0.1700, 1.55),
        "O" => (15.999, 2.96, 0.2100, 1.50),
        "S" => (32.06, 3.56, 0.2500, 1.80),
        _ => {
            if atom_name == "C" {
                (12.011, 3.40, 0.0860, 1.70)
            } else {
                (12.011, 3.40, 0.1094, 1.70)
            }
        }
    }
}

fn partial_charge_heuristic(res_name: &str, atom_name: &str) -> f32 {
    match atom_name {
        "N" => -0.4157,
        "CA" => 0.0337,
        "C" => 0.5973,
        "O" | "OXT" => -0.5679,
        "NZ" if res_name == "LYS" => 0.75,
        "NH1" | "NH2" if res_name == "ARG" => 0.45,
        "OD1" | "OD2" if res_name == "ASP" => -0.45,
        "OE1" | "OE2" if res_name == "GLU" => -0.45,
        _ => 0.0,
    }
}

/// Returns `(mass_amu, net_sidechain_charge, effective_sigma_ang, hydrophobic_well_kcal)`
/// tuned so hydrophobic core residues (`W`, `Y`, `F`, `I`, `L`, `V`) drive burial collapse.
fn sidechain_bead_params(aa: char) -> (f32, f32, f32, f32) {
    match aa.to_ascii_uppercase() {
        'W' => (130.15, 0.0, 4.30, 0.58), // Trp6 core anchor in Trp-cage
        'Y' => (107.11, 0.0, 4.10, 0.46),
        'F' => (91.11, 0.0, 4.05, 0.50),
        'I' | 'L' => (57.11, 0.0, 3.90, 0.44),
        'V' => (43.09, 0.0, 3.75, 0.38),
        'M' => (75.13, 0.0, 3.95, 0.40),
        'P' => (42.08, 0.0, 3.65, 0.30),
        'A' => (15.03, 0.0, 3.50, 0.22),
        'C' => (47.09, 0.0, 3.65, 0.32),
        'K' => (73.14, 1.0, 3.95, 0.16),  // Salt-bridge Asp9-Arg16 / Lys8
        'R' => (101.15, 1.0, 4.10, 0.18),
        'D' => (58.04, -1.0, 3.70, 0.16),
        'E' => (72.06, -1.0, 3.85, 0.16),
        'S' | 'T' => (38.05, 0.0, 3.55, 0.20),
        'N' | 'Q' | 'H' => (65.07, 0.0, 3.80, 0.22),
        _ => (29.06, 0.0, 3.60, 0.22),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::{compute_sasa_angstrom2, evaluate_forces_and_energy};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDirGuard(std::path::PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn trp_cage_compressed_bytes_and_pdb_roundtrip() {
        let (topo, coords) = ProteinTopology::build_trp_cage_seed();
        assert_eq!(topo.residue_count, 20);
        assert_eq!(topo.sequence_one_letter, "NLYIQWLKDGGPSSGRPPPS");
        assert_eq!(coords.len(), topo.atoms.len());

        let bytes = topo.to_compressed_bytes().expect("compress topology");
        let decoded = ProteinTopology::from_compressed_bytes(&bytes).expect("decompress topology");
        assert_eq!(topo, decoded);
        assert!(ProteinTopology::from_compressed_bytes(b"BAD").is_err());

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("beofold_topo_test_{}_{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let _guard = TempDirGuard(dir.clone());
        let pdb_path = dir.join("trp_cage.pdb");

        let sasa = compute_sasa_angstrom2(&topo, &coords);
        assert!(sasa > 0.0 && sasa.is_finite());

        let mut forces = vec![[0.0f32; 3]; topo.atoms.len()];
        let breakdown = evaluate_forces_and_energy(&topo, &coords, &mut forces);
        assert!(breakdown.total_kcal.is_finite());

        topo.write_pdb_snapshot(&coords, &pdb_path, "test snapshot", breakdown.total_kcal)
            .expect("write pdb");
        let (reloaded, reloaded_coords) =
            ProteinTopology::from_pdb_file(&pdb_path).expect("reload pdb");
        assert_eq!(reloaded.residue_count, 20);
        assert_eq!(reloaded.sequence_one_letter, "NLYIQWLKDGGPSSGRPPPS");
        assert_eq!(reloaded_coords.len(), topo.atoms.len());
    }

    #[test]
    fn sanitize_amino_acid_sequence_strips_fasta_and_rejects_invalid() {
        let fasta = ">sp|1L2Y|Trp-cage\n nlyiq wlkdg\n gpssg rppps \n";
        let clean = sanitize_amino_acid_sequence(fasta).expect("valid FASTA sequence");
        assert_eq!(clean, "NLYIQWLKDGGPSSGRPPPS");

        assert!(sanitize_amino_acid_sequence(">only_header\nA").is_err());
        assert!(sanitize_amino_acid_sequence("ACDEFX").is_err());
    }

    #[test]
    fn build_polypeptide_from_sequence_normalizes_lowercase_glycine() {
        let (lower_topo, lower_coords) =
            ProteinTopology::build_polypeptide_from_sequence("g", "agly");
        let (upper_topo, upper_coords) =
            ProteinTopology::build_polypeptide_from_sequence("g", "AGLY");

        assert_eq!(lower_topo.atoms.len(), 19);
        assert_eq!(lower_topo.atoms.len(), upper_topo.atoms.len());
        assert_eq!(lower_topo.sequence_one_letter, "AGLY");
        assert_eq!(lower_topo, upper_topo);
        assert_eq!(lower_coords, upper_coords);
        assert!(
            !lower_topo
                .atoms
                .iter()
                .any(|a| a.res_name == "GLY" && a.name == "CB"),
            "GLY residue must not contain a CB sidechain bead"
        );
    }
}
