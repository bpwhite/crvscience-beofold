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

use rkyv::{Archive, Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Four-dimensional Markov State Model trajectory chunk identifier:
/// Project (P) -> Run (R) -> Clone (C) -> Generation (G).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkUnitId {
    pub project: u32,
    pub run: u32,
    pub clone: u32,
    pub generation: u32,
}

impl WorkUnitId {
    pub const fn new(project: u32, run: u32, clone: u32, generation: u32) -> Self {
        Self {
            project,
            run,
            clone,
            generation,
        }
    }

    /// Returns the sequential successor generation `(P, R, C, G + 1)`.
    pub const fn next_generation(self) -> Self {
        Self {
            project: self.project,
            run: self.run,
            clone: self.clone,
            generation: self.generation.saturating_add(1),
        }
    }

    /// Encodes into a 16-byte big-endian key so B-tree range scans in `redb`
    /// preserve strict `(project, run, clone, generation)` lexicographic order.
    pub fn to_key_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&self.project.to_be_bytes());
        out[4..8].copy_from_slice(&self.run.to_be_bytes());
        out[8..12].copy_from_slice(&self.clone.to_be_bytes());
        out[12..16].copy_from_slice(&self.generation.to_be_bytes());
        out
    }

    /// Reconstructs a `WorkUnitId` from its 16-byte big-endian storage key.
    pub fn from_key_bytes(bytes: [u8; 16]) -> Self {
        let project = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let run = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let clone = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let generation = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        Self {
            project,
            run,
            clone,
            generation,
        }
    }
}

impl fmt::Display for WorkUnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P{}-R{}-C{}-G{}",
            self.project, self.run, self.clone, self.generation
        )
    }
}

impl FromStr for WorkUnitId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.trim().split('-').collect();
        if parts.len() != 4 {
            return Err(format!("invalid PRCG string '{}': expected P#-R#-C#-G#", s));
        }
        let parse_part = |part: &str, prefix: char| -> Result<u32, String> {
            let rest = part
                .strip_prefix(prefix)
                .ok_or_else(|| format!("missing '{}' prefix in '{}'", prefix, part))?;
            rest.parse::<u32>()
                .map_err(|e| format!("invalid integer in '{}': {}", part, e))
        };
        Ok(Self {
            project: parse_part(parts[0], 'P')?,
            run: parse_part(parts[1], 'R')?,
            clone: parse_part(parts[2], 'C')?,
            generation: parse_part(parts[3], 'G')?,
        })
    }
}

/// Benchmarked hardware capability tier for heterogeneous Windows and Linux nodes.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTier {
    /// Discrete DirectX 12 / Vulkan GPU via `wgpu` compute shaders.
    TierS,
    /// High-throughput multi-core CPU (`rayon` work-stealing + SIMD).
    TierA,
    /// Lightweight or older 2-to-4 core CPU / background throttled host.
    TierB,
}

impl ComputeTier {
    /// Recommended integration steps per leased generation chunk for target 5-15 min wall-clock turnaround.
    pub const fn recommended_steps(self) -> u64 {
        match self {
            ComputeTier::TierS => 1_000_000,
            ComputeTier::TierA => 250_000,
            ComputeTier::TierB => 50_000,
        }
    }
}

impl fmt::Display for ComputeTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeTier::TierS => write!(f, "Tier S (Discrete GPU)"),
            ComputeTier::TierA => write!(f, "Tier A (Many-Core CPU)"),
            ComputeTier::TierB => write!(f, "Tier B (Light CPU)"),
        }
    }
}

/// Advertised capability summary for LAN Beowulf auto-discovery and WAN cluster registration.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NodeCapabilityProfile {
    pub node_pubkey: [u8; 32],
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub logical_cores: u32,
    pub total_ram_mb: u64,
    pub villin_ns_per_day: f64,
    pub tier: ComputeTier,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workunit_id_generation_order_and_parsing() {
        let wu = WorkUnitId::new(12, 3, 4, 5);
        assert_eq!(wu.next_generation(), WorkUnitId::new(12, 3, 4, 6));
        assert_eq!(WorkUnitId::new(1, 0, 0, u32::MAX).next_generation().generation, u32::MAX);

        let wire_bytes = wu.to_key_bytes();
        let next_bytes = wu.next_generation().to_key_bytes();
        assert!(wire_bytes < next_bytes);
        assert_eq!(WorkUnitId::from_key_bytes(wire_bytes), wu);

        let text = wu.to_string();
        assert_eq!(text, "P12-R3-C4-G5");
        assert_eq!(text.parse::<WorkUnitId>().unwrap(), wu);

        assert!("P1-R2-C3".parse::<WorkUnitId>().is_err());
        assert!("X1-R2-C3-G4".parse::<WorkUnitId>().is_err());
        assert!("P1-R2-C3-Gabc".parse::<WorkUnitId>().is_err());
    }

    #[test]
    fn compute_tier_and_node_capability_wire_roundtrip() {
        assert_eq!(ComputeTier::TierS.recommended_steps(), 1_000_000);
        assert_eq!(ComputeTier::TierA.recommended_steps(), 250_000);
        assert_eq!(ComputeTier::TierB.recommended_steps(), 50_000);
        assert!(ComputeTier::TierS.to_string().contains("Tier S"));
        assert!(ComputeTier::TierA.to_string().contains("Tier A"));
        assert!(ComputeTier::TierB.to_string().contains("Tier B"));

        let profile = NodeCapabilityProfile {
            node_pubkey: [0xAB; 32],
            hostname: "beofold-worker-01".into(),
            os: "windows".into(),
            arch: "x86_64".into(),
            logical_cores: 16,
            total_ram_mb: 65_536,
            villin_ns_per_day: 42.5,
            tier: ComputeTier::TierA,
        };
        let frame = profile.to_compressed_frame().expect("encode profile");
        let decoded = NodeCapabilityProfile::from_compressed_frame(&frame).expect("decode profile");
        assert_eq!(profile, decoded);
    }
}
