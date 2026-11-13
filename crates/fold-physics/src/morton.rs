use std::collections::HashMap;

/// Spreads 10 bits of a coordinate integer across 30 bits (`0b001001001...`) for 3D Morton coding.
#[inline]
fn part_1_by_2(mut x: u32) -> u32 {
    x &= 0x0000_03ff;
    x = (x ^ (x << 16)) & 0xff00_00ff;
    x = (x ^ (x << 8)) & 0x0300_f00f;
    x = (x ^ (x << 4)) & 0x030c_30c3;
    x = (x ^ (x << 2)) & 0x0924_9249;
    x
}

/// Interleaves quantized 10-bit `(ix, iy, iz)` cell coordinates into a 30-bit 3D Z-order Morton code.
#[inline]
pub fn morton_encode_3d(ix: u32, iy: u32, iz: u32) -> u32 {
    part_1_by_2(ix) | (part_1_by_2(iy) << 1) | (part_1_by_2(iz) << 2)
}

/// Spatial cell-list grid accelerating non-bonded pair discovery to `O(N)` via 3D cell bucketing
/// ordered along a Morton Z-order curve for CPU cache locality.
pub struct SpatialNeighborGrid {
    pub cutoff_ang: f32,
    pub cutoff_sq: f32,
    min_corner: [f32; 3],
    inv_cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    morton_ordered_atoms: Vec<usize>,
}

impl SpatialNeighborGrid {
    pub fn build(positions: &[[f32; 3]], cutoff_ang: f32) -> Self {
        let cell_size = cutoff_ang.max(3.5);
        let inv_cell_size = 1.0 / cell_size;
        let mut min_corner = [f32::INFINITY; 3];
        for p in positions {
            for d in 0..3 {
                if p[d] < min_corner[d] {
                    min_corner[d] = p[d];
                }
            }
        }
        if !min_corner[0].is_finite() {
            min_corner = [0.0; 3];
        }

        let mut keyed: Vec<(u32, usize, (i32, i32, i32))> = Vec::with_capacity(positions.len());
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> =
            HashMap::with_capacity(positions.len() / 4 + 1);

        for (idx, p) in positions.iter().enumerate() {
            let cx = ((p[0] - min_corner[0]) * inv_cell_size).floor() as i32;
            let cy = ((p[1] - min_corner[1]) * inv_cell_size).floor() as i32;
            let cz = ((p[2] - min_corner[2]) * inv_cell_size).floor() as i32;
            let mx = (cx.clamp(0, 1023)) as u32;
            let my = (cy.clamp(0, 1023)) as u32;
            let mz = (cz.clamp(0, 1023)) as u32;
            let code = morton_encode_3d(mx, my, mz);
            keyed.push((code, idx, (cx, cy, cz)));
            cells.entry((cx, cy, cz)).or_default().push(idx);
        }

        keyed.sort_unstable_by_key(|entry| entry.0);
        let morton_ordered_atoms = keyed.into_iter().map(|e| e.1).collect();

        Self {
            cutoff_ang,
            cutoff_sq: cutoff_ang * cutoff_ang,
            min_corner,
            inv_cell_size,
            cells,
            morton_ordered_atoms,
        }
    }

    pub fn morton_order(&self) -> &[usize] {
        &self.morton_ordered_atoms
    }

    /// Visits all candidate neighbor atom indices `j` in the 27-cell Moore neighborhood of `pos`.
    #[inline]
    pub fn for_each_neighbor<F>(&self, pos: [f32; 3], mut f: F)
    where
        F: FnMut(usize),
    {
        let cx = ((pos[0] - self.min_corner[0]) * self.inv_cell_size).floor() as i32;
        let cy = ((pos[1] - self.min_corner[1]) * self.inv_cell_size).floor() as i32;
        let cz = ((pos[2] - self.min_corner[2]) * self.inv_cell_size).floor() as i32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in bucket {
                            f(j);
                        }
                    }
                }
            }
        }
    }
}
