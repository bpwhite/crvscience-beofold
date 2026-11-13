use fold_proto::WorkUnitId;
use std::f64::consts::TAU;

/// Deterministic, cross-platform pure-Rust ChaCha8 stream PRNG whose full internal state
/// serializes into `TrajectoryCheckpoint` for bit-exact multi-Generation restarts and canary audits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChaCha8Rng {
    state: [u32; 16],
    counter: u64,
}

impl ChaCha8Rng {
    /// Seeds deterministically from a `WorkUnitId` (`project`, `run`, `clone`) and campaign salt.
    pub fn from_prcg_seed(wu: WorkUnitId, campaign_salt: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"beofold-v1-chacha8-seed");
        hasher.update(campaign_salt);
        hasher.update(&wu.project.to_le_bytes());
        hasher.update(&wu.run.to_le_bytes());
        hasher.update(&wu.clone.to_le_bytes());
        let key_bytes = *hasher.finalize().as_bytes();
        Self::from_key_and_stream(key_bytes, ((wu.run as u64) << 32) | (wu.clone as u64))
    }

    pub fn from_key_and_stream(key: [u8; 32], stream_id: u64) -> Self {
        let mut state = [0u32; 16];
        // RFC 8439 "expand 32-byte k" constants
        state[0] = 0x6170_7865;
        state[1] = 0x3320_646e;
        state[2] = 0x7962_2d32;
        state[3] = 0x6b20_6574;
        for i in 0..8 {
            let off = i * 4;
            state[4 + i] = u32::from_le_bytes([key[off], key[off + 1], key[off + 2], key[off + 3]]);
        }
        state[12] = 0;
        state[13] = 0;
        state[14] = (stream_id & 0xffff_ffff) as u32;
        state[15] = (stream_id >> 32) as u32;
        Self { state, counter: 0 }
    }

    /// Restores exact PRNG state from a checkpointed generation boundary.
    pub const fn from_raw_parts(state: [u32; 16], counter: u64) -> Self {
        Self { state, counter }
    }

    pub const fn to_raw_parts(&self) -> ([u32; 16], u64) {
        (self.state, self.counter)
    }

    #[inline]
    fn quarter_round(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        x[a] = x[a].wrapping_add(x[b]);
        x[d] ^= x[a];
        x[d] = x[d].rotate_left(16);

        x[c] = x[c].wrapping_add(x[d]);
        x[b] ^= x[c];
        x[b] = x[b].rotate_left(12);

        x[a] = x[a].wrapping_add(x[b]);
        x[d] ^= x[a];
        x[d] = x[d].rotate_left(8);

        x[c] = x[c].wrapping_add(x[d]);
        x[b] ^= x[c];
        x[b] = x[b].rotate_left(7);
    }

    fn generate_block(&self, block_idx: u64) -> [u32; 16] {
        let mut base = self.state;
        base[12] = (block_idx & 0xffff_ffff) as u32;
        base[13] = (block_idx >> 32) as u32;

        let mut working = base;
        // 8 rounds = 4 double-rounds
        for _ in 0..4 {
            Self::quarter_round(&mut working, 0, 4, 8, 12);
            Self::quarter_round(&mut working, 1, 5, 9, 13);
            Self::quarter_round(&mut working, 2, 6, 10, 14);
            Self::quarter_round(&mut working, 3, 7, 11, 15);

            Self::quarter_round(&mut working, 0, 5, 10, 15);
            Self::quarter_round(&mut working, 1, 6, 11, 12);
            Self::quarter_round(&mut working, 2, 7, 8, 13);
            Self::quarter_round(&mut working, 3, 4, 9, 14);
        }

        for i in 0..16 {
            working[i] = working[i].wrapping_add(base[i]);
        }
        working
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let block_idx = self.counter / 16;
        let word_idx = (self.counter % 16) as usize;
        self.counter = self.counter.wrapping_add(1);
        let block = self.generate_block(block_idx);
        block[word_idx]
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Uniform open-interval float in `(0.0, 1.0)` with 53-bit mantissa precision.
    #[inline]
    pub fn next_f64_open01(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 1;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal `N(0, 1)` variate via deterministic Box-Muller transform.
    #[inline]
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Generates a 3D standard normal vector `[N(0,1), N(0,1), N(0,1)]`.
    #[inline]
    pub fn next_gaussian_vec3(&mut self) -> [f64; 3] {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        let r1 = (-2.0 * u1.ln()).sqrt();
        let theta1 = TAU * u2;

        let u3 = self.next_f64_open01();
        let u4 = self.next_f64_open01();
        let r2 = (-2.0 * u3.ln()).sqrt();
        let theta2 = TAU * u4;

        [r1 * theta1.cos(), r1 * theta1.sin(), r2 * theta2.cos()]
    }
}
