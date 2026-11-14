use crate::topology::ProteinTopology;
use crate::COULOMB_KCAL_A_E2;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use std::sync::mpsc;
use thiserror::Error;
use wgpu::util::DeviceExt;

pub const GPU_NONBONDED_CUTOFF_ANG: f32 = 12.0;

#[derive(Error, Debug)]
pub enum GpuComputeError {
    #[error("no compatible DirectX 12, Vulkan, Metal, or GL compute adapter found")]
    NoAdapter,
    #[error("GPU device initialization failed: {0}")]
    DeviceInit(String),
    #[error("GPU buffer map/readback error: {0}")]
    BufferMap(String),
}

/// Hardware summary returned by `probe_gpu_adapter` for capability tiering and GUI display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub is_discrete_gpu: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuUniforms {
    atom_count: u32,
    cutoff_ang: f32,
    coulomb_quarter: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuAtomParams {
    lj_sigma_ang: f32,
    lj_epsilon_kcal: f32,
    charge_e: f32,
    flags: u32,
    excl_offset: u32,
    excl_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuAtomOutput {
    fx: f32,
    fy: f32,
    fz: f32,
    _pad0: f32,
    e_lj: f32,
    e_elec: f32,
    e_solv: f32,
    _pad1: f32,
}

/// Probes local DirectX 12 (Windows), Vulkan (Windows/Linux), Metal (macOS), or GL adapters
/// without requiring a display window handle.
pub fn probe_gpu_adapter() -> Option<GpuAdapterSummary> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;

    let info = adapter.get_info();
    Some(GpuAdapterSummary {
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        is_discrete_gpu: info.device_type == wgpu::DeviceType::DiscreteGpu,
    })
}

/// Persistent `wgpu` compute pipeline bound to a specific `ProteinTopology` for evaluating
/// O(N^2) tiled shared-memory non-bonded forces and energies across GPU workgroups (`64` threads/group).
pub struct GpuForceEngine {
    summary: GpuAdapterSummary,
    atom_count: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pos_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    download_buf: wgpu::Buffer,
    out_byte_len: u64,
}

impl GpuForceEngine {
    pub fn try_new(topology: &ProteinTopology) -> Result<Self, GpuComputeError> {
        let atom_count = topology.atoms.len();
        if atom_count == 0 {
            return Err(GpuComputeError::DeviceInit("empty protein topology".into()));
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|_| GpuComputeError::NoAdapter)?;

        let info = adapter.get_info();
        let summary = GpuAdapterSummary {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            is_discrete_gpu: info.device_type == wgpu::DeviceType::DiscreteGpu,
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("beofold-gpu-compute"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| GpuComputeError::DeviceInit(e.to_string()))?;

        let (atom_params, mut csr_exclusions) = build_csr_exclusion_tables(topology);
        if csr_exclusions.is_empty() {
            // Ensure non-zero storage buffer binding size even for single-atom test cases
            csr_exclusions.push(0u32);
        }

        let uniforms = GpuUniforms {
            atom_count: atom_count as u32,
            cutoff_ang: GPU_NONBONDED_CUTOFF_ANG,
            coulomb_quarter: (COULOMB_KCAL_A_E2 / 4.0) as f32,
            _pad: 0,
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beofold-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beofold-atom-params"),
            contents: bytemuck::cast_slice(&atom_params),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let csr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beofold-csr-exclusions"),
            contents: bytemuck::cast_slice(&csr_exclusions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pos_byte_len = (atom_count * std::mem::size_of::<[f32; 4]>()) as u64;
        let out_byte_len = (atom_count * std::mem::size_of::<GpuAtomOutput>()) as u64;

        let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beofold-positions"),
            size: pos_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beofold-gpu-outputs"),
            size: out_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let download_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beofold-gpu-readback"),
            size: out_byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("beofold-nonbonded-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuUniforms>() as u64),
                    },
                    count: None,
                },
                storage_layout_entry(1, true, std::mem::size_of::<GpuAtomParams>() as u64),
                storage_layout_entry(2, true, 4),
                storage_layout_entry(3, true, 16),
                storage_layout_entry(4, false, std::mem::size_of::<GpuAtomOutput>() as u64),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beofold-nonbonded-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: csr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("nonbonded.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("beofold-nonbonded-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("beofold-nonbonded-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            summary,
            atom_count,
            device,
            queue,
            pipeline,
            bind_group,
            pos_buf,
            out_buf,
            download_buf,
            out_byte_len,
        })
    }

    pub fn adapter_summary(&self) -> &GpuAdapterSummary {
        &self.summary
    }

    /// Dispatches the WGSL tiled non-bonded force kernel and accumulates `(fx, fy, fz)` onto
    /// `forces_accum`, returning `(lj_kcal, electrostatics_gb_kcal, nonpolar_solvation_kcal)`.
    pub fn evaluate_nonbonded(
        &self,
        positions: &[[f32; 3]],
        forces_accum: &mut [[f32; 3]],
    ) -> Result<(f64, f64, f64), GpuComputeError> {
        assert_eq!(positions.len(), self.atom_count);
        assert_eq!(forces_accum.len(), self.atom_count);

        let mut packed_pos = Vec::with_capacity(self.atom_count);
        for p in positions {
            packed_pos.push([p[0], p[1], p[2], 0.0f32]);
        }
        self.queue
            .write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(&packed_pos));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beofold-nonbonded-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("beofold-nonbonded-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.atom_count.div_ceil(64)) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.download_buf, 0, self.out_byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.download_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuComputeError::BufferMap(e.to_string()))?;

        rx.recv()
            .map_err(|e| GpuComputeError::BufferMap(e.to_string()))?
            .map_err(|e| GpuComputeError::BufferMap(e.to_string()))?;

        let mut e_lj = 0.0f64;
        let mut e_elec = 0.0f64;
        let mut e_solv = 0.0f64;
        {
            let view = slice
                .get_mapped_range()
                .map_err(|e| GpuComputeError::BufferMap(e.to_string()))?;
            let gpu_out: &[GpuAtomOutput] = bytemuck::cast_slice(&view);
            for (i, out) in gpu_out.iter().enumerate() {
                forces_accum[i][0] += out.fx;
                forces_accum[i][1] += out.fy;
                forces_accum[i][2] += out.fz;
                e_lj += out.e_lj as f64;
                e_elec += out.e_elec as f64;
                e_solv += out.e_solv as f64;
            }
            // `view` (BufferView) MUST drop before `download_buf.unmap()` in wgpu 30
        }
        self.download_buf.unmap();

        Ok((e_lj, e_elec, e_solv))
    }
}

fn storage_layout_entry(
    binding: u32,
    read_only: bool,
    min_bytes: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_bytes.max(4)),
        },
        count: None,
    }
}

fn build_csr_exclusion_tables(topology: &ProteinTopology) -> (Vec<GpuAtomParams>, Vec<u32>) {
    let n = topology.atoms.len();
    let mut adj_lists: Vec<Vec<u32>> = vec![Vec::new(); n];

    // Policy 1u = Excluded (1-2 / 1-3)
    for &key in &topology.excluded_12_13 {
        let lo = (key >> 32) as usize;
        let hi = (key & 0xFFFF_FFFF) as usize;
        if lo < n && hi < n {
            adj_lists[lo].push((hi as u32 & 0x3FFF_FFFF) | (1u32 << 30));
            adj_lists[hi].push((lo as u32 & 0x3FFF_FFFF) | (1u32 << 30));
        }
    }

    // Policy 2u = Scaled14 (0.5 LJ, 1/1.2 electrostatics)
    for &key in &topology.scaled_14 {
        let lo = (key >> 32) as usize;
        let hi = (key & 0xFFFF_FFFF) as usize;
        if lo < n && hi < n {
            adj_lists[lo].push((hi as u32 & 0x3FFF_FFFF) | (2u32 << 30));
            adj_lists[hi].push((lo as u32 & 0x3FFF_FFFF) | (2u32 << 30));
        }
    }

    let mut params = Vec::with_capacity(n);
    let mut flat_csr = Vec::new();
    for (i, atom) in topology.atoms.iter().enumerate() {
        // Sort by partner index (low 30 bits) for binary search in WGSL
        adj_lists[i].sort_unstable_by_key(|&entry| entry & 0x3FFF_FFFF);
        let offset = flat_csr.len() as u32;
        let count = adj_lists[i].len() as u32;
        flat_csr.extend_from_slice(&adj_lists[i]);

        params.push(GpuAtomParams {
            lj_sigma_ang: atom.lj_sigma_ang,
            lj_epsilon_kcal: atom.lj_epsilon_kcal,
            charge_e: atom.charge_e,
            flags: if atom.is_backbone { 1u32 } else { 0u32 },
            excl_offset: offset,
            excl_count: count,
            _pad0: 0,
            _pad1: 0,
        });
    }

    (params, flat_csr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::{evaluate_forces_and_energy, evaluate_forces_and_energy_with_gpu};
    use crate::integrator::minimize_steric_clashes;

    #[test]
    fn cpu_vs_gpu_nonbonded_force_and_energy_parity_within_1e4() {
        let (topo, mut coords) = ProteinTopology::build_trp_cage_seed();
        minimize_steric_clashes(&topo, &mut coords, 80);

        let Ok(gpu) = GpuForceEngine::try_new(&topo) else {
            // Headless CI container without a Vulkan/DX12/Metal ICD falls back cleanly to CPU
            return;
        };

        let mut cpu_forces = vec![[0.0f32; 3]; topo.atoms.len()];
        let mut gpu_forces = vec![[0.0f32; 3]; topo.atoms.len()];

        let cpu_e = evaluate_forces_and_energy(&topo, &coords, &mut cpu_forces);
        let gpu_e = evaluate_forces_and_energy_with_gpu(&topo, Some(&gpu), &coords, &mut gpu_forces);

        let rel_total = (cpu_e.total_kcal - gpu_e.total_kcal).abs() / cpu_e.total_kcal.abs().max(1.0);
        assert!(
            rel_total < 1.0e-4,
            "CPU vs GPU total energy relative error {} exceeded 1e-4 (CPU={}, GPU={})",
            rel_total,
            cpu_e.total_kcal,
            gpu_e.total_kcal
        );

        for i in 0..topo.atoms.len() {
            for d in 0..3 {
                let denom = cpu_forces[i][d].abs().max(1.0);
                let rel_f = (cpu_forces[i][d] - gpu_forces[i][d]).abs() / denom;
                assert!(
                    rel_f < 1.0e-4,
                    "atom {} axis {} force parity error {} exceeded 1e-4",
                    i,
                    d,
                    rel_f
                );
            }
        }
    }
}
