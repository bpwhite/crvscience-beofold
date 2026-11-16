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

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod lan_client;
mod live_sandbox;
mod msm_tab;
mod plots;
mod viewport3d;

use app::{BeoFoldGuiApp, GuiLaunchConfig};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "fold-gui",
    version,
    about = "BeoFold Desktop Supervisor Workstation (LAN Beowulf Cluster + Apache WAN Federation + MSM Free-Energy Surface + 3D Conformation Viewer)"
)]
struct GuiCli {
    /// Initial LAN Beowulf peer socket address (`127.0.0.1:7443` or any LAN node IP:port).
    #[arg(long, default_value = "127.0.0.1:7443")]
    lan_peer: String,

    /// Shared Beowulf LAN cluster secret authenticating QUIC TLS 1.3 RPC frames.
    #[arg(long, default_value = "beofold-home-cluster")]
    cluster_secret: String,

    /// Optional Apache HTTP Web Sync Server URL (e.g. `https://yourdomain.com/beofold/api/v1`).
    #[arg(long, default_value = "")]
    sync_url: String,

    /// Optional local `redb` campaign directory for loading offline trajectory checkpoints & MSM vaults.
    #[arg(long, default_value = "./beofold_data")]
    data_dir: String,

    /// Disable `wgpu` WGSL compute shaders in the built-in live folding studio.
    #[arg(long, default_value_t = false)]
    no_gpu: bool,

    /// Render a single headless offscreen 1600x1000 RGBA PNG screenshot and exit without opening a window.
    #[arg(long, value_name = "OUT_PNG")]
    screenshot: Option<std::path::PathBuf>,

    /// Select which view preset to capture in headless screenshot mode (1=3D Studio, 2=MSM Surface, 3=LAN Matrix).
    #[arg(long = "screenshot-view", visible_alias = "view", default_value_t = 1)]
    screenshot_view: u8,
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    struct ThreadUnparker(std::thread::Thread);
    impl std::task::Wake for ThreadUnparker {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(ThreadUnparker(
        std::thread::current(),
    )));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFF_u32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if (c & 1) != 0 {
                0xEDB8_8320_u32 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

fn push_png_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    out.extend_from_slice(&png_crc32(&out[start..]).to_be_bytes());
}

fn write_rgba_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), String> {
    let row_bytes = (width as usize) * 4;
    if rgba.len() != row_bytes * (height as usize) {
        return Err("RGBA buffer length mismatch".to_string());
    }
    let mut raw = Vec::with_capacity((row_bytes + 1) * (height as usize));
    for row in rgba.chunks_exact(row_bytes) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut zlib = Vec::with_capacity(raw.len() + 256);
    zlib.extend_from_slice(&[0x78, 0x01]);
    let mut chunks = raw.chunks(65_535).peekable();
    while let Some(chunk) = chunks.next() {
        zlib.push(if chunks.peek().is_none() { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(chunk);
    }
    let (mut s1, mut s2) = (1_u32, 0_u32);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    zlib.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());
    let mut png = Vec::with_capacity(zlib.len() + 64);
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    push_png_chunk(&mut png, b"IHDR", &ihdr);
    push_png_chunk(&mut png, b"IDAT", &zlib);
    push_png_chunk(&mut png, b"IEND", &[]);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {e}"))?;
        }
    }
    std::fs::write(path, &png).map_err(|e| format!("Failed to write PNG: {e}"))
}

fn run_headless_screenshot(
    cfg: GuiLaunchConfig,
    out_path: &std::path::Path,
    view: u8,
) -> Result<(), String> {
    use eframe::egui_wgpu::{self, wgpu};

    const WIDTH: u32 = 1600;
    const HEIGHT: u32 = 1000;
    const BYTES_PER_ROW: u32 = WIDTH * 4;

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|| {
        block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
    })
    .ok_or_else(|| "No wgpu adapter available for offscreen screenshot.".to_string())?;

    let (device, queue) =
        block_on_future(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .map_err(|e| format!("wgpu request_device failed: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

    let ctx = egui::Context::default();
    let mut app = BeoFoldGuiApp::new_with_ctx(&ctx, cfg);
    app.populate_for_screenshot(view);

    let extent = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let render_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("beofold_headless_target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let render_view = render_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("beofold_headless_readback"),
        size: (BYTES_PER_ROW as u64) * (HEIGHT as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    for frame_idx in 0..4_u32 {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(WIDTH as f32, HEIGHT as f32),
            )),
            predicted_dt: 1.0 / 30.0,
            time: Some((frame_idx as f64) * (1.0 / 30.0)),
            ..Default::default()
        };

        let full_output = ctx.run(raw_input, |ctx| {
            app.render_ui(ctx);
        });
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [WIDTH, HEIGHT],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, delta) in &full_output.textures_delta.set {
            renderer.update_texture(&device, &queue, *id, delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("beofold_headless_encoder"),
        });
        let user_cmds =
            renderer.update_buffers(&device, &queue, &mut encoder, &paint_jobs, &screen_descriptor);

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("beofold_headless_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 15.0 / 255.0,
                            g: 20.0 / 255.0,
                            b: 28.0 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rpass = rpass.forget_lifetime();
            renderer.render(&mut rpass, &paint_jobs, &screen_descriptor);
        }

        if frame_idx == 3 {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &render_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(BYTES_PER_ROW),
                        rows_per_image: Some(HEIGHT),
                    },
                },
                extent,
            );
        }

        queue.submit(user_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        device.poll(wgpu::Maintain::Wait);

        for id in &full_output.textures_delta.free {
            renderer.free_texture(id);
        }
    }

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("Readback channel error: {e}"))?
        .map_err(|e| format!("Staging buffer map failed: {e}"))?;

    let rgba = slice.get_mapped_range().to_vec();
    let _ = slice;
    staging_buf.unmap();

    write_rgba_png(out_path, WIDTH, HEIGHT, &rgba)
}

fn main() -> eframe::Result<()> {
    let cli = GuiCli::parse();
    let launch_cfg = GuiLaunchConfig {
        lan_peer: cli.lan_peer,
        cluster_secret: cli.cluster_secret,
        sync_url: cli.sync_url,
        vault_dir: cli.data_dir,
        use_gpu: !cli.no_gpu,
    };

    if let Some(ref out_path) = cli.screenshot {
        let mut screenshot_cfg = launch_cfg;
        if cli.screenshot_view == 3 {
            screenshot_cfg.lan_peer.clear();
            screenshot_cfg.sync_url.clear();
        }
        if let Err(err) = run_headless_screenshot(screenshot_cfg, out_path, cli.screenshot_view) {
            eprintln!("[BeoFold Screenshot Error] {err}");
            std::process::exit(1);
        }
        println!(
            "Saved {} (1600x1000 RGBA offscreen screenshot, view {})",
            out_path.display(),
            cli.screenshot_view
        );
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("BeoFold v1.3 — Distributed Protein Folding & MSM Supervisor")
            .with_inner_size([1400.0, 860.0])
            .with_min_inner_size([1020.0, 640.0]),
        ..Default::default()
    };

    eframe::run_native(
        "BeoFold Supervisor Workstation",
        native_options,
        Box::new(move |cc| Ok(Box::new(BeoFoldGuiApp::new(cc, launch_cfg)))),
    )
}
