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
