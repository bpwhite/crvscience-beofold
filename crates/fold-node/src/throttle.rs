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

use sysinfo::System;

/// Monitors foreground CPU utilization on interactive Windows and Linux desktops (`--auto-throttle`)
/// so gaming or active workstation tasks automatically pause new Beowulf compute dispatches.
pub struct AutoThrottleMonitor {
    enabled: bool,
    max_system_cpu_pct: f32,
    sys: System,
}

impl AutoThrottleMonitor {
    pub fn new(enabled: bool, max_system_cpu_pct: f32) -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_usage();
        Self {
            enabled,
            max_system_cpu_pct: max_system_cpu_pct.clamp(15.0, 99.0),
            sys,
        }
    }

    /// Refreshes OS CPU counters and returns `(global_cpu_pct, should_throttle)`.
    pub fn sample(&mut self, currently_folding: bool) -> (f32, bool) {
        self.sys.refresh_cpu_usage();
        let cpus = self.sys.cpus();
        let avg_pct = if cpus.is_empty() {
            0.0
        } else {
            cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / (cpus.len() as f32)
        };

        if !self.enabled {
            return (avg_pct, false);
        }

        // Only pause *between* chunks when idle-system background load already exceeds ceiling
        let throttled = !currently_folding && avg_pct > self.max_system_cpu_pct;
        (avg_pct, throttled)
    }
}
