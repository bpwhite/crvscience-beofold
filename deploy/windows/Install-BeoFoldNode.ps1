# Copyright (C) 2026 CRV Science Network (https://crvscience.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://gnu.org>.

#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Registers BeoFold (`fold-node.exe`) on Windows 10/11 or Windows Server to run unattended on boot
    with DirectX 12 / Vulkan GPU compute access and scoped LAN Beowulf firewall rules.

.DESCRIPTION
    Supports two unattended execution modes:
      1. `ScheduledTask` (Default): Registers a built-in Windows boot task (`AtStartup` under
         `NT AUTHORITY\SYSTEM`, `RunLevel Highest`) via `Register-ScheduledTask`. Requires zero
         third-party service wrappers, restarts automatically on crash, and preserves full WDDM
         DirectX 12 / Vulkan hardware adapter visibility for `wgpu`.
      2. `NssmService`: Registers a native `services.msc` Windows Service via `nssm.exe` (Non-Sucking
         Service Manager) with Console Ctrl+C stop signaling (`AppStopMethodConsole = 15000`) so
         `redb` ACID checkpoints flush cleanly on service stop.

.EXAMPLE
    .\deploy\windows\Install-BeoFoldNode.ps1 -ClusterSecret "my-home-lan-key" -AutoThrottle
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$BinaryPath = ".\target\release\fold-node.exe",

    [Parameter(Mandatory = $false)]
    [ValidateSet("ScheduledTask", "NssmService")]
    [string]$Mode = "ScheduledTask",

    [Parameter(Mandatory = $false)]
    [string]$ClusterSecret = "beofold-home-cluster",

    [Parameter(Mandatory = $false)]
    [string]$BindAddress = "0.0.0.0:7443",

    [Parameter(Mandatory = $false)]
    [uint16]$QuicPort = 7443,

    [Parameter(Mandatory = $false)]
    [string]$SyncUrl = "",

    [Parameter(Mandatory = $false)]
    [string]$ClusterName = "Windows-Beowulf-Cluster",

    [Parameter(Mandatory = $false)]
    [string]$NodeLabel = $env:COMPUTERNAME,

    [Parameter(Mandatory = $false)]
    [switch]$AutoThrottle,

    [Parameter(Mandatory = $false)]
    [double]$MaxCpuPercent = 80.0,

    [Parameter(Mandatory = $false)]
    [switch]$DisableGpu,

    [Parameter(Mandatory = $false)]
    [switch]$InitTrpCage,

    [Parameter(Mandatory = $false)]
    [switch]$AllowPublicProfileSubnet,

    [Parameter(Mandatory = $false)]
    [string]$NssmPath = "nssm.exe"
)

$ErrorActionPreference = "Stop"

$ResolvedBin = Resolve-Path -Path $BinaryPath -ErrorAction SilentlyContinue
if (-not $ResolvedBin) {
    throw "fold-node.exe not found at '$BinaryPath'. Run 'cargo build --release -p fold-node' first."
}

$InstallDir = "C:\Program Files\BeoFold"
$DataRoot   = "C:\ProgramData\BeoFold"
$NodeVault  = Join-Path $DataRoot "node-vault"
$LogDir     = Join-Path $DataRoot "logs"
$ConfigPath = Join-Path $DataRoot "node-config.json"
$TargetExe  = Join-Path $InstallDir "fold-node.exe"
$RunnerPs1  = Join-Path $InstallDir "Start-BeoFoldNodeRunner.ps1"

New-Item -ItemType Directory -Force -Path $InstallDir, $DataRoot, $NodeVault, $LogDir | Out-Null
Copy-Item -Path $ResolvedBin.Path -Destination $TargetExe -Force
Write-Host "[+] Installed binary to $TargetExe"

# Write restricted-ACL JSON config so --cluster-secret is not exposed on world-readable command lines
$ConfigObj = [ordered]@{
    DataDir       = $NodeVault
    BindAddress   = $BindAddress
    ClusterSecret = $ClusterSecret
    SyncUrl       = $SyncUrl
    ClusterName   = $ClusterName
    NodeLabel     = $NodeLabel
    AutoThrottle  = [bool]$AutoThrottle
    MaxCpuPercent = $MaxCpuPercent
    DisableGpu    = [bool]$DisableGpu
    InitTrpCage   = [bool]$InitTrpCage
}
$ConfigObj | ConvertTo-Json -Depth 4 | Set-Content -Path $ConfigPath -Encoding UTF8
& icacls.exe $ConfigPath /inheritance:r /grant:r "NT AUTHORITY\SYSTEM:(F)" "BUILTIN\Administrators:(F)" | Out-Null
Write-Host "[+] Wrote restricted node configuration to $ConfigPath (SYSTEM & Administrators only)"

# Write log-rotating unattended runner wrapper
$RunnerContent = @'
$ErrorActionPreference = "Stop"
$ConfigPath = "C:\ProgramData\BeoFold\node-config.json"
$LogDir     = "C:\ProgramData\BeoFold\logs"
$LogFile    = Join-Path $LogDir "fold-node.log"
$ExePath    = "C:\Program Files\BeoFold\fold-node.exe"

if (Test-Path $LogFile) {
    $Item = Get-Item $LogFile
    if ($Item.Length -gt 20MB) {
        Move-Item -Path $LogFile -Destination "$LogFile.1" -Force
    }
}

$Cfg = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
$CliArgs = New-Object System.Collections.Generic.List[string]
if ($Cfg.DisableGpu) {
    $CliArgs.Add("--no-gpu")
}
$CliArgs.AddRange([string[]]@(
    "mesh",
    "--data-dir", $Cfg.DataDir,
    "--bind", $Cfg.BindAddress,
    "--cluster-secret", $Cfg.ClusterSecret,
    "--cluster-name", $Cfg.ClusterName,
    "--label", $Cfg.NodeLabel,
    "--max-cpu", ([string]$Cfg.MaxCpuPercent)
))
if (-not [string]::IsNullOrWhiteSpace($Cfg.SyncUrl)) {
    $CliArgs.Add("--sync-url")
    $CliArgs.Add($Cfg.SyncUrl)
}
if ($Cfg.AutoThrottle) {
    $CliArgs.Add("--auto-throttle")
}
if ($Cfg.InitTrpCage) {
    $CliArgs.Add("--init-trpcage")
}

& $ExePath @CliArgs *>> $LogFile
exit $LASTEXITCODE
'@
Set-Content -Path $RunnerPs1 -Value $RunnerContent -Encoding UTF8

# Configure scoped Windows Defender Firewall rules (LocalSubnet UDP 7443 QUIC + UDP 5353 mDNS)
$Profiles = if ($AllowPublicProfileSubnet) { @("Domain", "Private", "Public") } else { @("Domain", "Private") }

foreach ($RuleSpec in @(
    @{ Name = "BeoFold-QUIC-LAN"; Display = "BeoFold QUIC TLS 1.3 LAN Mesh (UDP $QuicPort)"; Port = $QuicPort },
    @{ Name = "BeoFold-mDNS-LAN"; Display = "BeoFold mDNS Peer Discovery (UDP 5353)";        Port = [uint16]5353 }
)) {
    Get-NetFirewallRule -Name $RuleSpec.Name -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue
    New-NetFirewallRule `
        -Name $RuleSpec.Name `
        -DisplayName $RuleSpec.Display `
        -Direction Inbound `
        -Protocol UDP `
        -LocalPort $RuleSpec.Port `
        -RemoteAddress LocalSubnet `
        -Profile $Profiles `
        -Program $TargetExe `
        -Action Allow | Out-Null
    Write-Host "[+] Configured Windows Defender Firewall rule '$($RuleSpec.Name)' (LocalSubnet)"
}

$TaskName = "BeoFold-Node"
if ($Mode -eq "ScheduledTask") {
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$RunnerPs1`"" `
        -WorkingDirectory $NodeVault

    $Trigger = New-ScheduledTaskTrigger -AtStartup
    $Principal = New-ScheduledTaskPrincipal `
        -UserId "NT AUTHORITY\SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest

    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Principal $Principal `
        -Settings $Settings `
        -Description "BeoFold unattended Beowulf LAN & multi-cluster protein folding node" `
        -Force | Out-Null

    Start-ScheduledTask -TaskName $TaskName
    Write-Host "[+] Registered and started unattended boot task '$TaskName' (Logs: $LogDir\fold-node.log)"
}
else {
    $NssmCmd = Get-Command $NssmPath -ErrorAction SilentlyContinue
    if (-not $NssmCmd) {
        throw "nssm.exe not found in PATH (or pass -Mode ScheduledTask for built-in zero-dependency boot registration)."
    }
    & $NssmCmd.Source stop $TaskName 2>$null | Out-Null
    & $NssmCmd.Source remove $TaskName confirm 2>$null | Out-Null
    & $NssmCmd.Source install $TaskName "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$RunnerPs1`""
    & $NssmCmd.Source set $TaskName AppDirectory $NodeVault
    & $NssmCmd.Source set $TaskName AppStopMethodConsole 15000
    & $NssmCmd.Source set $TaskName Start SERVICE_AUTO_START
    & $NssmCmd.Source start $TaskName
    Write-Host "[+] Registered and started Windows Service '$TaskName' via NSSM"
}
