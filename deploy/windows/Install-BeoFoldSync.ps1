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
    Registers BeoFold Web Synchronizing Server (`fold-sync.exe`) on Windows to run unattended on boot
    behind Apache HTTP Server (`Apache24` `mod_proxy_http`).

.EXAMPLE
    .\deploy\windows\Install-BeoFoldSync.ps1 -BindAddress "127.0.0.1:8780" -Quorum 1 -InitTrpCage
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$BinaryPath = ".\target\release\fold-sync.exe",

    [Parameter(Mandatory = $false)]
    [string]$BindAddress = "127.0.0.1:8780",

    [Parameter(Mandatory = $false)]
    [byte]$Quorum = 1,

    [Parameter(Mandatory = $false)]
    [switch]$InitTrpCage
)

$ErrorActionPreference = "Stop"

$ResolvedBin = Resolve-Path -Path $BinaryPath -ErrorAction SilentlyContinue
if (-not $ResolvedBin) {
    throw "fold-sync.exe not found at '$BinaryPath'. Run 'cargo build --release -p fold-sync' first."
}

$InstallDir = "C:\Program Files\BeoFold"
$SyncVault  = "C:\ProgramData\BeoFold\sync-vault"
$BlobsDir   = Join-Path $SyncVault "blobs"
$LogDir     = "C:\ProgramData\BeoFold\logs"
$TargetExe  = Join-Path $InstallDir "fold-sync.exe"
$RunnerPs1  = Join-Path $InstallDir "Start-BeoFoldSyncRunner.ps1"

New-Item -ItemType Directory -Force -Path $InstallDir, $SyncVault, $BlobsDir, $LogDir | Out-Null
Copy-Item -Path $ResolvedBin.Path -Destination $TargetExe -Force

$InitFlag = if ($InitTrpCage) { "`"--init-trpcage`"" } else { "" }
$RunnerContent = @"
`$ErrorActionPreference = "Stop"
`$LogFile = "C:\ProgramData\BeoFold\logs\fold-sync.log"
`$ArgsList = @("serve", "--bind", "$BindAddress", "--data-dir", "$SyncVault", "--quorum", "$Quorum")
if ("$InitFlag" -ne "") {
    `$ArgsList += "--init-trpcage"
}
& "$TargetExe" @ArgsList *>> `$LogFile
exit `$LASTEXITCODE
"@
Set-Content -Path $RunnerPs1 -Value $RunnerContent -Encoding UTF8

$TaskName  = "BeoFold-Sync-Server"
$Action    = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$RunnerPs1`"" `
    -WorkingDirectory $SyncVault
$Trigger   = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal `
    -UserId "NT AUTHORITY\SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest
$Settings  = New-ScheduledTaskSettingsSet `
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
    -Description "BeoFold multi-cluster synchronizing server behind Apache HTTP mod_proxy_http" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "[+] Registered and started '$TaskName' on http://$BindAddress/beofold/api/v1"
Write-Host "[i] Point Apache24 Alias '/beofold/blobs/' to 'C:/ProgramData/BeoFold/sync-vault/blobs/'"
