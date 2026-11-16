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
    Stops and unregisters BeoFold unattended Windows boot tasks/services and firewall rules.
    Preserves `C:\ProgramData\BeoFold` trajectory vaults unless `-PurgeData` is specified.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [switch]$PurgeData,

    [Parameter(Mandatory = $false)]
    [string]$NssmPath = "nssm.exe"
)

$ErrorActionPreference = "Stop"

foreach ($TaskName in @("BeoFold-Node", "BeoFold-Sync-Server")) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Task) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[-] Stopped and unregistered Scheduled Task '$TaskName'"
    }
}

$NssmCmd = Get-Command $NssmPath -ErrorAction SilentlyContinue
if ($NssmCmd) {
    foreach ($SvcName in @("BeoFold-Node", "BeoFold-Sync-Server")) {
        if (Get-Service -Name $SvcName -ErrorAction SilentlyContinue) {
            & $NssmCmd.Source stop $SvcName 2>$null | Out-Null
            & $NssmCmd.Source remove $SvcName confirm 2>$null | Out-Null
            Write-Host "[-] Removed NSSM Windows Service '$SvcName'"
        }
    }
}

foreach ($ProcName in @("fold-node", "fold-sync")) {
    Get-Process -Name $ProcName -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

foreach ($RuleName in @("BeoFold-QUIC-LAN", "BeoFold-mDNS-LAN")) {
    Get-NetFirewallRule -Name $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue
}
Write-Host "[-] Removed BeoFold Windows Defender Firewall rules"

if (Test-Path "C:\Program Files\BeoFold") {
    Remove-Item -Path "C:\Program Files\BeoFold" -Recurse -Force
}

if ($PurgeData -and (Test-Path "C:\ProgramData\BeoFold")) {
    Remove-Item -Path "C:\ProgramData\BeoFold" -Recurse -Force
    Write-Host "[-] Purged C:\ProgramData\BeoFold trajectory vaults and logs"
}
else {
    Write-Host "[=] Preserved C:\ProgramData\BeoFold trajectory vaults (pass -PurgeData to delete)"
}
