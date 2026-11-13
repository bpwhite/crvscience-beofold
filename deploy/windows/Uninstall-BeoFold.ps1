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
