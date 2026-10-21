# Put this computer's SSH public key on a server, test the key login, and add a short alias to ~/.ssh/config.
# Windows PowerShell version (uses the OpenSSH that comes with Windows, and the key in %USERPROFILE%\.ssh).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\ssh_setup.ps1 -HostName connect.bjb1.seetacloud.com -Port 36315 -Alias gpu2
#   (-User defaults to root, -Key defaults to $env:USERPROFILE\.ssh\id_rsa)
#
# The server password is typed into ssh itself (once, when the key is installed); it is never stored or logged.
param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$User = "root",
    [string]$Alias = "",
    [string]$Key = "$env:USERPROFILE\.ssh\id_rsa"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Key) -or -not (Test-Path "$Key.pub")) {
    Write-Host "No key pair at $Key(.pub). Create one with: ssh-keygen -t ed25519"
    exit 1
}
$target = "$User@$HostName"
$common = @("-p", $Port, "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new")

Write-Host "== 1/3 does key login already work?"
& ssh @common -i $Key -o BatchMode=yes -o PasswordAuthentication=no $target true 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "yes, nothing to install"
} else {
    Write-Host "no: installing $Key.pub (type the server password when asked)"
    $remote = 'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; ' +
              'read -r k; grep -qxF "$k" ~/.ssh/authorized_keys || echo "$k" >> ~/.ssh/authorized_keys'
    $pub = (Get-Content "$Key.pub" -Raw).Trim()
    $pub | & ssh @common -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no $target $remote
    if ($LASTEXITCODE -ne 0) { Write-Host "installing the key failed"; exit 1 }
}

Write-Host "== 2/3 test the key login"
& ssh @common -i $Key -o BatchMode=yes -o PasswordAuthentication=no $target `
    'echo "login OK on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"; df -h / | tail -n 1'
if ($LASTEXITCODE -ne 0) { Write-Host "key login does not work yet"; exit 1 }

if ($Alias) {
    Write-Host "== 3/3 ssh alias '$Alias'"
    $cfgDir = "$env:USERPROFILE\.ssh"; $cfg = "$cfgDir\config"
    if (-not (Test-Path $cfgDir)) { New-Item -ItemType Directory $cfgDir | Out-Null }
    if (-not (Test-Path $cfg)) { New-Item -ItemType File $cfg | Out-Null }
    if (Select-String -Path $cfg -Pattern "^Host\s+$([regex]::Escape($Alias))(\s|$)" -Quiet) {
        Write-Host "alias already in $cfg, left as it is"
    } else {
        $block = "`r`nHost $Alias`r`n  HostName $HostName`r`n  Port $Port`r`n  User $User`r`n  IdentityFile $Key`r`n  ServerAliveInterval 30`r`n"
        Add-Content -Path $cfg -Value $block -Encoding ascii
        Write-Host "added; connect with: ssh $Alias"
    }
} else {
    Write-Host "== 3/3 no alias requested; connect with: ssh -p $Port $target"
}
