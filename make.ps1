# Windows counterpart of the Makefile (also runs under PowerShell 7 on macOS/Linux).
#   powershell -ExecutionPolicy Bypass -File make.ps1 setup
#   .\make.ps1 help
param(
    [Parameter(Position = 0)] [string] $Target = "help",
    [switch] $Cpu,                  # setup: CPU build of torch (no NVIDIA GPU)
    [string] $Scans = "demo",       # run / run-vr: "all" lists every downloaded patient
    [int] $Port = 8000,
    [string] $Ip = "",              # certs / run-vr: override the LAN address
    [string] $Llm = "qwen2.5:7b"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# Python reads and writes text as UTF-8 regardless of the Windows code page.
$env:PYTHONUTF8 = "1"

$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or $IsWindows
$Py = if ($OnWindows) { Join-Path ".venv" "Scripts\python.exe" } else { Join-Path ".venv" "bin/python" }

function Invoke-Checked([string] $Exe, [string[]] $Arguments) {
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed (exit $LASTEXITCODE)" }
}

function Find-Python {
    # torch, vtk and matplotlib have no wheels for 3.15 yet: 3.12-3.14 only.
    $check = "import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))"
    # Each entry is "command [launcher flag]"; the Windows launcher `py` picks
    # an installed version with -3.x, and python.exe may be the Store stub.
    $candidates = "py -3.14", "py -3.13", "py -3.12", "python3.14", "python3.13", "python3.12", "python3", "python"
    foreach ($spec in $candidates) {
        $exe, $flags = $spec.Split(" ", 2)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $prefix = @()
        if ($flags) { $prefix = @($flags) }
        try { & $exe @prefix -c $check 2>$null } catch { continue }
        if ($LASTEXITCODE -eq 0) { return [pscustomobject]@{ Exe = $exe; Prefix = $prefix } }
    }
    throw "no Python 3.12-3.14 found: .\make.ps1 system-deps (or winget install Python.Python.3.13)"
}

function Ensure-Venv {
    if (Test-Path $Py) { return }
    $python = Find-Python
    Invoke-Checked $python.Exe ($python.Prefix + @("-m", "venv", ".venv"))
    Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
}

function Get-LanIp {
    if ($Ip) { return $Ip }
    # The local address of a UDP "connection" to a public IP: no packet is sent.
    $socket = New-Object System.Net.Sockets.Socket([System.Net.Sockets.AddressFamily]::InterNetwork,
        [System.Net.Sockets.SocketType]::Dgram, [System.Net.Sockets.ProtocolType]::Udp)
    try { $socket.Connect("1.1.1.1", 80); return $socket.LocalEndPoint.Address.ToString() }
    catch { throw "no LAN address found; pass one: .\make.ps1 certs -Ip 192.168.1.20" }
    finally { $socket.Dispose() }
}

switch ($Target) {
    "help" {
        @"
.\make.ps1 system-deps        Python 3.13, mkcert and Ollama with winget (Windows only)
.\make.ps1 setup [-Cpu]       virtualenv in .venv + pinned dependencies (-Cpu: no NVIDIA GPU)
.\make.ps1 scans              download the 7 demo patients from Zenodo (~300 MB)
.\make.ps1 scans-candidates   the 24 patients the demo set was picked from (~1 GB)
.\make.ps1 scans-all          all 100 scans, for retraining (~3 GB)
.\make.ps1 llm                pull the local LLM ($Llm) for free-form voice commands
.\make.ps1 run [-Scans all]   viewer on http://127.0.0.1:$Port
.\make.ps1 certs              HTTPS certificate for this machine's LAN address (mkcert)
.\make.ps1 run-vr             viewer over HTTPS for a headset on the same network
.\make.ps1 test               the tests (~30 s, synthetic scans, no download needed)
"@ | Write-Host
    }
    "system-deps" {
        if (-not $OnWindows) { throw "on macOS/Linux use: make system-deps" }
        foreach ($id in @("Python.Python.3.13", "FiloSottile.mkcert", "Ollama.Ollama")) {
            Invoke-Checked "winget" @("install", "--id", $id, "--exact", "--accept-source-agreements", "--accept-package-agreements")
        }
        Write-Host "open a new terminal so the new commands are on PATH, then: .\make.ps1 setup"
    }
    "setup" {
        Ensure-Venv
        if ($Cpu) {
            $torch = (Select-String -Path requirements.txt -Pattern "^torch==(.+)$").Matches[0].Groups[1].Value
            Invoke-Checked $Py @("-m", "pip", "install", "torch==$torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        }
        Invoke-Checked $Py @("-m", "pip", "install", "-r", "requirements.txt")
        Write-Host "`nnext: .\make.ps1 scans; .\make.ps1 run"
    }
    "scans" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--demo", "--splits") }
    "scans-candidates" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--candidates", "--splits") }
    "scans-all" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--splits", "all") }
    "llm" {
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama is not installed: https://ollama.com/download" }
        Invoke-Checked "ollama" @("pull", $Llm)
    }
    "run" {
        Ensure-Venv
        $env:SCANS = $Scans
        Invoke-Checked $Py @("server.py", "--port", "$Port")
    }
    "certs" {
        if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) { throw "mkcert is not installed: .\make.ps1 system-deps" }
        $lan = Get-LanIp
        New-Item -ItemType Directory -Force certs | Out-Null
        Invoke-Checked "mkcert" @("-cert-file", "certs/lan.pem", "-key-file", "certs/lan-key.pem", $lan, "localhost", "127.0.0.1")
    }
    "run-vr" {
        Ensure-Venv
        if (-not (Test-Path "certs/lan.pem")) { throw "no certificate yet: .\make.ps1 certs" }
        Write-Host "headset browser: https://$(Get-LanIp):$Port/"
        $env:SCANS = $Scans
        Invoke-Checked $Py @("server.py", "--host", "0.0.0.0", "--port", "$Port",
            "--ssl-certfile", "certs/lan.pem", "--ssl-keyfile", "certs/lan-key.pem")
    }
    "test" {
        Ensure-Venv
        Invoke-Checked $Py @("-m", "pytest", "-q")
    }
    default { throw "unknown target '$Target': .\make.ps1 help" }
}
