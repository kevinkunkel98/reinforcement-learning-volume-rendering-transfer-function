# Windows counterpart of the Makefile (also runs under PowerShell 7 on macOS/Linux).
#   powershell -ExecutionPolicy Bypass -File make.ps1 setup
#   .\make.ps1 help
param(
    [Parameter(Position = 0)] [string] $Target = "help",
    [switch] $Cpu,                  # setup: CPU build of torch (no NVIDIA GPU)
    [string] $Scans = "all",        # run / run-vr: "demo" lists only the demo patients
    [int] $Port = 8000,
    [string] $Ip = "",              # certs / run-vr: override the LAN address
    [string] $Cert = "public",      # run-vr: "mkcert" uses the .\make.ps1 certs certificate
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

function Wait-Ollama([int] $Seconds) {
    # The same address the app uses (commands.OLLAMA_HOST).
    $url = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://localhost:11434" }
    if ($url -notmatch "^https?://") { $url = "http://$url" }
    for ($i = 0; $i -lt $Seconds; $i++) {
        try { Invoke-WebRequest "$url/api/version" -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true }
        catch { Start-Sleep -Seconds 1 }
    }
    return $false
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
.\make.ps1 scans-eval         the larger held-out splits val_extra + test_extra (~2 GB)
.\make.ps1 llm                install and start Ollama if needed, then pull the local LLM ($Llm)
.\make.ps1 run [-Scans demo]  viewer on http://127.0.0.1:$Port
.\make.ps1 certs              mkcert certificate for this machine's LAN address, for run-vr -Cert mkcert
.\make.ps1 run-vr [-Cert mkcert]  viewer over HTTPS for a headset on the same network
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
        Write-Host "`nnext: .\make.ps1 scans; .\make.ps1 llm; .\make.ps1 run"
    }
    "scans" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--demo", "--splits") }
    "scans-candidates" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--candidates", "--splits") }
    "scans-all" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--splits", "all") }
    "scans-eval" { Ensure-Venv; Invoke-Checked $Py @("-u", "-m", "tools.fetch_scans", "--splits", "val_extra", "test_extra") }
    "llm" {
        # Install Ollama when it is missing, start it when nothing answers, then pull the model.
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            if (-not $OnWindows) { throw "on macOS/Linux use: make llm" }
            Invoke-Checked "winget" @("install", "--id", "Ollama.Ollama", "--exact", "--accept-source-agreements", "--accept-package-agreements")
            # winget does not update this session's PATH: reread it from the registry.
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
            if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama installed but not on PATH yet: open a new terminal and run .\make.ps1 llm again" }
        }
        if (-not (Wait-Ollama 5)) {
            Write-Host "Ollama is not running: starting ollama serve in the background"
            if ($OnWindows) { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden }
            else { Start-Process ollama -ArgumentList "serve" }
            if (-not (Wait-Ollama 30)) { throw "Ollama did not start: run 'ollama serve' to see why" }
        }
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
        $lan = Get-LanIp
        $env:SCANS = $Scans
        if ($Cert -eq "mkcert") {
            if (-not (Test-Path "certs/lan.pem")) { throw "no certificate yet: .\make.ps1 certs" }
            Write-Host "headset browser: https://${lan}:$Port/"
            Invoke-Checked $Py @("server.py", "--host", "0.0.0.0", "--port", "$Port",
                "--ssl-certfile", "certs/lan.pem", "--ssl-keyfile", "certs/lan-key.pem")
            break
        }
        # local-ip.sh's Let's Encrypt certificate for *.local-ip.sh (see the Makefile).
        New-Item -ItemType Directory -Force certs | Out-Null
        try {
            Invoke-WebRequest "https://local-ip.sh/server.pem" -OutFile "certs/public.pem" -UseBasicParsing -TimeoutSec 20
            Invoke-WebRequest "https://local-ip.sh/server.key" -OutFile "certs/public-key.pem" -UseBasicParsing -TimeoutSec 20
        } catch {
            if (-not (Test-Path "certs/public.pem")) { throw "local-ip.sh unreachable and no saved certificate: .\make.ps1 run-vr -Cert mkcert" }
            Write-Host "local-ip.sh unreachable; using the saved certificate"
        }
        Write-Host "headset browser: https://$($lan -replace '\.', '-').local-ip.sh:$Port/"
        Invoke-Checked $Py @("server.py", "--host", "0.0.0.0", "--port", "$Port",
            "--ssl-certfile", "certs/public.pem", "--ssl-keyfile", "certs/public-key.pem")
    }
    "test" {
        Ensure-Venv
        Invoke-Checked $Py @("-m", "pytest", "-q")
    }
    default { throw "unknown target '$Target': .\make.ps1 help" }
}
