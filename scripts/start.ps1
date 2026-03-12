# MusicVision launcher — run from an admin PowerShell
# Caps the 5090 to 450W, starts backend + frontend, restores on exit.

param(
    [int]$GpuIndex = 1,
    [int]$PowerLimit = 450,
    [string]$ProjectDir = ""
)

# Read default power limit before changing it
$defaultPL = (nvidia-smi -i $GpuIndex --query-gpu=power.default_limit --format=csv,noheader,nounits).Trim()
Write-Host "GPU $GpuIndex default power limit: ${defaultPL}W"

nvidia-smi -i $GpuIndex -pl $PowerLimit
Write-Host "GPU $GpuIndex power limit set to ${PowerLimit}W"

# Build the backend command
$serveCmd = "cd ~/musicvision && uv run musicvision serve"
if ($ProjectDir) {
    $serveCmd += " $ProjectDir"
}

try {
    # Start backend (FastAPI on :8000)
    $backend = Start-Process wsl -ArgumentList "-e", "bash", "-c", $serveCmd `
        -PassThru -NoNewWindow

    # Start frontend (Vite on :5173)
    $frontend = Start-Process wsl -ArgumentList "-e", "bash", "-c", "cd ~/musicvision/frontend && npm run dev" `
        -PassThru -NoNewWindow

    Write-Host ""
    Write-Host "Backend:  http://localhost:8000"
    Write-Host "Frontend: http://localhost:5173"
    Write-Host "Press Ctrl+C to stop both and restore power limit."
    Write-Host ""

    # Wait for either process to exit
    while (!$backend.HasExited -and !$frontend.HasExited) {
        Start-Sleep -Milliseconds 500
    }
}
finally {
    # Clean up both processes
    foreach ($proc in @($backend, $frontend)) {
        if ($proc -and !$proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }

    # Restore power limit
    nvidia-smi -i $GpuIndex -pl $defaultPL
    Write-Host "GPU $GpuIndex power limit restored to ${defaultPL}W"
}
