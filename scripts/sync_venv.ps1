$VenvPath = ".venv"
$RequirementsFile = "requirements.txt"
$HashFile = "$VenvPath\.requirements.hash"

# Create venv if missing
if (-not (Test-Path $VenvPath)) {
    Write-Host ".venv not found. Creating virtual environment..."
    python -m venv $VenvPath
}

# Activate venv
& "$VenvPath\Scripts\Activate.ps1"

# No requirements file → nothing to sync
if (-not (Test-Path $RequirementsFile)) {
    Write-Host "requirements.txt not found. Skipping dependency sync."
    return
}

# Compute current requirements hash
$CurrentHash = (Get-FileHash $RequirementsFile -Algorithm SHA256).Hash

# Load previous hash if it exists
$PreviousHash = if (Test-Path $HashFile) {
    Get-Content $HashFile
} else {
    ""
}

# Sync only if requirements changed
if ($CurrentHash -ne $PreviousHash) {
    Write-Host "requirements.txt changed. Updating virtual environment..."

    python -m pip install --upgrade pip --quiet
    python -m pip install -r $RequirementsFile
    $InstallExitCode = $LASTEXITCODE

    if ($InstallExitCode -eq 0) {
        $CurrentHash | Out-File $HashFile -Encoding ascii
        Write-Host "Virtual environment updated."
    }
    else {
        Write-Error "Dependency installation failed. Hash not updated."
        exit $InstallExitCode
    }
}
else {
    Write-Host "Virtual environment already up to date."
}
