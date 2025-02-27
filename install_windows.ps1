Write-Output "🔧 Setting up Windows environment..."

# Detect CUDA support
$cudaInstalled = $false
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $cudaInstalled = $true
    Write-Output "✅ CUDA detected. Enabling CUDA support."
    $env:FORCE_CUDA = "1"
}

# Run batch install script
Start-Process -FilePath "cmd.exe" -ArgumentList "/c install_windows.bat" -Wait -NoNewWindow
