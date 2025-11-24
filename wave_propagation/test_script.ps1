$ErrorActionPreference = "Stop"

# 1. Compile
Write-Host "Compiling..." -ForegroundColor Cyan
g++ -O3 -fopenmp -std=c++17 main.cpp Network.cpp WavePropagator.cpp Benchmark.cpp -o wave_propagation.exe
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation failed!"
}
Write-Host "Compilation successful." -ForegroundColor Green

# Function to run a test case
function Run-Test {
    param (
        [string]$Name,
        [string]$Args
    )
    Write-Host "Running Test: $Name" -ForegroundColor Yellow
    $cmd = "./wave_propagation.exe $Args"
    Write-Host "Command: $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Test '$Name' failed!"
    }
    Write-Host "Test '$Name' passed." -ForegroundColor Green
    Write-Host "----------------------------------------"
}

# 2. Run Tests

# Test 1: 1D Basic
Run-Test -Name "1D Basic" -Args "--network 1d --N 100 --steps 100 --dump-frames --frame-every 10"

# Test 2: 2D Basic
Run-Test -Name "2D Basic" -Args "--network 2d --Lx 50 --Ly 50 --steps 50 --dump-frames --frame-every 10"

# Test 3: Noise PerNode (Bonus)
Run-Test -Name "Noise PerNode" -Args "--network 1d --N 100 --steps 50 --noise pernode --omega-mu 1.0 --omega-sigma 0.1 --S0 1.0"

# Test 4: Noise Single (Bonus)
Run-Test -Name "Noise Single" -Args "--network 2d --Lx 30 --Ly 30 --steps 50 --noise single --noise-node 450 --omega-mu 1.0 --omega-sigma 0.1 --S0 1.0"

# Test 5: OpenMP Threads
$env:OMP_NUM_THREADS = "4"
Run-Test -Name "OpenMP 4 Threads" -Args "--network 2d --Lx 100 --Ly 100 --steps 20 --threads 4"

Write-Host "All tests completed successfully!" -ForegroundColor Green
