# SIM-swap fine-tune of Laya - run everything on your CUDA GPU.
# From PowerShell:   cd D:\safe\finetune\fraud_v2 ;  .\run_all.ps1
# (if scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run_all.ps1)

$ErrorActionPreference = "Stop"
$py = "D:\safe\safe\Scripts\python.exe"
Set-Location $PSScriptRoot

# 1. CUDA check. Your venv currently has torch 2.15.0+cpu (no GPU support).
& $py -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 3)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "No CUDA build of PyTorch in D:\safe\safe. Install one, then re-run this script:" -ForegroundColor Yellow
    Write-Host "  $py -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu132"
    Write-Host "  (check 'nvidia-smi' for your driver; if cu132 is not offered for your driver, pick the matching"
    Write-Host "   CUDA build from https://pytorch.org/get-started/locally/)"
    exit 3
}

# 2. ONNX export dependencies (Laya's own extra)
& $py -m pip install -q "laya[onnx]"

# 3. Base model in the local HF cache (you already loaded it once with laya.load("english"))
& $py -c "import laya.train as T; print('base checkpoint:', T.resolve_checkpoint_dir('convaiinnovations/laya'))"

# 4. Attempt ladder: baseline -> fine-tune -> per-epoch checkpoints -> tipping-point stop -> calibrate
#    Defaults are sized for the RTX 3050 6 GB. If CUDA runs out of memory the rung is retried automatically with a smaller footprint.
& $py run_attempts.py --base convaiinnovations/laya --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 5. Before/after comparison and every hard-test failure
& $py compare.py | Tee-Object -FilePath runs\comparison.txt

# 6. Deployable export (ONNX) + agreement/speed check
& $py export_final.py --device cuda
