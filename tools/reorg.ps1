<#
.SYNOPSIS
  Reorganise the Diffusion_as_Memory repo: historic artefacts -> reference/, live code -> root.

.DESCRIPTION
  Idempotent. Run with -DryRun first to see the plan. Moves are plain Move-Item; git detects
  the renames on `git add -A` (several targets are gitignored, so `git mv` would fail on them).

.EXAMPLE
  pwsh tools/reorg.ps1 -DryRun
  pwsh tools/reorg.ps1
#>
[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "Repo root: $Root" -ForegroundColor Cyan
if ($DryRun) { Write-Host "DRY RUN - nothing will be moved`n" -ForegroundColor Yellow }

# Destination folders that must exist before any move. NB: 'training' is deliberately absent --
# it is created by moving scripts/training, and pre-creating it would make that move a no-op.
$dirs = @(
    'reference', 'reference/runs', 'reference/scripts', 'reference/baseline_ldlm',
    'reference/dataset_prep', 'reference/docs',
    'configs', 'tests', 'tools', 'ablations', 'baselines', 'data_pipeline'
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        if ($DryRun) { Write-Host "  mkdir  $d" } else { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    }
}

# source -> destination *parent*. Source may be a dir, a file, or a glob.
$moves = [ordered]@{
    # --- run artefacts -------------------------------------------------------
    'output'                      = 'reference/runs'
    'saved_diffusion_outputs'     = 'reference/runs'
    'saved_latent_models_outputs' = 'reference/runs'
    'wandb'                       = 'reference/runs'
    'slurm-*.out'                 = 'reference/runs'
    # --- launch scripts ------------------------------------------------------
    'scripts/autoencoder'         = 'reference/scripts'
    'scripts/diffusion'           = 'reference/scripts'
    'scripts/inference'           = 'reference/scripts'
    'scripts/sbatch'              = 'reference/scripts'
    'scripts/archive'             = 'reference/scripts'
    'scripts/FINAL_commands_to_run.sh' = 'reference/scripts'
    'sbatch_scripts'              = 'reference/scripts'
    'run_pipeline.sh'             = 'reference/scripts'
    # --- the Lovelace latent-diffusion fork we started from ------------------
    'diffusion'                   = 'reference/baseline_ldlm'
    'latent_models'               = 'reference/baseline_ldlm'
    'model'                       = 'reference/baseline_ldlm'
    'dataset_utils'               = 'reference/baseline_ldlm'
    'datasets_1'                  = 'reference/baseline_ldlm'
    'train_latent_model.py'       = 'reference/baseline_ldlm'
    'train_text_diffusion.py'     = 'reference/baseline_ldlm'
    'inference_forget_recall.py'  = 'reference/baseline_ldlm'
    'CONSTANTS.py'                = 'reference/baseline_ldlm'
    'generate_xt.py'              = 'reference/baseline_ldlm'
    # --- data provenance -----------------------------------------------------
    'dataset_prep'                = 'reference/dataset_prep'
    'dataset_generation'          = 'reference/dataset_prep'
    'privasis_gpt4o_abstraction.json'   = 'reference/dataset_prep'
    'privasis_privacy_abstraction.csv'  = 'reference/dataset_prep'
    # --- prior write-ups -----------------------------------------------------
    'Analyzing Latent Diffusion Architectures.md' = 'reference/docs'
    'codebase_walkthrough.md'                     = 'reference/docs'
    'implementation_plan.md'                      = 'reference/docs'
    'Running Privasis Diffusion Baseline.md'      = 'reference/docs'
    'Final_Report___698DS__Bala_s_Copy_.pdf'      = 'reference/docs'
    'figures'                                     = 'reference/docs'
    # --- promote live training code to root ----------------------------------
    'scripts/training'            = '.'
}

$moved = 0; $skipped = 0
foreach ($src in $moves.Keys) {
    $dstParent = $moves[$src]
    $matches = @(Get-Item -Path $src -ErrorAction SilentlyContinue)
    if ($matches.Count -eq 0) { $skipped++; continue }

    foreach ($m in $matches) {
        $leaf = $m.Name
        $dst  = if ($dstParent -eq '.') { $leaf } else { Join-Path $dstParent $leaf }
        if (Test-Path $dst) {
            Write-Host "  SKIP   $leaf (already at $dst)" -ForegroundColor DarkGray
            $skipped++
            continue
        }
        Write-Host "  move   $($m.FullName.Replace("$Root\",'')) -> $dst"
        if (-not $DryRun) { Move-Item -LiteralPath $m.FullName -Destination $dst -Force }
        $moved++
    }
}

# scripts/training -> training  (rename after the move above lands it at ./training)
if ((Test-Path 'training') -and -not (Test-Path 'training/__init__.py') -and -not $DryRun) {
    New-Item -ItemType File -Path 'training/__init__.py' -Force | Out-Null
}

# scripts/ should now be empty
if ((Test-Path 'scripts') -and -not $DryRun) {
    $left = @(Get-ChildItem 'scripts' -Recurse -File)
    if ($left.Count -eq 0) { Remove-Item 'scripts' -Recurse -Force; Write-Host "  rmdir  scripts (empty)" }
    else { Write-Host "  KEEP   scripts/ ($($left.Count) files left)" -ForegroundColor Yellow }
}

Write-Host "`nmoved=$moved skipped=$skipped" -ForegroundColor Green
if ($DryRun) { Write-Host "Re-run without -DryRun to apply." -ForegroundColor Yellow }
