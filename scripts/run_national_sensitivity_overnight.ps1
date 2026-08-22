param(
    [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv-win\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project Python environment not found: $pythonExe"
}
$logRoot = Join-Path $projectRoot 'logs\national_sensitivity_overnight'
$statusRoot = Join-Path $projectRoot 'results\national_sensitivity_overnight'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path $statusRoot -Force | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$masterLog = Join-Path $logRoot "run_$runStamp.log"
$statusPath = Join-Path $statusRoot 'status.json'

function Write-RunLog {
    param([string]$Message)
    $line = "[$((Get-Date).ToString('s'))] $Message"
    $line | Tee-Object -FilePath $masterLog -Append | Out-Host
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$Marker
    )
    if (Test-Path -LiteralPath $Marker) {
        Write-RunLog "SKIP $Name; marker exists: $Marker"
        return
    }
    Write-RunLog "START $Name"
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $pythonExe @Arguments 2>&1 | ForEach-Object { "$_" } |
            Tee-Object -FilePath $masterLog -Append | Out-Host
        $stepExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($stepExit -ne 0) {
        throw "$Name failed with exit code $stepExit"
    }
    if (-not (Test-Path -LiteralPath $Marker)) {
        throw "$Name completed without marker: $Marker"
    }
    Write-RunLog "DONE $Name"
}

function Invoke-NetworkPipeline {
    param(
        [string]$Name,
        [string]$Config,
        [string]$PanelRoot,
        [string]$DiagnosticRoot,
        [string]$RoughnessRoot,
        [string]$ValidationRoot,
        [string]$Label,
        [string]$ObservationLimit
    )
    $result = [ordered]@{
        network = $Name
        started = (Get-Date).ToString('o')
        status = 'running'
        failed_step = $null
        error = $null
    }
    try {
        Invoke-PythonStep -Name "$Name model/reference extraction" -Arguments @(
            'scripts\45_extract_weather5k_weatherbench2.py',
            '--config', $Config
        ) -Marker (Join-Path $PanelRoot 'manifest.json')

        Invoke-PythonStep -Name "$Name panel diagnostics" -Arguments @(
            'scripts\46_diagnose_weather5k_panel.py',
            '--panel', (Join-Path $PanelRoot 'panel.parquet'),
            '--output', $DiagnosticRoot,
            '--analysis-role', 'national_network_sensitivity_unknown_assimilation_route_not_confirmatory',
            '--observation-limit', $ObservationLimit,
            '--force'
        ) -Marker (Join-Path $DiagnosticRoot 'diagnostics.json')

        Invoke-PythonStep -Name "$Name model-cell-day roughness" -Arguments @(
            'scripts\47_materialise_global_roughness.py',
            '--panel', (Join-Path $PanelRoot 'panel.parquet'),
            '--sites', (Join-Path $DiagnosticRoot 'sites.csv'),
            '--output', $RoughnessRoot,
            '--model-config', $Config,
            '--analysis-role', 'national_network_sensitivity_unknown_assimilation_route_not_confirmatory',
            '--interpretation-warning', $ObservationLimit,
            '--workers', "$Workers"
        ) -Marker (Join-Path $RoughnessRoot 'manifest.json')

        Invoke-PythonStep -Name "$Name national sensitivity analysis" -Arguments @(
            'scripts\50_analyse_weather5k_global_validation.py',
            '--errors', (Join-Path $DiagnosticRoot 'cell_day_errors.parquet'),
            '--roughness', (Join-Path $RoughnessRoot 'roughness.parquet'),
            '--sites', (Join-Path $DiagnosticRoot 'sites.csv'),
            '--output', $ValidationRoot,
            '--errors-manifest', (Join-Path $DiagnosticRoot 'diagnostics.json'),
            '--roughness-manifest', (Join-Path $RoughnessRoot 'manifest.json'),
            '--sites-manifest', (Join-Path $DiagnosticRoot 'diagnostics.json'),
            '--analysis-label', $Label,
            '--analysis-role', 'national-network sensitivity; not independent global confirmation',
            '--observation-limit', $ObservationLimit,
            '--force'
        ) -Marker (Join-Path $ValidationRoot 'summary.json')

        $result.status = 'completed'
    }
    catch {
        $result.status = 'failed'
        $result.error = $_.Exception.Message
        Write-RunLog "FAILED ${Name}: $($result.error)"
    }
    $result.finished = (Get-Date).ToString('o')
    return [pscustomobject]$result
}

Set-Location $projectRoot
Write-RunLog "Overnight national sensitivity run started with $Workers workers"
$results = @()
$results += Invoke-NetworkPipeline `
    -Name 'midas_unknown' `
    -Config 'config\midas_unknown_weatherbench2_2020.yaml' `
    -PanelRoot 'data\interim\midas_unknown_weatherbench2_2020' `
    -DiagnosticRoot 'results\midas_unknown_weatherbench2_2020' `
    -RoughnessRoot 'data\interim\midas_unknown_roughness_2020' `
    -ValidationRoot 'results\midas_unknown_validation_2020' `
    -Label 'MIDAS-UK 2020 · sensibilidad nacional del mecanismo AVAMET' `
    -ObservationLimit 'No se detectó solapamiento Weather5K/ISD, pero la ausencia de toda ruta de asimilación ERA5 no está documentada.'

$results += Invoke-NetworkPipeline `
    -Name 'inmet_unknown' `
    -Config 'config\inmet_unknown_weatherbench2_2020.yaml' `
    -PanelRoot 'data\interim\inmet_unknown_weatherbench2_2020' `
    -DiagnosticRoot 'results\inmet_unknown_weatherbench2_2020' `
    -RoughnessRoot 'data\interim\inmet_unknown_roughness_2020' `
    -ValidationRoot 'results\inmet_unknown_validation_2020' `
    -Label 'INMET-Brasil 2020 · sensibilidad nacional del mecanismo AVAMET' `
    -ObservationLimit 'No se detectó solapamiento Weather5K/ISD, pero la ruta de difusión y asimilación ERA5 de INMET sigue sin documentar.'

$summary = [ordered]@{
    started_from = $projectRoot
    log = $masterLog
    workers = $Workers
    python = $pythonExe
    finished = (Get-Date).ToString('o')
    networks = $results
    all_completed = -not [bool]($results | Where-Object { $_.status -ne 'completed' })
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8
Write-RunLog "Overnight run finished; all_completed=$($summary.all_completed)"
if (-not $summary.all_completed) {
    exit 1
}
