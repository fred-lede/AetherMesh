$ErrorActionPreference = "Stop"
$log = "D:\Ai\ai_inference_hub\logs\AIIH-Healthcheck.log"
New-Item -ItemType Directory -Force "D:\Ai\ai_inference_hub\logs" | Out-Null
Add-Content $log "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="

function Test-Endpoint($url) {
  try {
    $null = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 8
    Add-Content $log "[OK] $url"
    return $true
  } catch {
    Add-Content $log "[FAIL] $url -> $($_.Exception.Message)"
    return $false
  }
}

function Get-Workers-HealthSummary {
  try {
    $w = Invoke-RestMethod -Uri "http://127.0.0.1:9200/cluster/workers" -Method Get -TimeoutSec 8
    $items = @($w.workers)
    $total = $items.Count
    $healthy = @($items | Where-Object { "$($_.status)".ToLower() -eq "healthy" }).Count
    Add-Content $log "[INFO] workers healthy=$healthy/$total"

    $degraded = @($items | Where-Object { "$($_.status)".ToLower() -eq "degraded" } | ForEach-Object { $_.worker_id })
    if ($degraded.Count -gt 0) {
      Add-Content $log "[WARN] degraded workers: $($degraded -join ', ')"
    }

    $dead = @($items | Where-Object { "$($_.status)".ToLower() -eq "dead" } | ForEach-Object { $_.worker_id })
    if ($dead.Count -gt 0) {
      Add-Content $log "[WARN] dead workers: $($dead -join ', ')"
    }
  } catch {
    Add-Content $log "[WARN] workers summary unavailable -> $($_.Exception.Message)"
  }
}

function Run-Platform-Restart {
  Add-Content $log "[ACTION] Run task: AIIH-Platform"
  schtasks /Run /TN AIIH-Platform | Out-Null
}

$checks = @(
  "http://127.0.0.1:8001/health",
  "http://127.0.0.1:9200/health",
  "http://127.0.0.1:9100/health",
  "http://127.0.0.1:11434/api/tags",
  "http://127.0.0.1:11435/api/tags"
)

$allOk = $true
foreach ($u in $checks) {
  if (-not (Test-Endpoint $u)) { $allOk = $false }
}

if (-not $allOk) {
  Run-Platform-Restart
  Start-Sleep -Seconds 20
  Add-Content $log "[INFO] Re-check after restart..."
  $allOk2 = $true
  foreach ($u in $checks) {
    if (-not (Test-Endpoint $u)) { $allOk2 = $false }
  }
  if (-not $allOk2) {
    Add-Content $log "[WARN] Still unhealthy after one restart attempt."
  } else {
    Add-Content $log "[OK] Recovered after restart."
    Get-Workers-HealthSummary
  }
} else {
  Add-Content $log "[OK] All services healthy."
  Get-Workers-HealthSummary
}
