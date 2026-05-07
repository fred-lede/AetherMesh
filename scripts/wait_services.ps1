# Wait for services using env-configured ports when available
$aiihHost = if ($env:AIIH_HOST -and $env:AIIH_HOST -ne '') { $env:AIIH_HOST } else { '127.0.0.1' }
# For health checks, avoid using 0.0.0.0 or :: as target — use loopback instead
$checkHost = if ($aiihHost -in @('0.0.0.0','::')) { '127.0.0.1' } else { $aiihHost }
$controlPort   = if ($env:AIIH_CONTROL_PORT -and $env:AIIH_CONTROL_PORT -ne '') { $env:AIIH_CONTROL_PORT } else { 8000 }
$routerPort    = if ($env:AIIH_ROUTER_PORT -and $env:AIIH_ROUTER_PORT -ne '') { $env:AIIH_ROUTER_PORT } else { 8001 }
$metricsPort   = if ($env:AIIH_METRICS_PORT -and $env:AIIH_METRICS_PORT -ne '') { $env:AIIH_METRICS_PORT } else { 8002 }
$dashboardPort = if ($env:AIIH_DASHBOARD_PORT -and $env:AIIH_DASHBOARD_PORT -ne '') { $env:AIIH_DASHBOARD_PORT } else { 8003 }

$targets = @(
    @{ name = 'control-plane'; url = ("http://{0}:{1}/health" -f $checkHost, $controlPort) }
    @{ name = 'router';        url = ("http://{0}:{1}/health" -f $checkHost, $routerPort) }
    @{ name = 'metrics';       url = ("http://{0}:{1}/health" -f $checkHost, $metricsPort) }
    @{ name = 'dashboard';     url = ("http://{0}:{1}/health" -f $checkHost, $dashboardPort) }
)

Write-Host 'Waiting for service health (up to 90s)...'

$deadline = (Get-Date).AddSeconds(90)
$healthy = $false

while ((Get-Date) -lt $deadline) {
    $allOk = $true

    foreach ($t in $targets) {
        try {
            $r = Invoke-WebRequest -Uri $t.url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -lt 200 -or $r.StatusCode -ge 300) {
                $allOk = $false
            }
        }
        catch {
            $allOk = $false
        }
    }

    if ($allOk) {
        $healthy = $true
        break
    }

    Start-Sleep -Seconds 3
}

if ($healthy) {
    Write-Host 'All services healthy: control-plane/router/metrics/dashboard'
}
else {
    Write-Host 'Timed out waiting for service health. Check logs/windows for failed service startup.'

    foreach ($t in $targets) {
        try {
            $r = Invoke-WebRequest -Uri $t.url -UseBasicParsing -TimeoutSec 5
            Write-Host ('[OK] ' + $t.name + ' ' + $t.url + ' -> HTTP ' + $r.StatusCode)
        }
        catch {
            Write-Host ('[FAIL] ' + $t.name + ' ' + $t.url + ' -> ' + $_.Exception.Message)
        }
    }

    exit 1
}