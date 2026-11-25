# Resolve the Ollama blob filename for a rerank model.
# Prints "sha256-<hex>" of the GGUF (image.model) layer recorded in the
# Ollama manifest, so re-pulling a model (which changes its digest) does
# not break the pinned path in start-rerank-server.bat.
# Exits 1 without output when no manifest/blob matches; the caller then
# falls back to the pinned sha.
param([string]$Model = "bge-reranker-v2-m3")

$stores = @(
    $env:OLLAMA_MODELS,
    (Join-Path $env:USERPROFILE ".ollama\models"),
    (Join-Path $env:LOCALAPPDATA "Ollama\models")
) | Where-Object { $_ -and (Test-Path $_) }

foreach ($store in $stores) {
    $manifests = Join-Path $store "manifests"
    if (-not (Test-Path $manifests)) { continue }
    $files = Get-ChildItem -Path $manifests -Recurse -Filter "latest" -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -ieq $Model }
    foreach ($f in $files) {
        try {
            $manifest = Get-Content $f.FullName -Raw | ConvertFrom-Json
        } catch { continue }
        $layer = $manifest.layers |
            Where-Object { $_.mediaType -like "*image.model" } |
            Select-Object -First 1
        if (-not $layer) { continue }
        $blob = $layer.digest -replace "^sha256:", "sha256-"
        if (Test-Path (Join-Path $store ("blobs\" + $blob))) {
            Write-Output $blob
            exit 0
        }
    }
}
exit 1
