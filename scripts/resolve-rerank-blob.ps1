# Resolve the full path of the Ollama GGUF blob for a rerank model.
# Prints the absolute blob path of the GGUF (image.model) layer recorded in
# the Ollama manifest, so re-pulling a model (which changes its digest) does
# not break the pinned path in start-rerank-server.bat.
# Exits 1 without output when no manifest/blob matches; the caller then
# falls back to the pinned sha.
param([string]$Model = "bge-reranker-v2-m3")

function StoreCandidates {
    $list = @(
        $env:OLLAMA_MODELS,
        (Join-Path $env:USERPROFILE ".ollama\models"),
        (Join-Path $env:LOCALAPPDATA "Ollama\models")
    )

    # A scheduled task created with /RU SYSTEM runs with USERPROFILE set to
    # C:\WINDOWS\system32\config\systemprofile, where Ollama has never written
    # a model store. Scan the real user profiles in that case.
    if ($env:USERPROFILE -like "*systemprofile*" -or -not (Test-Path (Join-Path $env:USERPROFILE ".ollama\models"))) {
        foreach ($profile in Get-ChildItem "C:\Users" -Directory -ErrorAction SilentlyContinue) {
            $list += (Join-Path $profile.FullName ".ollama\models")
            $list += (Join-Path $profile.FullName "AppData\Local\Ollama\models")
        }
    }

    return $list | Where-Object { $_ -and (Test-Path $_) }
}

foreach ($store in StoreCandidates) {
    foreach ($manifestRoot in @("manifests", "manifests-v2")) {
        $manifests = Join-Path $store $manifestRoot
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
            $path = Join-Path $store ("blobs\" + $blob)
            if (Test-Path $path) {
                Write-Output $path
                exit 0
            }
        }
    }
}
exit 1
