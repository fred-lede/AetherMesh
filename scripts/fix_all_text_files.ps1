param(
    [string]$Root = "."
)

$extensions = @(
    ".sh", ".bash", ".zsh",
    ".py",
    ".yaml", ".yml",
    ".json", ".md", ".txt",
    ".toml", ".ini", ".conf"
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

Get-ChildItem -Path $Root -Recurse -File | Where-Object {
    $extensions -contains $_.Extension.ToLower()
} | ForEach-Object {
    $path = $_.FullName
    try {
        $content = [System.IO.File]::ReadAllText($path)

        $content = $content -replace "`r`n", "`n"
        $content = $content -replace "`r", "`n"

        [System.IO.File]::WriteAllText($path, $content, $utf8NoBom)
        Write-Host "[FIXED] $path"
    }
    catch {
        Write-Host "[ERROR] $path : $($_.Exception.Message)"
    }
}