$root = ".\docs"

$pattern = 'C:/Users/' + [regex]::Escape($env:USERNAME)
$replacement = 'C:/Users/UserName'

Get-ChildItem $root -Recurse -File -Filter *.html |
ForEach-Object {

    $content = Get-Content $_.FullName -Raw

    $newContent = $content -replace $pattern, $replacement

    if ($newContent -ne $content) {
        Set-Content $_.FullName $newContent -Encoding UTF8
        Write-Host "Updated $($_.FullName)"
    }
}