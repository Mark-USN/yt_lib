Set-Location $PSScriptRoot\..
Remove-Item .\docs -Recurse -Force -ErrorAction SilentlyContinue
uv run pdoc .\src\yt_lib -o .\docs

