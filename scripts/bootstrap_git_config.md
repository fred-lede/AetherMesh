# Git setup for cross-platform development

## Windows
```bash
git config --global core.autocrlf false
git config --global core.eol lf
```

## macOS / Ubuntu
```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

After adding .gitattributes for the first time

Run at repo root:

git add --renormalize .
git status

Then commit the normalization once.



---

# 9. `README` 可加的一段

你可以把下面這段貼到 `README.md`：

```md
## Cross-platform development rules

This repo is developed on Windows and macOS, and tested on Windows, macOS, and Ubuntu.

Please follow these rules:

- Use LF for source/config/script files
- Use CRLF only for `.bat`, `.cmd`, `.ps1`
- Do not save files with UTF-8 BOM
- Prefer platform-specific scripts under `scripts/`
- Use `pathlib` (Python) or `path.join` (Node.js) instead of hardcoded path separators

### One-time setup
```bash
git config --global core.autocrlf false
git config --global core.eol lf


Normalize repo after pulling new rules
git add --renormalize .
Fix text files

macOS / Linux:

./scripts/fix_all_text_files.sh .

Windows:

powershell -ExecutionPolicy Bypass -File .\scripts\fix_all_text_files.ps1