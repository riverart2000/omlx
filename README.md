# oMLX customizations backup

Version-controlled backup of my local oMLX + TTS Studio customizations, so I can
revert to any previous state. Pushed to `riverart2000/omlx` on the
**`customizations`** branch (independent history from the oMLX `main` mirror).

## What's tracked
See `files.txt` (LIVE_PATH|REPO_PATH). Currently:

| Live location | What it is |
|---|---|
| `~/mlx-audio/tts_server.py`, `tts-studio.sh`, `tts_config.json`, `text_to_audio.py`, `tools/ci_enhance{,.swift}` | TTS Studio server + Core Image enhancer |
| `/Applications/oMLX.app/.../admin/templates/chat.html` | Customized oMLX chat UI (lives inside the app bundle — wiped on app update!) |
| `~/.omlx/settings.json`, `~/.omlx/model_settings.json`, `~/.config/omlx/mcp.json` | Config |

## Save the current state to GitHub
```bash
cd ~/omlx-backup
./backup.sh "describe what changed"
```

## Revert to an earlier state
```bash
cd ~/omlx-backup
git log --oneline                 # pick the version you want
git checkout <commit> -- .        # load it into the working tree
./restore.sh                      # write those files back to the live locations
git checkout customizations -- .  # (optional) return working tree to latest
```
Then restart the services:
```bash
bash ~/mlx-audio/tts-studio.sh restart   # TTS Studio
# restart the oMLX app to reload chat.html
```

## Add a new file to the backup
Append a `LIVE_PATH|REPO_PATH` line to `files.txt`, then run `./backup.sh`.

## Secrets
This repo is **public**, so `backup.sh` automatically redacts secret-looking
values (`secret_key`, `api_key`, `*_API_KEY`, tokens, passwords) in
`config/*.json` to `__SECRET_REDACTED__` before committing. `restore.sh` merges
configs back and keeps your real local secret values, so reverting never writes
a placeholder over a working key. Real secrets never leave your Mac.
