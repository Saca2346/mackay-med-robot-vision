# Working on the lab server from the laptop, with Claude Code and Codex

Problems so far: the Remote Desktop clipboard does not work (copy / paste between laptop and server fails), training in a
Remote Desktop session is slow and crashes, the laptop copy and the server copy of the labelling project have drifted apart,
and Claude on the laptop cannot run anything on the server (every server step needed `SERVER_JALANKAN.bat` and a password).

The fix is to stop using Remote Desktop for work: use **SSH** for everything, with a key so no password is typed each time.

## Step 1 - SSH key (once, about 5 minutes)

**Laptop, cmd:**
```bat
ssh-keygen -t ed25519 -f %USERPROFILE%\.ssh\id_ed25519 -N ""
type %USERPROFILE%\.ssh\id_ed25519.pub | ssh Zen@140.113.149.94 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
The second command asks the server password one last time. Then add a short name for the server:

**Laptop, cmd:**
```bat
notepad %USERPROFILE%\.ssh\config
```
Paste this, save, close:
```
Host trse-server
    HostName 140.113.149.94
    User Zen
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
```
Test (must print the server name without asking a password):
```bat
ssh trse-server hostname
```

## Step 2 - VS Code on the laptop opens the server folder (Remote - SSH)

1. In VS Code on the laptop install the extension **Remote - SSH** (Microsoft).
2. `F1` -> `Remote-SSH: Connect to Host...` -> `trse-server`.
3. `File > Open Folder` -> `/home/Zen/Downloads/prelabel_size_label_full`.

Now the editor, terminal and file explorer are on the server, but the window is on the laptop: copy / paste works normally,
and no Remote Desktop is needed. Close the Remote Desktop session while training (it uses GPU and RAM on the server).

## Step 3 - training that survives a disconnect

**Server terminal (in VS Code Remote-SSH):**
```bash
tmux new -s train            # a session that keeps running when the laptop disconnects
cd ~/Downloads/prelabel_size_label_full
source ~/trse_venv/bin/activate
python tools/...             # the training command of the current guide
# leave with Ctrl+B then D; come back later with:  tmux attach -t train
```

## Step 4 - Claude Code (this laptop) can then run server steps itself

With the key from step 1, Claude Code on the laptop can run `ssh trse-server "<command>"` and `scp` without a password
prompt, so results no longer need to be copied by hand. Claude still asks before running anything that changes the server.
Optionally, Claude Code can also be used inside the VS Code Remote-SSH window (it then runs on the server).

## Step 5 - one source of truth for data and code

| What | Master copy | How the other side gets it |
| --- | --- | --- |
| Code (`medicine_shelf_vision`, labelling tools) | GitHub (public: code only) | `git pull` on laptop and server |
| Photos, labels, label specs, models, results | **the server** | laptop pulls with `scp` / `rsync` when needed |

Today the copies differ (14 Oct): the server has expiry status 51 done / 112 auto, the laptop 76 done; the server also has
`hasil_tanggal/`, `PANDUAN_LANJUTAN_3KELAS.md` and changed files in `tools/`, `labels_final*`. Before the next training,
decide which copy is right (the server copy is more recent and was reviewed there), then copy the laptop-only work to the
server once and from then on edit labels on the server only.

## Step 6 - Claude Code and Codex together

Both assistants work in the same repositories under shared rules:

* `AGENTS.md` (read by Codex) holds the project rules; `CLAUDE.md` (read by Claude Code) points to it.
* `docs/HANDOFF.md`: each assistant claims a topic before starting and writes what it changed, how it was tested and what is open.
* One git branch per assistant (`claude/<topic>`, `codex/<topic>`); merge after the tests pass and one reviews the other's diff.
* Good split: Codex on a well-defined code task (e.g. a new REF rule with tests), Claude Code on test sessions, data and docs, or
  the other way round, but never both on the same file at the same time.
* Neither assistant gets passwords or tokens; logins happen on the server and in the tools themselves.
