# SSH setup: laptop -> desktop

Lets Claude Code (running on the laptop) dispatch commands directly to the
desktop's terminal -- for running training/CPT jobs on whichever machine has
the right hardware for the job, without manually copy-pasting between them.

## 1. On the desktop, as Administrator: enable OpenSSH Server

```powershell
# Install OpenSSH Server (if not already present)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start it now and make it start automatically on boot
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Confirm the firewall rule exists (Windows usually creates this automatically)
Get-NetFirewallRule -Name *ssh* | Select-Object Name, Enabled, Direction, Action

# Get the IP to connect to
ipconfig | Select-String "IPv4"
```

## 2. On the desktop, as Administrator: install the laptop's public key

Password-based SSH won't work for automated use (the environment driving
these commands is non-interactive and can't respond to a password prompt),
so key-based auth is required.

```powershell
$pubkey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPfrBYs9AXyt9wA211UfKrTjAc3wI6Y1IZ/VBoYtl/XC claude-code-laptop-to-desktop"

# If your desktop account IS an administrator (most common on a personal PC):
New-Item -ItemType Directory -Force -Path "C:\ProgramData\ssh" | Out-Null
Add-Content -Path "C:\ProgramData\ssh\administrators_authorized_keys" -Value $pubkey
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /grant "SYSTEM:F" "Administrators:F"

# If your desktop account is NOT an administrator, use this instead:
# New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh" | Out-Null
# Add-Content -Path "$env:USERPROFILE\.ssh\authorized_keys" -Value $pubkey

# Restart sshd so it picks up the key
Restart-Service sshd
```

Windows OpenSSH quirk: an Administrator account uses
`C:\ProgramData\ssh\administrators_authorized_keys` instead of the normal
per-user `~\.ssh\authorized_keys` -- use whichever block above matches the
desktop account.

## 3. Back on the laptop

Once the desktop's IP and Windows username are known, a connection test:

```bash
ssh <username>@<desktop-ip> "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv"
```

First connection will prompt to accept the host key (only once). After that,
commands can be dispatched directly, e.g.:

```bash
ssh <username>@<desktop-ip> "cd D:\path\to\project && python pc_benchmark.py"
```

## Open question: how the two machines share project files

Two options once SSH is live:

- **Git-sync** (recommended): the desktop clones the same GitHub repo(s) this
  project uses; work stays version-controlled and both machines stay in sync
  through normal git pull/push, same as this repo already does.
- **Fresh/untracked copy**: work directly on the desktop's own drive without
  syncing back through git -- simpler short-term, easier to end up out of
  sync long-term.
