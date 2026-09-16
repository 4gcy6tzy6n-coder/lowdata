#!/usr/bin/env python
"""ssh_server.py — minimal paramiko SSH client for the AutoDL server (password auth).

Usage:
    python ssh_server.py "<remote command>"
    python ssh_server.py --label   # print environment info (status check)
"""
from __future__ import annotations

import os
import sys

import paramiko

# Connection details come from the environment; credentials are never hardcoded
# in a tracked file.  Set them before running, e.g.
#   export QN_SSH_HOST=connect.example.seetacloud.com
#   export QN_SSH_PORT=35038
#   export QN_SSH_PASSWORD='...'
HOST = os.environ.get("QN_SSH_HOST", "")
PORT = int(os.environ.get("QN_SSH_PORT", "22"))
USER = os.environ.get("QN_SSH_USER", "root")
PASS = os.environ.get("QN_SSH_PASSWORD", "")

if not HOST or not PASS:
    raise SystemExit(
        "Set QN_SSH_HOST, QN_SSH_PORT and QN_SSH_PASSWORD before running "
        "(optionally QN_SSH_USER; default 'root')."
    )


def ssh_exec(cmd: str, timeout: int = 120) -> str:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, look_for_keys=False, allow_agent=False)
    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    client.close()
    return (out + ("\n[stderr]\n" + err if err.strip() else "")).strip()


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--label":
        cmd = (
            "hostname; date; "
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
            "ls -d /root/quality_noise 2>/dev/null && echo PROJECT_PRESENT || echo NO_PROJECT; "
            "ls /root/quality_noise/results/estimator 2>/dev/null | head -3; "
            "find /root -name 'CIFAR-10_human_anno.npz' 2>/dev/null | head -2"
        )
    else:
        cmd = " ".join(args) if args else "echo EMPTY"
    print(ssh_exec(cmd))


if __name__ == "__main__":
    main()
