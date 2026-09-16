#!/usr/bin/env python
"""sync_project.py — archive the project source and push to the AutoDL server.

Uploads a tar (sources + configs + scripts + tests, excluding heavy output dirs)
via paramiko SFTP, then extracts it into /root/quality_noise.
"""
from __future__ import annotations

import io
import os
import tarfile

import paramiko

# Connection details come from the environment; credentials are never hardcoded
# in a tracked file.
HOST = os.environ.get("QN_SSH_HOST", "")
PORT = int(os.environ.get("QN_SSH_PORT", "22"))
USER = os.environ.get("QN_SSH_USER", "root")
PASS = os.environ.get("QN_SSH_PASSWORD", "")
PROJ = os.environ.get("QN_REMOTE_PROJ", "/root/quality_noise")

if not HOST or not PASS:
    raise SystemExit(
        "Set QN_SSH_HOST, QN_SSH_PORT and QN_SSH_PASSWORD before running "
        "(optionally QN_SSH_USER, QN_REMOTE_PROJ)."
    )


def _archive_bytes() -> bytes:
    """Tar the project (source only) into an in-memory gzip."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirname in ["src", "configs", "scripts", "tests"]:
            full = os.path.join(root, dirname)
            if os.path.isdir(full):
                tar.add(full, arcname=dirname)
        for fname in ["pyproject.toml", "README.md"]:
            p = os.path.join(root, fname)
            if os.path.exists(p):
                tar.add(p, arcname=fname)
    return buf.getvalue()


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, look_for_keys=False, allow_agent=False)

    data = _archive_bytes()
    sftp = client.open_sftp()
    remote = "/tmp/qn_source.tar.gz"
    with sftp.open(remote, "wb") as f:
        f.write(data)
    print(f"uploaded {len(data)/1e6:.2f} MB source -> {remote}")

    for c in [f"mkdir -p {PROJ}", f"tar xzf {remote} -C {PROJ}"]:
        _stdin, stdout, stderr = client.exec_command(c, timeout=120)
        out = stdout.read().decode(); err = stderr.read().decode()
        if err.strip():
            print(f"[cmd] {c} -> stderr: {err.strip()[:300]}")
    print("extracted to", PROJ)
    client.close()


if __name__ == "__main__":
    main()
