#!/usr/bin/env python3
"""
SSH connection manager for driving rorqual's automation node from CI,
through the jump server.
"""
import io
import os
import subprocess


class ClusterSSH:

    def __init__(self, host_alias="robot-rorqual", ssh_opts=None):
        self.host = host_alias
        self.ssh_opts = ssh_opts or [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=30",
        ]

    def run(self, remote_cmd, check=True, capture=True):
        """Run a command on the automation node. Returns (rc, stdout, stderr).

        The robot account is behind allowed_commands.sh, which rejects unknown
        commands but still exits 0, so rejection must be detected from stdout.
        That whitelist also expands $SSH_ORIGINAL_COMMAND unquoted, so ';',
        '&&' and redirects are passed as literal arguments: one command only.
        """
        cmd = ["ssh"] + self.ssh_opts + [self.host, remote_cmd]
        p = subprocess.run(cmd, capture_output=capture, text=True)
        if "Command rejected by" in (p.stdout or ""):
            if check:
                raise RuntimeError(f"remote command not permitted: {remote_cmd}")
            return 126, (p.stdout or ""), (p.stderr or "")
        if check and p.returncode != 0:
            raise RuntimeError(
                f"remote command failed (rc={p.returncode}): {remote_cmd}\n"
                f"stdout: {p.stdout}\nstderr: {p.stderr}")
        return p.returncode, (p.stdout or ""), (p.stderr or "")

    def exists(self, remote_path):
        """True if a file/dir exists on the cluster."""
        cmd = ["rsync"] + self.ssh_opts_for_rsync() + [
            "--list-only", f"{self.host}:{remote_path}"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode == 0

    def ssh_opts_for_rsync(self):
        joined = "ssh " + " ".join(self.ssh_opts)
        return ["-e", joined]

    def scp_up(self, local_path, remote_path):
        """Copy a local file/dir up to the cluster (through the proxyjump alias)."""
        cmd = ["scp"] + self.ssh_opts + ["-r", local_path, f"{self.host}:{remote_path}"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"scp up failed: {local_path} -> {remote_path}\n{p.stderr}")

    def scp_down(self, remote_path, local_path):
        """Copy a file/dir down from the cluster."""
        cmd = ["scp"] + self.ssh_opts + ["-r", f"{self.host}:{remote_path}", local_path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"scp down failed: {remote_path} -> {local_path}\n{p.stderr}")

    def squeue_job_names(self, user="arelbaha"):
        """Return the set of job NAMES currently queued/running for the user."""
        rc, out, _ = self.run(f"squeue -u {user} -h -o %j", check=False)
        if rc != 0:
            return set()
        return {line.strip() for line in out.splitlines() if line.strip()}
