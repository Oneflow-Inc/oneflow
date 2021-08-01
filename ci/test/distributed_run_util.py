import os
import subprocess
import socket
import argparse
import uuid
import getpass
import atexit
import asyncio
import glob

HARD_CODED_AFFILIATIONS = {
    "192.168.1.11": ["192.168.1.12",],
    "192.168.1.12": ["192.168.1.11",],
    "192.168.1.13": ["192.168.1.11",],
    "192.168.1.15": ["192.168.1.16",],
    "192.168.1.16": ["192.168.1.15",],
}


def is_img_existing(tag):
    returncode = subprocess.run(
        "docker image inspect {}".format(tag),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    if returncode == 0:
        print("[OK]", tag)
        return True
    else:
        return False


def get_affiliations(host):
    # TODO(tsai): Implement a HTTP endpoint to retrieve affiliations
    if host in HARD_CODED_AFFILIATIONS:
        return HARD_CODED_AFFILIATIONS[host]
    else:
        return None


def resolve_hostname_hardcoded(host: str):
    if host.startswith("oneflow"):
        number = host.split("-")[-1]
        return f"192.168.1.{number}"
    else:
        return host


async def spawn_shell_and_check(cmd: str = None):
    p = await asyncio.create_subprocess_shell(cmd,)
    await p.wait()
    assert p.returncode == 0, cmd


async def spawn_shell(cmd: str = None):
    p = await asyncio.create_subprocess_shell(cmd,)
    await p.wait()


async def remove_containers_by_name(remote_hosts=None, container_name=None):
    rm_cmd = f"docker rm -f {container_name}"
    assert container_name
    assert remote_hosts
    await asyncio.gather(
        *[spawn_shell(f"ssh {remote_host} {rm_cmd}") for remote_host in remote_hosts],
        spawn_shell(rm_cmd),
    )

def get_oneflow_wheel_path(args):
    oneflow_wheel_path = args.oneflow_wheel_path
    if oneflow_wheel_path and os.path.isdir(oneflow_wheel_path):
        whl_paths = [
            name for name in glob.glob(os.path.join(oneflow_wheel_path, f"*.whl",))
        ]
        if len(whl_paths) == 1:
            oneflow_wheel_path = whl_paths[0]
        else:
            assert args.oneflow_wheel_python_version
            assert args.oneflow_wheel_python_version in [
                "3.6",
                "3.7",
                "3.8",
                "3.9",
                "3.10",
                "3.11",
            ]
            ver_cat = args.oneflow_wheel_python_version.replace(".", "")
            found = False
            for whl_path in whl_paths:
                if f"cp{ver_cat}" in whl_path:
                    oneflow_wheel_path = whl_path
                    found = True
            assert found, whl_paths
    return oneflow_wheel_path
