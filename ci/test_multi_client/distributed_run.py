from multiprocessing.connection import Listener
import os
import subprocess
import socket
import tempfile
from contextlib import closing
import argparse
import uuid
import getpass
import atexit
import pathlib
import asyncio
import glob
from datetime import date

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


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


async def spawn_shell_and_check(cmd: str = None):
    p = await asyncio.create_subprocess_shell(cmd,)
    await p.wait()
    assert p.returncode == 0, cmd


async def spawn_shell(cmd: str = None):
    p = await asyncio.create_subprocess_shell(cmd,)
    await p.wait()


async def build_docker_img(remote_host=None, workspace_dir=None):
    if remote_host:
        assert workspace_dir
        await spawn_shell_and_check(
            f"ssh  {remote_host} bash {workspace_dir}/docker/ci/test/build.sh",
        )
    else:
        await spawn_shell_and_check(f"bash docker/ci/test/build.sh")


async def create_remote_workspace_dir(remote_host=None, workspace_dir=None):
    await spawn_shell_and_check(f"ssh {remote_host} mkdir -p {workspace_dir}")
    await spawn_shell_and_check("rm -f > oneflow-src.zip")
    await spawn_shell_and_check("git diff --quiet")
    await spawn_shell_and_check("git archive --format zip HEAD > oneflow-src.zip")
    await spawn_shell_and_check(
        f"scp oneflow-src.zip {remote_host}:{workspace_dir}/oneflow-src.zip",
    )
    await spawn_shell_and_check(
        f"ssh {remote_host} unzip -qq {workspace_dir}/oneflow-src.zip -d {workspace_dir}",
    )
    print("create_remote_workspace_dir done")


async def launch_remote_container(
    remote_host=None,
    survival_time=None,
    workspace_dir=None,
    container_name=None,
    img_tag=None,
    oneflow_wheel_path=None,
    cmd=None,
    node_rank=None,
):
    print("launching remote container at", remote_host)
    assert img_tag
    pythonpath_args = None
    if oneflow_wheel_path:
        pythonpath_args = ""
    else:
        raise ValueError("must have oneflow_wheel_path")
    docker_cmd = f"""docker run --privileged -d --network host --shm-size=8g --rm -v {workspace_dir}:{workspace_dir} -w {workspace_dir} -v /dataset:/dataset -v /model_zoo:/model_zoo --name {container_name} {pythonpath_args} {img_tag} sleep {survival_time}
"""
    await spawn_shell_and_check(f"ssh {remote_host} {docker_cmd}")
    if oneflow_wheel_path:
        whl_basename = os.path.basename(oneflow_wheel_path)
        await spawn_shell_and_check(
            f"ssh {remote_host} docker exec {container_name} python3 -m pip install {workspace_dir}/{whl_basename}"
        )
    await spawn_shell(
        f"ssh {remote_host} docker exec {container_name} python3 -m oneflow --doctor"
    )
    if cmd:
        await spawn_shell(
            f"ssh {remote_host} docker exec --env NODE_RANK={node_rank} {container_name} {cmd}"
        )


async def remove_containers_by_name(remote_hosts=None, container_name=None):
    rm_cmd = f"docker rm -f {container_name}"
    assert container_name
    assert remote_hosts
    await asyncio.gather(
        *[spawn_shell(f"ssh {remote_host} {rm_cmd}") for remote_host in remote_hosts],
        spawn_shell(rm_cmd),
    )


def get_remote_hosts(args):
    if len(args.machine) == 0:
        this_host = socket.gethostname()
        this_host = resolve_hostname_hardcoded(this_host)

        affiliations = get_affiliations(this_host)
        assert (
            affiliations
        ), f"no affiliated node found for {this_host}, you should specify one"
        remote_host = affiliations[0]
        remote_host = socket.gethostbyname(remote_host)
        machines = [this_host, remote_host]
    else:
        machines = args.machine
    return machines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", required=False, default=False)
    parser.add_argument("--bash_script", type=str, required=False)
    parser.add_argument("--machine", action="append", default=[])
    parser.add_argument("--oneflow_wheel_path", type=str, required=True, default=None)
    parser.add_argument(
        "--oneflow_wheel_python_version", type=str, required=False, default=None
    )
    parser.add_argument("--custom_img_tag", type=str, required=False, default=None)
    parser.add_argument("--cmd", type=str, required=False, default=None)
    parser.add_argument("--timeout", type=int, required=False, default=1 * 60 * 60)
    args = parser.parse_args()

    assert bool(args.bash_script) != bool(args.cmd)
    if args.bash_script:
        args.cmd = f'bash {args.bash_script}'
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

    machines = get_remote_hosts(args)

    print(f"machines: {machines}", flush=True)
    sub_dir = str(uuid.uuid4())
    if args.debug:
        sub_dir = "debug"
    workspace_dir = os.path.join(
        os.path.expanduser("~"), "distributed_run_workspace", sub_dir
    )
    print("workspace_dir", workspace_dir)
    container_name = (
        getpass.getuser()
        + "-distributed-run-main-node-at-"
        + machines[0].replace(".", "-")
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[
                create_remote_workspace_dir(
                    remote_host=remote_host, workspace_dir=workspace_dir
                )
                for remote_host in machines
            ],
            remove_containers_by_name(
                remote_hosts=machines, container_name=container_name
            ),
        ),
    )
    loop.run_until_complete(
        asyncio.gather(
            *[
                spawn_shell_and_check(
                    f"rsync -azP --omit-dir-times --no-perms --no-group {oneflow_wheel_path} {remote_host}:{workspace_dir}"
                )
                for remote_host in machines
            ]
        )
    )
    default_docker_image = "oneflow-test:$USER"
    ci_user_docker_image = "oneflow-test:0.2"
    img_tag = None
    if args.custom_img_tag == None:
        if is_img_existing(default_docker_image):
            img_tag = default_docker_image
        elif is_img_existing(ci_user_docker_image):
            img_tag = ci_user_docker_image
        else:
            loop.run_until_complete(
                asyncio.gather(
                    *[
                        build_docker_img(
                            remote_host=remote_host, workspace_dir=workspace_dir
                        )
                        for remote_host in machines
                    ],
                    build_docker_img(workspace_dir=workspace_dir),
                )
            )
            img_tag = default_docker_image
    else:
        img_tag = args.custom_img_tag
    assert img_tag
    agent_port = find_free_port()
    agent_authkey = str(uuid.uuid4())

    def exit_handler():
        print(
            "---------start cleanup, you should ignore errors below and check the errors above---------"
        )
        loop.run_until_complete(
            asyncio.gather(
                *[
                    spawn_shell(
                        f"ssh {remote_host} docker run --rm -v {workspace_dir}:/p -w /p busybox chmod -R 777 .",
                    )
                    for remote_host in machines
                ],
            )
        )
        assert workspace_dir
        if args.debug == False:
            print("removing docker workspace_dir:", workspace_dir)
            loop.run_until_complete(
                asyncio.gather(
                    *[
                        spawn_shell(f"ssh {remote_host} rm -rf {workspace_dir}",)
                        for remote_host in machines
                    ],
                )
            )
        print("removing docker container:", container_name)
        loop.run_until_complete(
            remove_containers_by_name(
                remote_hosts=machines, container_name=container_name
            )
        )

    atexit.register(exit_handler)

    loop.run_until_complete(
        asyncio.gather(
            *[
                launch_remote_container(
                    remote_host=remote_host,
                    survival_time=args.timeout,
                    workspace_dir=workspace_dir,
                    container_name=container_name,
                    oneflow_wheel_path=oneflow_wheel_path,
                    img_tag=img_tag,
                    cmd=args.cmd,
                    node_rank=node_rank,
                )
                for node_rank, remote_host in enumerate(machines)
            ],
        )
    )

