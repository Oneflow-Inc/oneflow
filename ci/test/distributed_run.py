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
from pathlib import Path

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


async def spawn_shell(cmd: str = None):
    p = await asyncio.create_subprocess_shell(cmd,)
    await p.wait()
    assert p.returncode == 0, cmd


async def spawn_shell_ignoring_failure(cmd: str = None):
    p = await asyncio.create_subprocess_shell(cmd,)
    await p.wait()


async def build_docker_img(remote_host=None, workspace_dir=None):
    if remote_host:
        assert workspace_dir
        await spawn_shell("rm -f > oneflow-src.zip")
        await spawn_shell("git archive --format zip HEAD > oneflow-src.zip")
        await spawn_shell(
            f"scp oneflow-src.zip {remote_host}:{workspace_dir}/oneflow-src.zip",
        )
        await spawn_shell(
            f"ssh  {remote_host} unzip {workspace_dir}/oneflow-src.zip -d {workspace_dir}/oneflow-src",
        )
        await spawn_shell(
            f"ssh  {remote_host} bash {workspace_dir}/oneflow-src/docker/ci/test/build.sh",
        )
    else:
        await spawn_shell(f"bash docker/ci/test/build.sh")


async def create_remote_workspace_dir(
    remote_host=None, workspace_dir=None, copy_files=None
):
    await spawn_shell(f"ssh {remote_host} mkdir -p {workspace_dir}")
    if copy_files is not None:
        for path in copy_files:
            # Reference: https://stackoverflow.com/a/31278462
            if os.path.isdir(path) and path[-1] != "/":
                path += "/"
            await spawn_shell(f"ssh {remote_host} mkdir -p {workspace_dir}/{path}")
            await spawn_shell(
                f"rsync -azPq --omit-dir-times --no-perms --no-group --copy-links --exclude='__pycache__' {path} {remote_host}:{workspace_dir}/{path}"
            )
    print("create_remote_workspace_dir done")


def get_docker_cache_args():
    return " ".join(
        [
            f"-v {Path.home() / 'test-container-cache/dot-local'}:/root/.local",
            f"-v {Path.home() / 'test-container-cache/dot-cache'}:/root/.cache",
        ]
    )


async def launch_remote_container(
    remote_host=None,
    survival_time=None,
    workspace_dir=None,
    container_name=None,
    img_tag=None,
    oneflow_wheel_path=None,
    oneflow_python_path=None,
    cmd=None,
    node_rank=None,
    master_addr=None,
):
    print("launching remote container at", remote_host)
    assert img_tag
    multi_client_args = [node_rank, master_addr]
    multi_client_arg_has_value = [x is not None for x in multi_client_args]
    assert all(multi_client_arg_has_value)
    pythonpath_args = None
    if oneflow_wheel_path:
        pythonpath_args = ""
    elif oneflow_python_path:
        pythonpath_args = f"--env PYTHONPATH={workspace_dir}/python"
    else:
        raise ValueError("must have oneflow_wheel_path or oneflow_python_path")
    docker_cmd = f"""docker run --privileged -d --network host --shm-size=8g --rm {get_docker_cache_args()} -v {workspace_dir}:{workspace_dir} -w {workspace_dir} -v /dataset:/dataset -v /model_zoo:/model_zoo --name {container_name} {pythonpath_args} {img_tag} sleep {survival_time}
"""
    await spawn_shell(f"ssh {remote_host} {docker_cmd}")
    if oneflow_wheel_path:
        whl_basename = os.path.basename(oneflow_wheel_path)
        await spawn_shell(
            f"ssh {remote_host} docker exec {container_name} python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        await spawn_shell(
            f"ssh {remote_host} docker exec {container_name} python3 -m pip install {workspace_dir}/{whl_basename}"
        )
    await spawn_shell(
        f"ssh {remote_host} docker exec {container_name} python3 -m oneflow --doctor"
    )
    if cmd:
        multi_client_docker_args = (
            # Use _MASTER_ADDR to avoid name conflict with OneFlow's built-in MASTER_ADDR
            f"--env NODE_RANK={node_rank} --env _MASTER_ADDR={master_addr}"
        )
        await spawn_shell(
            f"ssh {remote_host} docker exec {multi_client_docker_args} {container_name} {cmd}"
        )


def handle_cast(conn=None, cmd=None):
    received_cmd: str = conn.recv().decode()
    assert received_cmd.startswith("cast/")
    received_cmd = received_cmd.replace("cast/", "")
    assert received_cmd == cmd, (received_cmd, cmd)
    return conn.recv().decode()


def handle_call(conn=None, cmd=None, response=None):
    received_cmd: str = conn.recv().decode()
    assert received_cmd.startswith("call/")
    received_cmd = received_cmd.replace("call/", "")
    assert received_cmd == cmd, (received_cmd, cmd)
    msg = conn.recv().decode()
    conn.send(response.encode())
    return msg


class DockerAgent:
    def __init__(
        self,
        port=None,
        authkey=None,
        this_host=None,
        remote_hosts=None,
        container_name=None,
        timeout=None,
        workspace_dir=None,
        img_tag=None,
        oneflow_wheel_path=None,
        oneflow_python_path=None,
        oneflow_test_tmp_dir=None,
        extra_docker_args: str = None,
    ) -> None:
        # info
        self.this_host = this_host
        self.remote_hosts = remote_hosts
        self.container_name = container_name
        self.timeout = timeout
        self.common_docker_args = "--privileged --rm --network host --shm-size=8g -v $HOME:$HOME -v /dataset:/dataset -v /model_zoo:/model_zoo"
        self.workspace_dir = workspace_dir
        self.img_tag = img_tag
        self.oneflow_wheel_path = oneflow_wheel_path
        self.oneflow_python_path = oneflow_python_path
        self.oneflow_test_tmp_dir = oneflow_test_tmp_dir
        # impl
        self.env_proto_txt = None
        self.bash_tmp_file = None
        self.bash_proc = None
        self.remote_docker_proc = {}
        self.agent_port = port
        self.agent_authkey = authkey
        self.extra_docker_args = extra_docker_args

    def __enter__(self):
        return self

    def run_bash_script_async(self, bash_script=None, cmd=None):
        remote_hosts_str = ",".join(self.remote_hosts)
        ctrl_port = find_free_port()
        data_port = find_free_port()
        exports = f"""
export ONEFLOW_TEST_MASTER_PORT={ctrl_port}
export ONEFLOW_TEST_DATA_PORT={data_port}
export ONEFLOW_TEST_NODE_LIST="{self.this_host},{remote_hosts_str}"
export ONEFLOW_WORKER_KEEP_LOG=1
export ONEFLOW_TEST_TMP_DIR="{self.oneflow_test_tmp_dir}"
export NCCL_DEBUG=INFO
export ONEFLOW_TEST_WORKER_AGENT_PORT={agent_port}
export ONEFLOW_TEST_WORKER_AGENT_AUTHKEY={agent_authkey}
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
"""
        if self.oneflow_wheel_path:
            exports += f"python3 -m pip install {self.oneflow_wheel_path}"
        if self.oneflow_python_path:
            exports += f"export PYTHONPATH={self.oneflow_python_path}:$PYTHONPATH\n"
        bash_cmd = None
        if bash_script:
            assert os.path.exists(bash_script)
            bash_cmd = f"""set -ex
{exports}
bash {bash_script}
"""
        elif cmd:
            bash_cmd = f"""set -ex
{exports}
{cmd}
"""
        else:
            raise ValueError("not impl")
        assert bash_cmd

        def get_docker_cmd(f, cmd):
            f_name = f.name
            f.write(cmd)
            f.flush()
            return f"docker run {self.common_docker_args} {self.extra_docker_args} {get_docker_cache_args()} -v /tmp:/host/tmp:ro -v $PWD:$PWD -w $PWD --name {self.container_name} {self.img_tag} bash /host{f_name}"

        f = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=True)
        run_docker_cmd = get_docker_cmd(f, bash_cmd)
        self.bash_tmp_file = f
        self.bash_proc = subprocess.Popen(run_docker_cmd, shell=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


async def fix_and_sync_libs(oneflow_internal_path=None, remote_hosts=None):
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_lib_dir = os.path.join(tmp_dir.name, "libs")
    os.mkdir(tmp_lib_dir)
    await spawn_shell(
        """ldd file | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' destination""".replace(
            "file", oneflow_internal_path
        ).replace(
            "destination", tmp_lib_dir
        ),
    )
    libs = os.listdir(tmp_lib_dir)
    assert len(libs) > 0
    excludelist_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "excludelist"
    )
    excludelist = open(excludelist_path).read().split("\n")
    await spawn_shell(f"cp {oneflow_internal_path} {tmp_dir.name}")

    def handle_lib(lib):
        if lib in excludelist or "libpython" in lib:
            print("excluding", lib)
            return spawn_shell(f"rm {tmp_lib_dir}/{lib}")
        else:
            print("keeping", lib)
            return spawn_shell(f"patchelf --set-rpath '$ORIGIN' {tmp_lib_dir}/{lib}")

    await asyncio.gather(*(handle_lib(lib) for lib in libs))

    tmp_oneflow_internal_path = os.path.join(
        tmp_dir.name, pathlib.Path(oneflow_internal_path).name
    )
    print("before fixing .so")
    await spawn_shell(f"ldd {tmp_oneflow_internal_path}")
    print("fixing .so")
    await spawn_shell(
        f"patchelf --set-rpath '$ORIGIN/libs' {tmp_oneflow_internal_path}"
    )

    await asyncio.gather(
        *[
            spawn_shell(
                f"ssh {remote_host} 'mkdir -p {workspace_dir}/python/oneflow/libs'",
            )
            for remote_host in remote_hosts
        ]
    )

    async def copy_file(path=None, remote_host=None):
        relpath = os.path.relpath(path, tmp_dir.name)
        await spawn_shell(
            f"scp {path} {remote_host}:{workspace_dir}/python/oneflow/{relpath}",
        )

    files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(tmp_dir.name, topdown=True)
        for name in files
    ]

    await asyncio.gather(
        *[
            copy_file(path=f, remote_host=remote_host)
            for remote_host in remote_hosts
            for f in files
        ],
        spawn_shell(f"ldd {tmp_oneflow_internal_path}"),
    )


async def remove_containers_by_name(remote_hosts=None, container_name=None):
    rm_cmd = f"docker rm -f {container_name}"
    assert container_name
    assert remote_hosts
    await asyncio.gather(
        *[
            spawn_shell_ignoring_failure(f"ssh {remote_host} {rm_cmd}")
            for remote_host in remote_hosts
        ],
        spawn_shell_ignoring_failure(rm_cmd),
    )


def get_remote_hosts(args):
    remote_hosts = None
    if len(args.remote_host) == 1:
        remote_hosts = args.remote_host.split(",")
    elif len(args.remote_host) == 0:
        affiliations = get_affiliations(this_host)
        assert (
            affiliations
        ), f"no affiliated node found for {this_host}, you should specify one"
        remote_host = affiliations[0]
        remote_host = socket.gethostbyname(remote_host)
        remote_hosts = [remote_host]
    else:
        remote_hosts = args.remote_host
    return remote_hosts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", required=False, default=False)
    parser.add_argument(
        "--skip_libs", action="store_true", required=False, default=False
    )
    parser.add_argument("--bash_script", type=str, required=False)
    default_this_host = socket.gethostname()
    parser.add_argument(
        "--this_host", type=str, required=False, default=default_this_host
    )
    parser.add_argument("--remote_host", action="append", default=[])
    parser.add_argument("--oneflow_wheel_path", type=str, required=False, default=None)
    parser.add_argument(
        "--oneflow_wheel_python_version", type=str, required=False, default=None
    )
    parser.add_argument("--oneflow_python_path", type=str, required=False, default=None)
    parser.add_argument("--custom_img_tag", type=str, required=False, default=None)
    parser.add_argument("--cmd", type=str, required=False, default=None)
    parser.add_argument(
        "--oneflow_test_tmp_dir", type=str, required=False, default="distributed-tmp"
    )
    parser.add_argument("--timeout", type=int, required=False, default=1 * 60 * 60)
    parser.add_argument("--mode", type=str, required=False, default="multi_client")
    parser.add_argument("--copy_files", action="append", default=[])
    args = parser.parse_args()

    assert args.mode in ["multi_client"]
    assert bool(args.oneflow_wheel_path) != bool(args.oneflow_python_path)
    assert bool(args.bash_script) != bool(args.cmd)
    if args.skip_libs:
        assert args.debug, "--skip_libs only works with --debug"
        assert (
            args.oneflow_python_path
        ), "--skip_libs only works with --oneflow_python_path"

    oneflow_wheel_path = args.oneflow_wheel_path
    main_node_extra_docker_args = []
    if oneflow_wheel_path and os.path.isdir(oneflow_wheel_path):
        assert os.path.isabs(oneflow_wheel_path)
        main_node_extra_docker_args.append(
            f"-v {oneflow_wheel_path}:{oneflow_wheel_path}:ro"
        )
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

    this_host = args.this_host
    this_host = resolve_hostname_hardcoded(this_host)

    remote_hosts = get_remote_hosts(args)

    print(f"this_host: {this_host}, remote_hosts: {remote_hosts}", flush=True)
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
        + this_host.replace(".", "-")
    )
    if args.mode == "multi_client":
        remote_hosts = [this_host] + remote_hosts
    loop = asyncio.get_event_loop()
    # add host key to all machines (needed by ssh/scp/rsync)
    loop.run_until_complete(
        asyncio.gather(
            *[
                spawn_shell(f"ssh -o StrictHostKeyChecking=no {remote_host} true")
                for remote_host in remote_hosts
            ],
        ),
    )
    loop.run_until_complete(
        asyncio.gather(
            *[
                create_remote_workspace_dir(
                    remote_host=remote_host,
                    workspace_dir=workspace_dir,
                    copy_files=args.copy_files,
                )
                for remote_host in remote_hosts
            ],
            remove_containers_by_name(
                remote_hosts=remote_hosts, container_name=container_name
            ),
        ),
    )
    if args.oneflow_python_path:
        so_paths = [
            name
            for name in glob.glob(
                os.path.join(
                    args.oneflow_python_path, f"oneflow/_oneflow_internal.*.so",
                )
            )
        ]
        assert len(so_paths) == 1, so_paths
        oneflow_internal_path = so_paths[0]
        oneflow_internal_path = os.path.join(
            args.oneflow_python_path, oneflow_internal_path
        )
        tmp_dir = None
        print("copying oneflow python dir")
        loop.run_until_complete(
            asyncio.gather(
                *[
                    spawn_shell(
                        f"rsync -azPq --omit-dir-times --no-perms --no-group --copy-links --include='*.py' --exclude='*.so' --exclude='__pycache__' --exclude='oneflow/include' --include='*/' --exclude='*' {args.oneflow_python_path} {remote_host}:{workspace_dir}"
                    )
                    for remote_host in remote_hosts
                ]
            )
        )
        if args.skip_libs == False:
            print("copying .so")
            loop.run_until_complete(
                fix_and_sync_libs(
                    oneflow_internal_path=oneflow_internal_path,
                    remote_hosts=remote_hosts,
                )
            )
    elif oneflow_wheel_path:
        loop.run_until_complete(
            asyncio.gather(
                *[
                    spawn_shell(
                        f"rsync -azPq --omit-dir-times --no-perms --no-group {oneflow_wheel_path} {remote_host}:{workspace_dir}"
                    )
                    for remote_host in remote_hosts
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
                        for remote_host in remote_hosts
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
        if args.oneflow_python_path:
            print("fixing permission of", args.oneflow_python_path)
            subprocess.call(
                f"docker run --rm -v {args.oneflow_python_path}:/p -w /p busybox chmod -R o+w .",
                shell=True,
            )
        loop.run_until_complete(
            asyncio.gather(
                *[
                    spawn_shell_ignoring_failure(
                        f"ssh {remote_host} docker run --rm -v {workspace_dir}:/p -w /p busybox chmod -R 777 .",
                    )
                    for remote_host in remote_hosts
                ],
            )
        )
        print("copying artifacts")
        extra_exclude_args = ""
        for path in args.copy_files:
            extra_exclude_args += f"--exclude='{path}' "
        loop.run_until_complete(
            asyncio.gather(
                *[
                    spawn_shell_ignoring_failure(
                        f"rsync -azPq --omit-dir-times --no-perms --no-group --exclude='*.whl' --exclude='python' {extra_exclude_args} {remote_host}:{workspace_dir}/ {args.oneflow_test_tmp_dir}/{remote_host}"
                    )
                    for remote_host in remote_hosts
                ]
            )
        )
        assert workspace_dir
        if args.debug == False:
            print("removing docker workspace_dir:", workspace_dir)
            loop.run_until_complete(
                asyncio.gather(
                    *[
                        spawn_shell_ignoring_failure(
                            f"ssh {remote_host} rm -rf {workspace_dir}",
                        )
                        for remote_host in remote_hosts
                    ],
                )
            )
        print("removing docker container:", container_name)
        loop.run_until_complete(
            remove_containers_by_name(
                remote_hosts=remote_hosts, container_name=container_name
            )
        )

    atexit.register(exit_handler)
    if args.mode == "multi_client":
        if args.bash_script:
            args.cmd = f"bash {args.bash_script}"
        loop.run_until_complete(
            asyncio.gather(
                *[
                    launch_remote_container(
                        remote_host=remote_host,
                        survival_time=args.timeout,
                        workspace_dir=workspace_dir,
                        container_name=container_name,
                        oneflow_wheel_path=oneflow_wheel_path,
                        oneflow_python_path=args.oneflow_python_path,
                        img_tag=img_tag,
                        cmd=args.cmd,
                        node_rank=node_rank,
                        master_addr=this_host,
                    )
                    for node_rank, remote_host in enumerate(remote_hosts)
                ],
            )
        )
    else:
        loop.run_until_complete(
            asyncio.gather(
                *[
                    launch_remote_container(
                        remote_host=remote_host,
                        survival_time=args.timeout,
                        workspace_dir=workspace_dir,
                        container_name=container_name,
                        oneflow_wheel_path=oneflow_wheel_path,
                        oneflow_python_path=args.oneflow_python_path,
                        img_tag=img_tag,
                    )
                    for remote_host in remote_hosts
                ],
            )
        )
