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


def build_docker_img(hostname=None, workspace_dir=None):
    if hostname:
        assert workspace_dir
        subprocess.check_call("rm -f > oneflow-src.zip", shell=True)
        subprocess.check_call(
            "git archive --format zip HEAD > oneflow-src.zip", shell=True
        )
        subprocess.check_call(
            f"scp oneflow-src.zip {hostname}:{workspace_dir}/oneflow-src.zip",
            shell=True,
        )
        subprocess.check_call(
            f"ssh  {hostname} unzip {workspace_dir}/oneflow-src.zip -d {workspace_dir}/oneflow-src",
            shell=True,
        )
        subprocess.check_call(
            f"ssh  {hostname} bash {workspace_dir}/oneflow-src/docker/ci/test/build.sh",
            shell=True,
        )
    else:
        subprocess.check_call(f"bash docker/ci/test/build.sh", shell=True)


def create_remote_workspace_dir(hostname, workspace_dir):
    subprocess.check_call(f"ssh {hostname} mkdir -p {workspace_dir}", shell=True)
    print("create_remote_workspace_dir done")


def launch_remote_container(
    hostname=None,
    survival_time=None,
    workspace_dir=None,
    container_name=None,
    img_tag=None,
    oneflow_wheel_path=None,
    oneflow_build_path=None,
):
    print("launching remote container at", hostname)
    assert img_tag
    pythonpath_args = None
    if oneflow_wheel_path:
        pythonpath_args = ""
    elif oneflow_build_path:
        pythonpath_args = f"--env PYTHONPATH={workspace_dir}/python_scripts"
    else:
        raise ValueError("must have oneflow_wheel_path or oneflow_build_path")
    docker_cmd = f"""docker run --privileged -d --network host --shm-size=8g --rm -v {workspace_dir}:{workspace_dir} -w {workspace_dir} -v /dataset:/dataset -v /model_zoo:/model_zoo --name {container_name} {pythonpath_args} {img_tag} sleep {survival_time}
"""
    ssh_cmd = f"ssh {hostname} {docker_cmd}"
    subprocess.check_call(ssh_cmd, shell=True)
    if oneflow_wheel_path:
        whl_basename = os.path.basename(oneflow_wheel_path)
        run_docker_cmd = f"ssh {hostname} docker exec {container_name} python3 -m pip install {workspace_dir}/{whl_basename}"
        subprocess.check_call(run_docker_cmd, shell=True)
    subprocess.check_call(
        f"ssh {hostname} docker exec {container_name} python3 -m oneflow --doctor",
        shell=True,
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


def launch_workers(agent_port=None, agent_authkey=None, remote_hosts=None):
    listener = Listener(("localhost", agent_port), authkey=agent_authkey)
    while True:
        conn = listener.accept()
        remote_docker_proc = {}
        for remote_host in remote_hosts:
            assert handle_cast(conn=conn, cmd="host"), remote_host
            env_proto_txt = handle_cast(conn=conn, cmd="env_proto")
            print("[docker agent]", f"[{remote_host}]", env_proto_txt)
            f = tempfile.NamedTemporaryFile(mode="wb+", delete=True)
            f.write(env_proto_txt.encode())
            f.flush()
            subprocess.check_call(
                f"rsync -azP --omit-dir-times --no-perms --no-group {f.name} {remote_host}:{workspace_dir}/env.prototxt",
                shell=True,
            )
            run_docker_cmd = f"ssh {remote_host} docker exec {container_name}"
            run_docker_cmd += f" python3 -m oneflow --start_worker --env_proto={workspace_dir}/env.prototxt"
            print("[docker agent]", run_docker_cmd)
            remote_docker_proc[remote_host] = subprocess.Popen(
                run_docker_cmd, shell=True
            )
            handle_call(conn=conn, cmd="start_worker", response="ok")
        for k, v in remote_docker_proc.items():
            assert v.wait() == 0


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
        oneflow_build_path=None,
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
        self.oneflow_build_path = oneflow_build_path
        # impl
        self.env_proto_txt = None
        self.bash_tmp_file = None
        self.bash_proc = None
        self.remote_docker_proc = {}
        self.agent_port = port
        self.agent_authkey = authkey

    def __enter__(self):
        return self

    def run_bash_script_async(self, bash_script=None):
        remote_hosts_str = ",".join(self.remote_hosts)
        assert os.path.exists(bash_script)
        log_dir = "./unittest-log-" + str(uuid.uuid4())
        ctrl_port = find_free_port()
        data_port = find_free_port()
        exports = f"""
export ONEFLOW_TEST_MASTER_PORT={ctrl_port}
export ONEFLOW_TEST_DATA_PORT={data_port}
export ONEFLOW_TEST_LOG_DIR={log_dir}
export ONEFLOW_TEST_NODE_LIST="{self.this_host},{remote_hosts_str}"
export ONEFLOW_WORKER_KEEP_LOG=1
export ONEFLOW_TEST_TMP_DIR="./distributed-tmp"
export NCCL_DEBUG=INFO
export ONEFLOW_TEST_WORKER_AGENT_PORT={agent_port}
export ONEFLOW_TEST_WORKER_AGENT_AUTHKEY={agent_authkey}
"""
        if self.oneflow_wheel_path:
            exports += f"python3 -m pip install {self.oneflow_wheel_path}"
        if self.oneflow_build_path:
            exports += f"export PYTHONPATH={self.oneflow_build_path}/python_scripts:$PYTHONPATH\n"
        bash_cmd = f"""set -ex
    {exports}
    bash {bash_script}
    """

        def get_docker_cmd(f, cmd):
            f_name = f.name
            f.write(cmd)
            f.flush()
            return f"docker run {self.common_docker_args} -v /tmp:/host/tmp:ro -v $PWD:$PWD -w $PWD --name {self.container_name} {self.img_tag} bash /host{f_name}"

        f = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=True)
        run_docker_cmd = get_docker_cmd(f, bash_cmd)
        self.bash_tmp_file = f
        self.bash_proc = subprocess.Popen(run_docker_cmd, shell=True)

    def block(self):
        from multiprocessing import Process

        p = None
        kwargs = {
            "agent_port": self.agent_port,
            "agent_authkey": self.agent_authkey,
            "remote_hosts": self.remote_hosts,
        }
        p = Process(target=launch_workers, kwargs=kwargs,)
        p.start()
        print("[docker agent]", "blocking")
        # TODO: it is too easy to deadlock here, refactor polling with async await
        while self.bash_proc.poll() is None and p.is_alive() == True:
            pass
        p.terminate()
        assert self.bash_proc.returncode == 0
        print("[docker agent]", "bash execution done")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def fix_and_sync_libs():
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_lib_dir = os.path.join(tmp_dir.name, "libs")
    os.mkdir(tmp_lib_dir)
    subprocess.check_call(
        """ldd file | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' destination""".replace(
            "file", oneflow_internal_path
        ).replace(
            "destination", tmp_lib_dir
        ),
        shell=True,
    )
    libs = os.listdir(tmp_lib_dir)
    assert len(libs) > 0
    excludelist_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "excludelist"
    )
    excludelist = open(excludelist_path).read().split("\n")
    subprocess.check_call(
        f"cp {oneflow_internal_path} {tmp_dir.name}", shell=True,
    )
    for lib in libs:
        if lib in excludelist or "libpython" in lib:
            print("excluding", lib)
            subprocess.check_call(
                f"rm {tmp_lib_dir}/{lib}", shell=True,
            )
        else:
            print("keeping", lib)
            subprocess.check_call(
                f"patchelf --set-rpath '$ORIGIN' {tmp_lib_dir}/{lib}", shell=True,
            )
    tmp_oneflow_internal_path = os.path.join(
        tmp_dir.name, pathlib.Path(oneflow_internal_path).name
    )
    print("before fixing .so")
    subprocess.check_call(
        f"ldd {tmp_oneflow_internal_path}", shell=True,
    )
    print("fixing .so")
    subprocess.check_call(
        f"patchelf --set-rpath '$ORIGIN/libs' {tmp_oneflow_internal_path}", shell=True,
    )
    subprocess.check_call(
        f"ldd {tmp_oneflow_internal_path}", shell=True,
    )


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
    parser.add_argument("--oneflow_build_path", type=str, required=False, default=None)
    parser.add_argument("--custom_img_tag", type=str, required=False, default=None)
    parser.add_argument("--timeout", type=int, required=False, default=6 * 60 * 60)
    args = parser.parse_args()

    assert bool(args.oneflow_wheel_path) != bool(args.oneflow_build_path)
    this_host = args.this_host
    this_host = resolve_hostname_hardcoded(this_host)

    remote_hosts = []
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

    print(f"this_host: {this_host}, remote_hosts: {remote_hosts}", flush=True)
    sub_dir = str(uuid.uuid4())
    if args.debug:
        sub_dir = "debug"
    workspace_dir = os.path.join(
        os.path.expanduser("~"), "distributed_run_workspace", sub_dir
    )
    print("workspace_dir", workspace_dir)
    create_remote_workspace_dir(remote_host, workspace_dir)
    if args.oneflow_build_path:
        # TODO: infer a proper path and check there is only one
        oneflow_internal_path = (
            "python_scripts/oneflow/_oneflow_internal.cpython-36m-x86_64-linux-gnu.so"
        )
        oneflow_internal_path = os.path.join(
            args.oneflow_build_path, oneflow_internal_path
        )
        tmp_dir = None
        if args.skip_libs == False:
            print("copying .so")
            fix_and_sync_libs()
        print("copying python_scripts dir")
        subprocess.check_call(
            f"rsync -azP --omit-dir-times --no-perms --no-group --copy-links --include='*.py' --exclude='*.so' --exclude='__pycache__' --exclude='python_scripts/oneflow/include' --include='*/' --exclude='*' {args.oneflow_build_path}/python_scripts {remote_host}:{workspace_dir}",
            shell=True,
        )
        if args.skip_libs == False:
            subprocess.check_call(
                f"rsync -azP --omit-dir-times --no-perms --no-group {tmp_dir.name}/ {remote_host}:{workspace_dir}/python_scripts/oneflow",
                shell=True,
            )
    elif args.oneflow_wheel_path:
        subprocess.check_call(
            f"rsync -azP --omit-dir-times --no-perms --no-group {args.oneflow_wheel_path} {remote_host}:{workspace_dir}",
            shell=True,
        )
    default_docker_image = "oneflow-test:$USER"
    ci_user_docker_image = "oneflow-test:ci-user"
    img_tag = None
    if args.custom_img_tag == None:
        if is_img_existing(default_docker_image):
            img_tag = default_docker_image
        elif is_img_existing(ci_user_docker_image):
            img_tag = ci_user_docker_image
        else:
            build_docker_img(remote_host, workspace_dir)
            img_tag = default_docker_image
    else:
        img_tag = args.custom_img_tag
    assert img_tag
    assert args.bash_script
    agent_port = find_free_port()
    agent_authkey = str(uuid.uuid4())
    container_name = getpass.getuser() + "-distributed-run"

    def exit_handler():
        print("removing local docker container:", container_name)
        rm_cmd = f"docker rm -f {container_name}"
        subprocess.call(f"{rm_cmd}", shell=True)
        for remote_host in remote_hosts:
            print(f"removing local docker container at {remote_host}:", container_name)
            subprocess.call(
                f"ssh {remote_host} {rm_cmd}", shell=True,
            )

    atexit.register(exit_handler)
    for remote_host in remote_hosts:
        launch_remote_container(
            hostname=remote_host,
            survival_time=args.timeout,
            workspace_dir=workspace_dir,
            container_name=container_name,
            oneflow_wheel_path=args.oneflow_wheel_path,
            oneflow_build_path=args.oneflow_build_path,
            img_tag=img_tag,
        )
    with DockerAgent(
        port=agent_port,
        authkey=agent_authkey.encode(),
        this_host=this_host,
        remote_hosts=[remote_host],
        container_name=container_name,
        workspace_dir=workspace_dir,
        oneflow_wheel_path=args.oneflow_wheel_path,
        oneflow_build_path=args.oneflow_build_path,
        img_tag=img_tag,
    ) as agent:
        agent.run_bash_script_async(bash_script=args.bash_script,)
        agent.block()

    # copy artifacts
    exit(0)
