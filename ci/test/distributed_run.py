from multiprocessing.connection import Listener
import os
import subprocess
import socket
import tempfile
from contextlib import closing
from subprocess import TimeoutExpired
import argparse
import uuid
import getpass
import atexit

HARD_CODED_AFFILIATIONS = {
    "192.168.1.11": ["192.168.1.12",],
    "192.168.1.12": ["192.168.1.11",],
    "192.168.1.13": ["192.168.1.11",],
    "192.168.1.15": ["192.168.1.16",],
    "192.168.1.16": ["192.168.1.15",],
}


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
    hostname=None, survival_time=None, workspace_dir=None, container_name=None
):
    docker_cmd = f"""docker run --privileged -d --network host --shm-size=8g --rm -v {workspace_dir}:{workspace_dir} -w {workspace_dir} -v /dataset:/dataset -v /model_zoo:/model_zoo --name {container_name} oneflow-test:$USER sleep {survival_time}
"""
    ssh_cmd = f"ssh {hostname} {docker_cmd}"
    subprocess.check_call(ssh_cmd, shell=True)


class DockerAgent:
    def __init__(self, port=None, authkey=None) -> None:
        self.listener = Listener(("localhost", port), authkey=authkey)
        self.env_proto_txt = None
        self.bash_tmp_file = None
        self.bash_proc = None
        print("[docker agent]", "initializing", {"port": port, "authkey": authkey})

    def __enter__(self):
        return self

    def run_bash_script_async(
        self,
        bash_script=None,
        timeout=None,
        this_host=None,
        remote_hosts=None,
        oneflow_wheel_path=None,
        oneflow_build_path=None,
        agent_port=None,
        agent_authkey=None,
        container_name=None,
    ):
        remote_hosts_str = ",".join(remote_hosts)
        assert os.path.exists(bash_script)
        log_dir = "./unittest-log-" + str(uuid.uuid4())
        ctrl_port = find_free_port()
        data_port = find_free_port()
        exports = f"""
export ONEFLOW_TEST_MASTER_PORT={ctrl_port}
export ONEFLOW_TEST_DATA_PORT={data_port}
export ONEFLOW_TEST_LOG_DIR={log_dir}
export ONEFLOW_TEST_NODE_LIST="{this_host},{remote_hosts_str}"
export ONEFLOW_WORKER_KEEP_LOG=1
export ONEFLOW_TEST_TMP_DIR="./distributed-tmp"
export NCCL_DEBUG=INFO
export ONEFLOW_TEST_WORKER_AGENT_PORT={agent_port}
export ONEFLOW_TEST_WORKER_AGENT_AUTHKEY={agent_authkey}
"""
        if oneflow_wheel_path:
            exports += f"export ONEFLOW_WHEEL_PATH={oneflow_wheel_path}\n"
        if oneflow_build_path:
            exports += f"export ONEFLOW_BUILD_DIR={oneflow_build_path}\n"
        bash_cmd = f"""set -ex
    {exports}
    bash {bash_script}
    """

        def get_docker_cmd(f, cmd):
            f_name = f.name
            f.write(cmd)
            f.flush()
            return f"docker run --privileged --network host --shm-size=8g --rm -v /tmp:/host/tmp:ro -v $PWD:$PWD -v $HOME:$HOME -w $PWD -v /dataset:/dataset -v /model_zoo:/model_zoo --name {container_name} oneflow-test:$USER bash /host{f_name}"

        f = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=True)
        run_docker_cmd = get_docker_cmd(f, bash_cmd)
        self.bash_tmp_file = f
        self.bash_proc = subprocess.Popen(run_docker_cmd, shell=True)

    def launch_workers(self):
        self.conn = self.listener.accept()
        self.env_proto_txt = self.conn.recv()
        print("[docker agent]", "[env proto]", self.env_proto_txt)
        # do_launch_workers
        print("[docker agent]", "sending ok")
        self.conn.send(b"ok")

    def block(self, timeout=None):
        self.bash_proc.communicate(timeout=timeout)
        assert self.bash_proc.returncode == 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build_docker_img", action="store_true", required=False, default=False
    )
    parser.add_argument("--debug", action="store_true", required=False, default=False)
    parser.add_argument("--bash_script", type=str, required=False)
    default_this_host = socket.gethostname()
    parser.add_argument(
        "--this_host", type=str, required=False, default=default_this_host
    )
    parser.add_argument("--remote_host", type=str, required=False)
    parser.add_argument("--oneflow_wheel_path", type=str, required=False, default=None)
    parser.add_argument("--oneflow_build_path", type=str, required=False, default=None)
    parser.add_argument("--timeout", type=int, required=False, default=6 * 60 * 60)
    args = parser.parse_args()

    assert bool(args.oneflow_wheel_path) != bool(args.oneflow_build_path)
    this_host = args.this_host
    this_host = resolve_hostname_hardcoded(this_host)

    remote_host = None
    if args.remote_host:
        assert len(args.remote_host.split(",")) == 1, "only support 2-nodes run for now"
        remote_host = args.remote_host
    else:
        affiliations = get_affiliations(this_host)
        assert (
            affiliations
        ), f"no affiliated node found for {this_host}, you should specify one"
        remote_host = affiliations[0]
        remote_host = socket.gethostbyname(remote_host)

    print(f"this_host: {this_host}, remote_host: {remote_host}", flush=True)
    sub_dir = str(uuid.uuid4())
    if args.debug:
        sub_dir = "debug"
    workspace_dir = os.path.join(
        os.path.expanduser("~"), "distributed_run_workspace", sub_dir
    )
    print("workspace_dir", workspace_dir)
    create_remote_workspace_dir(remote_host, workspace_dir)
    if args.oneflow_build_path:
        print("copying python_scripts dir")
        subprocess.check_call(
            f"rsync -azP --omit-dir-times --no-perms --no-group --include='*.py' --include='*.so' --exclude='__pycache__' --exclude='python_scripts/oneflow/include' --include='*/' --exclude='*' {args.oneflow_build_path}/python_scripts {remote_host}:{workspace_dir}",
            shell=True,
        )
    if args.build_docker_img:
        build_docker_img(remote_host, workspace_dir)
    assert args.bash_script
    agent_port = find_free_port()
    agent_authkey = str(uuid.uuid4())
    container_name = getpass.getuser() + "-distributed-run"
    remote_hosts = [remote_host]

    def exit_handler():
        print("removing local docker container:", container_name)
        subprocess.check_call(f"docker rm -f {container_name} || true", shell=True)
        for remote_host in remote_hosts:
            print(f"removing local docker container at {remote_host}:", container_name)
            subprocess.check_call(
                f"ssh {remote_host} docker rm -f {container_name} || true", shell=True,
            )

    atexit.register(exit_handler)
    launch_remote_container(
        hostname=remote_host,
        survival_time=args.timeout,
        workspace_dir=workspace_dir,
        container_name=container_name,
    )
    with DockerAgent(port=agent_port, authkey=agent_authkey.encode()) as agent:
        agent.run_bash_script_async(
            bash_script=args.bash_script,
            timeout=args.timeout,
            this_host=this_host,
            remote_hosts=[remote_host],
            oneflow_wheel_path=args.oneflow_wheel_path,
            oneflow_build_path=args.oneflow_build_path,
            agent_port=agent_port,
            agent_authkey=agent_authkey,
            container_name=container_name,
        )
        agent.launch_workers()
        agent.block(args.timeout)
        # TODO: remove container when exit
    # copy artifacts
    exit(0)
