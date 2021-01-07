import os
import subprocess
import socket
import tempfile
from contextlib import closing
from subprocess import TimeoutExpired
import argparse
import uuid

FIX_SSH_PERMISSION = """
mkdir -p /run/sshd
chown root ~/.ssh
chmod 700 ~/.ssh
chown root ~/.ssh/*
chmod 600 ~/.ssh/*
chmod 400 ~/.ssh/id_rsa
chmod 400 ~/.ssh/id_rsa.pub
chmod 600 ~/.ssh/config
"""

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


def make_dotssh(dotssh_dir):
    bash_cmd = f"""set -ex
rm -rf /root/.ssh
ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys
/etc/init.d/ssh start && \
    ssh-keyscan -H localhost >> /root/.ssh/known_hosts

cp -r /root/.ssh/* {dotssh_dir}
chmod 777 {dotssh_dir}
chmod 777 {dotssh_dir}/*
"""
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
        f_name = f.name
        f.write(bash_cmd)
        f.flush()
        subprocess.check_call(
            f"docker run --rm -v /tmp:/host/tmp -v {dotssh_dir}:{dotssh_dir} -w $PWD oneflow-test:$USER bash /host/{f_name}",
            shell=True,
        )
    config_content = """Host *
	StrictHostKeyChecking no
"""
    with open(os.path.join(dotssh_dir, "config"), "w") as f:
        f.write(config_content)


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


def launch_remote_container(
    hostname, docker_ssh_port, survival_time, dotssh_dir, workspace_dir
):
    subprocess.check_call(
        f"scp -r {dotssh_dir} {hostname}:{workspace_dir}/dotssh", shell=True
    )
    bash_cmd = f"""set -ex
{FIX_SSH_PERMISSION}
/usr/sbin/sshd -p {docker_ssh_port}
sleep {survival_time}
"""
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
        f_name = f.name
        f.write(bash_cmd)
        f.flush()
        subprocess.check_call(
            f"scp {f_name} {hostname}:{workspace_dir}/launch_ssh_server.sh", shell=True,
        )
    docker_cmd = f"""docker run --privileged --cidfile {workspace_dir}/worker.cid --network host --shm-size=8g --rm -v {workspace_dir}/dotssh:/root/.ssh -v {workspace_dir}:{workspace_dir} -w {workspace_dir} -v /dataset:/dataset -v /model_zoo:/model_zoo oneflow-test:$USER bash launch_ssh_server.sh
"""
    ssh_cmd = f"ssh {hostname} {docker_cmd}"
    print(ssh_cmd, flush=True)
    proc = subprocess.Popen(ssh_cmd, shell=True,)
    try:
        proc.wait(timeout=10)
        raise ValueError("sshd quit early, returncode:", proc.returncode)
    except TimeoutExpired:
        survival_time_min = survival_time / 60
        survival_time_min = int(survival_time_min)
        print(
            f"remote container launched, host: {hostname}, ssh port: {docker_ssh_port}, .ssh dir: {dotssh_dir}, survival: {survival_time_min} mins",
            flush=True,
        )


def run_bash_script(
    bash_script,
    timeout,
    ssh_port,
    dotssh_dir,
    this_host,
    remote_host,
    oneflow_worker_bin,
    oneflow_wheel_path,
):
    assert os.path.exists(bash_script)
    log_dir = "./unittest-log-" + str(uuid.uuid4())
    ctrl_port = find_free_port()
    data_port = find_free_port()
    exports = f"""
export ONEFLOW_TEST_CTRL_PORT={ctrl_port}
export ONEFLOW_TEST_DATA_PORT={data_port}
export ONEFLOW_TEST_SSH_PORT={ssh_port}
export ONEFLOW_TEST_LOG_DIR={log_dir}
export ONEFLOW_TEST_NODE_LIST="{this_host},{remote_host}"
export ONEFLOW_WORKER_KEEP_LOG=1
export NCCL_DEBUG=INFO
"""
    if oneflow_worker_bin:
        exports += f"export ONEFLOW_WORKER_BIN={oneflow_worker_bin}\n"
    if oneflow_wheel_path:
        exports += f"export ONEFLOW_WHEEL_PATH={oneflow_wheel_path}\n"
    bash_cmd = f"""set -ex
{exports}
rm -rf ~/.ssh
cp -r /dotssh ~/.ssh
{FIX_SSH_PERMISSION}
bash {bash_script}
"""
    artifact_cmd = f"""set -ex
{exports}
rm -rf ~/.ssh
cp -r /dotssh ~/.ssh
{FIX_SSH_PERMISSION}
mkdir -p oneflow_temp
rm -rf oneflow_temp/{remote_host}
scp -P {ssh_port} -r {remote_host}:~/oneflow_temp oneflow_temp/{remote_host}
rm -f oneflow_temp/{remote_host}/*/oneflow_worker
chmod -R o+w oneflow_temp
chmod -R o+r oneflow_temp
"""
    returncode = None

    def get_docker_cmd(f, cmd):
        f_name = f.name
        print(cmd, flush=True)
        f.write(cmd)
        f.flush()
        return f"docker run --privileged --network host --shm-size=8g --rm -v /tmp:/host/tmp -v $PWD:$PWD -v $HOME:$HOME -w $PWD -v {dotssh_dir}:/dotssh -v /dataset:/dataset -v /model_zoo:/model_zoo oneflow-test:$USER bash /host{f_name}"

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
        run_docker_cmd = get_docker_cmd(f, bash_cmd)
        returncode = subprocess.call(run_docker_cmd, shell=True, timeout=timeout)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
        artifact_docker_cmd = get_docker_cmd(f, artifact_cmd)
        subprocess.check_call(artifact_docker_cmd, shell=True, timeout=timeout)

    if returncode != 0:
        raise ValueError(run_docker_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch_remote_container", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--make_dotssh", action="store_true", required=False, default=False
    )
    parser.add_argument("--run", action="store_true", required=False, default=False)
    parser.add_argument(
        "--build_docker_img", action="store_true", required=False, default=False
    )
    parser.add_argument("--bash_script", type=str, required=False)
    default_this_host = socket.gethostname()
    parser.add_argument(
        "--this_host", type=str, required=False, default=default_this_host
    )
    parser.add_argument("--remote_host", type=str, required=False)
    default_dotssh_dir = os.path.expanduser("~/distributed_run_dotssh")
    parser.add_argument(
        "--dotssh_dir", type=str, required=False, default=default_dotssh_dir
    )
    parser.add_argument("--oneflow_worker_bin", type=str, required=False, default=None)
    parser.add_argument("--oneflow_wheel_path", type=str, required=False, default=None)
    parser.add_argument("--ssh_port", type=int, required=False, default=None)
    parser.add_argument("--timeout", type=int, required=False, default=6 * 60 * 60)
    args = parser.parse_args()

    ssh_port = None
    if args.ssh_port:
        ssh_port = args.ssh_port
    else:
        ssh_port = find_free_port()
    assert ssh_port
    if args.make_dotssh:
        make_dotssh(args.dotssh_dir)

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
    workspace_dir = os.path.join(
        os.path.expanduser("~"), "distributed_run_workspace", str(uuid.uuid4())
    )
    create_remote_workspace_dir(remote_host, workspace_dir)
    if args.launch_remote_container:
        launch_remote_container(remote_host, ssh_port, args.timeout, args.dotssh_dir)
    if args.build_docker_img:
        build_docker_img()
        build_docker_img(remote_host, workspace_dir)
    if args.run:
        launch_remote_container(
            remote_host, ssh_port, args.timeout, args.dotssh_dir, workspace_dir,
        )
        assert args.bash_script
        run_bash_script(
            args.bash_script,
            args.timeout,
            ssh_port,
            args.dotssh_dir,
            this_host,
            remote_host,
            args.oneflow_worker_bin,
            args.oneflow_wheel_path,
        )
        exit(0)
