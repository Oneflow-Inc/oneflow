import subprocess


def launch_remote_container(hostname, docker_ssh_port, survival_time):
    bash_cmd = f"mkdir -p /run/sshd && /usr/sbin/sshd -p {docker_ssh_port} && sleep {survival_time}"
    cmd = f"""
docker docker run --shm-size=8g --rm -v /dataset:/dataset -v /model_zoo:/model_zoo bash -c "{bash_cmd}"
"""
    subprocess.Popen(
        cmd,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=cuda_visible_devices,
            ONEFLOW_TEST_CTRL_PORT=str(find_free_port()),
            ONEFLOW_TEST_LOG_DIR=("./unittest-log-" + str(uuid.uuid4())),
        ),
        shell=True,
    )
