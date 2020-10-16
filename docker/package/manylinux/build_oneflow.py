import os
import subprocess


def build_arg_env(env_var_name):
    val = os.getenv(env_var_name)
    return f"--build-arg {env_var_name}={val}"


def build_img(cuda_version, oneflow_src_dir, use_tuna=False, use_system_proxy=True):
    cudnn_version = 7
    if str(cuda_version).startswith("11"):
        cudnn_version = 8
    from_img = f"nvidia/cuda:{cuda_version}-cudnn{cudnn_version}-devel-centos7"
    img_tag = f"oneflow:manylinux2014-cuda{cuda_version}"
    tuna_build_arg = ""
    if use_tuna:
        tuna_build_arg = '--build-arg use_tuna_yum=0 --build-arg pip_args=""'
    proxy_build_args = []
    if use_system_proxy:
        for v in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            proxy_build_args.append(build_arg_env(v))
    proxy_build_arg = " ".join(proxy_build_args)
    cmd = f"docker build -f docker/package/manylinux/Dockerfile {proxy_build_arg} {tuna_build_arg} --build-arg from={from_img} -t {img_tag} ."
    print(cmd)
    subprocess.check_call(cmd, cwd=oneflow_src_dir, shell=True)
    return img_tag


def build_third_party():
    pass


def build_oneflow():
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", type=str, required=False, default="manylinux2014-build-cache"
    )
    parser.add_argument(
        "--wheel_house_dir", type=str, required=False, default="wheelhouse",
    )
    parser.add_argument(
        "--python_version", type=str, required=False, default="3.5, 3.6, 3.7, 3.8",
    )
    parser.add_argument(
        "--cuda_version", type=str, required=False, default="10.2",
    )
    parser.add_argument(
        "--extra_oneflow_cmake_args", type=str, required=False, default="",
    )
    parser.add_argument(
        "--oneflow_src_dir", type=str, required=False, default=os.getcwd(),
    )
    parser.add_argument(
        "--skip_third_party", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--skip_wheel", default=False, action="store_true", required=False
    )
    args = parser.parse_args()
    build_img(args.cuda_version, args.oneflow_src_dir)
