import os
import subprocess


def build_arg_env(env_var_name):
    val = os.getenv(env_var_name)
    return f"--build-arg {env_var_name}={val}"


def build_img(cuda_version, oneflow_src_dir, use_tuna, use_system_proxy, skip_img):
    cudnn_version = 7
    if str(cuda_version).startswith("11"):
        cudnn_version = 8
    from_img = f"nvidia/cuda:{cuda_version}-cudnn{cudnn_version}-devel-centos7"
    img_tag = f"oneflow:manylinux2014-cuda{cuda_version}"
    if skip_img == False:
        tuna_build_arg = ""
        if use_tuna:
            tuna_build_arg = '--build-arg use_tuna_yum=1 --build-arg pip_args="-i https://pypi.tuna.tsinghua.edu.cn/simple"'
        proxy_build_args = []
        if use_system_proxy:
            for v in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
                proxy_build_args.append(build_arg_env(v))
        proxy_build_arg = " ".join(proxy_build_args)
        cmd = f"docker build -f docker/package/manylinux/Dockerfile {proxy_build_arg} {tuna_build_arg} --build-arg from={from_img} -t {img_tag} ."
        print(cmd)
        subprocess.check_call(cmd, cwd=oneflow_src_dir, shell=True)
    return img_tag


def common_cmake_args(cache_dir):
    third_party_install_dir = os.path.join(cache_dir, "build-third-party-install")
    return f"-DCMAKE_BUILD_TYPE=Release -DBUILD_RDMA=ON -DTHIRD_PARTY_DIR={third_party_install_dir}"


def get_build_dir_arg(cache_dir, oneflow_src_dir):
    build_dir_real = os.path.join(cache_dir, "build")
    build_dir_mount = os.path.join(oneflow_src_dir, "build")
    return f"-v {build_dir_real}:{build_dir_mount}"


def build_third_party(img_tag, oneflow_src_dir, cache_dir, extra_oneflow_cmake_args):
    third_party_build_dir = os.path.join(cache_dir, "build-third-party")
    cmake_cmd = " ".join(
        [
            "cmake",
            common_cmake_args(cache_dir),
            "-DTHIRD_PARTY=ON, -DONEFLOW=OFF",
            extra_oneflow_cmake_args,
            oneflow_src_dir,
        ]
    )
    cwd = os.getcwd()
    pwd_arg = f"-v {cwd}:{cwd}"
    cache_dir_arg = f"-v {cache_dir}:{cache_dir}"
    build_dir_arg = get_build_dir_arg(cache_dir, oneflow_src_dir)
    docker_cmd = f"docker run --rm -it -v {oneflow_src_dir}:{oneflow_src_dir} {pwd_arg} {cache_dir_arg} {build_dir_arg} -w {third_party_build_dir} {img_tag} {cmake_cmd}"
    cmd = f"{docker_cmd} {cmake_cmd}"
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    cmd = f"{docker_cmd} make -j`nproc` prepare_oneflow_third_party"
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def build_oneflow():
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default=os.path.join(os.getcwd(), "manylinux2014-build-cache"),
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
    parser.add_argument(
        "--skip_img", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--use_tuna", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--use_system_proxy", default=False, action="store_true", required=False
    )
    parser.add_argument("--xla", default=False, action="store_true", required=False)
    parser.add_argument(
        "--aliyun_mirror", default=False, action="store_true", required=False
    )
    parser.add_argument("--cpu", default=False, action="store_true", required=False)
    args = parser.parse_args()
    img_tag = build_img(
        args.cuda_version,
        args.oneflow_src_dir,
        args.use_tuna,
        args.use_system_proxy,
        args.skip_img,
    )
    extra_oneflow_cmake_args = args.extra_oneflow_cmake_args
    if args.aliyun_mirror:
        extra_oneflow_cmake_args += " -DTHIRD_PARTY_MIRROR=aliyun"
    if args.cpu:
        extra_oneflow_cmake_args += " -DBUILD_CUDA=OFF"
    else:
        extra_oneflow_cmake_args += " -DBUILD_CUDA=ON"
    build_third_party(
        img_tag, args.oneflow_src_dir, args.cache_dir, extra_oneflow_cmake_args
    )
