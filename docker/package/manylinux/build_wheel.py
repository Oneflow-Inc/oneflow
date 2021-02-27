import os
import subprocess
import tempfile
from pathlib import Path


def build_arg_env(env_var_name: str):
    val = os.getenv(env_var_name)
    assert val, f"system environment variable {env_var_name} found empty"
    return f"--build-arg {env_var_name}={val}"


def build_img(
    cuda_version,
    oneflow_src_dir,
    use_aliyun_mirror,
    use_tuna,
    use_system_proxy,
    img_tag,
):
    cudnn_version = 7
    if str(cuda_version).startswith("11"):
        cudnn_version = 8
    from_img = f"nvidia/cuda:{cuda_version}-cudnn{cudnn_version}-devel-centos7"
    tuna_build_arg = ""
    if use_tuna:
        tuna_build_arg = '--build-arg use_tuna_yum=1 --build-arg pip_args="-i https://mirrors.aliyun.com/pypi/simple"'
    if use_aliyun_mirror:
        tuna_build_arg += ' --build-arg bazel_url="https://oneflow-static.oss-cn-beijing.aliyuncs.com/deps/bazel-3.4.1-linux-x86_64"'
    proxy_build_args = []
    if use_system_proxy:
        if os.getenv("HTTP_PROXY"):
            for v in ["HTTP_PROXY", "HTTPS_PROXY"]:
                proxy_build_args.append(build_arg_env(v))
        if os.getenv("http_proxy"):
            for v in ["http_proxy", "https_proxy"]:
                proxy_build_args.append(build_arg_env(v))
    proxy_build_arg = " ".join(proxy_build_args)
    cmd = f"docker build -f docker/package/manylinux/Dockerfile {proxy_build_arg} {tuna_build_arg} --build-arg from={from_img} -t {img_tag} ."
    print(cmd)
    subprocess.check_call(cmd, cwd=oneflow_src_dir, shell=True)


def common_cmake_args(cache_dir):
    third_party_install_dir = os.path.join(cache_dir, "build-third-party-install")
    return f"-DCMAKE_BUILD_TYPE=Release -DBUILD_RDMA=ON -DTHIRD_PARTY_DIR={third_party_install_dir}"


def get_build_dir_arg(cache_dir, oneflow_src_dir):
    build_dir_real = os.path.join(cache_dir, "build")
    build_dir_mount = os.path.join(oneflow_src_dir, "build")
    return f"-v {build_dir_real}:{build_dir_mount}"


def force_rm_dir(dir_to_clean):
    print("cleaning:", dir_to_clean)
    assert dir_to_clean
    clean_cmd = f"docker run --rm -v {dir_to_clean}:{dir_to_clean} -w {dir_to_clean} busybox rm -rf {dir_to_clean}/*"
    subprocess.check_call(clean_cmd, shell=True)


def create_tmp_bash_and_run(docker_cmd, img, bash_cmd, bash_args, bash_wrap, dry):
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as wrapper_f:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
            w_name = "/host" + wrapper_f.name
            f_name = "/host" + f.name
            bash_cmd = "PATH=/opt/python/cp37-cp37m/bin:$PATH\n" + bash_cmd
            f.write(bash_cmd)
            f.flush()
            wrapper_f.write(
                f"""{bash_wrap}
bash {bash_args} {f_name}
"""
            )
            wrapper_f.flush()
            print(bash_cmd)
            docker_cmd = f"{docker_cmd} -v /tmp:/host/tmp {img}"
            cmd = f"{docker_cmd} bash {bash_args} {w_name}"
            print(cmd)
            if dry:
                print("dry run, skipping")
            else:
                subprocess.check_call(cmd, shell=True)


def get_common_docker_args(
    oneflow_src_dir=None, cache_dir=None, current_dir=None, house_dir=None
):
    root = Path(cache_dir)
    child = Path(current_dir)
    assert root in child.parents
    cwd = os.getcwd()
    pwd_arg = f"-v {cwd}:{cwd}"
    cache_dir_arg = f"-v {cache_dir}:{cache_dir}"
    house_dir_arg = ""
    if house_dir:
        house_dir_arg = f"-v {house_dir}:{house_dir}"
    build_dir_arg = get_build_dir_arg(cache_dir, oneflow_src_dir)
    return f"-v {oneflow_src_dir}:{oneflow_src_dir} {pwd_arg} {house_dir_arg} {cache_dir_arg} {build_dir_arg} -w {current_dir}"


def build_third_party(
    img_tag,
    oneflow_src_dir,
    cache_dir,
    extra_oneflow_cmake_args,
    bash_args,
    bash_wrap,
    dry,
):
    third_party_build_dir = os.path.join(cache_dir, "build-third-party")
    cmake_cmd = " ".join(
        [
            "cmake",
            common_cmake_args(cache_dir),
            "-DTHIRD_PARTY=ON -DONEFLOW=OFF",
            extra_oneflow_cmake_args,
            oneflow_src_dir,
        ]
    )

    bash_cmd = f"""set -ex
export TEST_TMPDIR={cache_dir}/bazel_cache
{cmake_cmd}
make -j`nproc` prepare_oneflow_third_party
"""
    common_docker_args = get_common_docker_args(
        oneflow_src_dir=oneflow_src_dir,
        cache_dir=cache_dir,
        current_dir=third_party_build_dir,
    )
    docker_cmd = f"docker run --rm {common_docker_args}"
    create_tmp_bash_and_run(docker_cmd, img_tag, bash_cmd, bash_args, bash_wrap, dry)


def get_python_bin(version):
    assert version in ["3.5", "3.6", "3.7", "3.8"]
    py_ver = "".join(version.split("."))
    py_abi = f"cp{py_ver}-cp{py_ver}"
    if py_ver != "38":
        py_abi = f"{py_abi}m"
    py_root = f"/opt/python/{py_abi}"
    py_bin = f"{py_root}/bin/python"
    return py_bin


def build_oneflow(
    img_tag,
    oneflow_src_dir,
    cache_dir,
    extra_oneflow_cmake_args,
    python_version,
    skip_wheel,
    package_name,
    house_dir,
    bash_args,
    bash_wrap,
    dry,
):
    oneflow_build_dir = os.path.join(cache_dir, "build-oneflow")
    python_bin = get_python_bin(python_version)
    cmake_cmd = " ".join(
        [
            "cmake",
            common_cmake_args(cache_dir),
            "-DTHIRD_PARTY=OFF -DONEFLOW=ON",
            extra_oneflow_cmake_args,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=1",
            f"-DPython3_EXECUTABLE={python_bin}",
            oneflow_src_dir,
        ]
    )
    common_docker_args = get_common_docker_args(
        oneflow_src_dir=oneflow_src_dir,
        cache_dir=cache_dir,
        current_dir=oneflow_build_dir,
        house_dir=house_dir,
    )
    docker_cmd = f"docker run --rm {common_docker_args}"
    bash_cmd = f"""set -ex
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/lib:$LD_LIBRARY_PATH
{cmake_cmd}
cmake --build . -j `nproc`
"""
    if skip_wheel:
        return 0
    else:
        bash_cmd += f"""
rm -rf {oneflow_build_dir}/python_scripts/*.egg-info
cd {oneflow_src_dir}
rm -rf build/*
{python_bin} setup.py bdist_wheel -d /tmp/tmp_wheel --build_dir {oneflow_build_dir} --package_name {package_name}
auditwheel repair /tmp/tmp_wheel/*.whl --wheel-dir {house_dir}
"""
        return create_tmp_bash_and_run(
            docker_cmd, img_tag, bash_cmd, bash_args, bash_wrap, dry
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_img_tag", type=str, required=False, default=None,
    )
    parser.add_argument(
        "--cache_dir", type=str, required=False, default=None,
    )
    default_wheel_house_dir = os.path.join(os.getcwd(), "wheelhouse")
    parser.add_argument(
        "--wheel_house_dir", type=str, required=False, default=default_wheel_house_dir,
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
    parser.add_argument("--dry", default=False, action="store_true", required=False)
    parser.add_argument(
        "--use_system_proxy", default=False, action="store_true", required=False
    )
    parser.add_argument("--xla", default=False, action="store_true", required=False)
    parser.add_argument(
        "--use_aliyun_mirror", default=False, action="store_true", required=False
    )
    parser.add_argument("--cpu", default=False, action="store_true", required=False)
    parser.add_argument("--retry", default=1, type=int)
    args = parser.parse_args()
    extra_oneflow_cmake_args = args.extra_oneflow_cmake_args

    cuda_versions = []
    if args.use_aliyun_mirror:
        extra_oneflow_cmake_args += " -DTHIRD_PARTY_MIRROR=aliyun"
    if args.cpu:
        extra_oneflow_cmake_args += " -DBUILD_CUDA=OFF"
        cuda_versions = ["10.2"]
    else:
        extra_oneflow_cmake_args += " -DBUILD_CUDA=ON"
    cuda_versions = args.cuda_version.split(",")
    cuda_versions = [v.strip() for v in cuda_versions]
    if args.xla:
        extra_oneflow_cmake_args += " -DWITH_XLA=ON"
    else:
        extra_oneflow_cmake_args += " -DWITH_XLA=Off"
    if args.xla == True and args.cpu == True:
        raise ValueError("flag xla can't coexist with flag cpu")
    for cuda_version in cuda_versions:

        cache_dir = None

        def build():
            img_tag = None
            skip_img = args.skip_img
            if args.custom_img_tag:
                img_tag = args.custom_img_tag
                skip_img = True
            else:
                img_tag = f"oneflow:manylinux2014-cuda{cuda_version}"
            if skip_img == False:
                build_img(
                    cuda_version,
                    args.oneflow_src_dir,
                    args.use_aliyun_mirror,
                    args.use_tuna,
                    args.use_system_proxy,
                    img_tag,
                )
            bash_args = ""
            if args.xla:
                bash_args = "-l"
            bash_wrap = ""
            if args.xla:
                bash_wrap = """
source scl_source enable devtoolset-7
gcc --version
"""
            else:
                bash_wrap = "gcc --version"

            global cache_dir
            if args.cache_dir:
                cache_dir = args.cache_dir
            else:
                cache_dir = os.path.join(os.getcwd(), "manylinux2014-build-cache")
                sub_dir = cuda_version
                if args.xla:
                    sub_dir += "-xla"
                if args.cpu:
                    assert len(cuda_versions) == 1
                    sub_dir = "cpu"
                cache_dir = os.path.join(cache_dir, sub_dir)
            if args.skip_third_party == False:
                build_third_party(
                    img_tag,
                    args.oneflow_src_dir,
                    cache_dir,
                    extra_oneflow_cmake_args,
                    bash_args,
                    bash_wrap,
                    args.dry,
                )
            cuda_version_literal = "".join(cuda_version.split("."))
            assert len(cuda_version_literal) == 3
            python_versions = args.python_version.split(",")
            python_versions = [pv.strip() for pv in python_versions]
            package_name = None
            if args.cpu:
                package_name = "oneflow_cpu"
            else:
                package_name = f"oneflow_cu{cuda_version_literal}"
                if args.xla:
                    package_name += "_xla"
            for python_version in python_versions:
                build_oneflow(
                    img_tag,
                    args.oneflow_src_dir,
                    cache_dir,
                    extra_oneflow_cmake_args,
                    python_version,
                    args.skip_wheel,
                    package_name,
                    args.wheel_house_dir,
                    bash_args,
                    bash_wrap,
                    args.dry,
                )

        try:
            build()
        except subprocess.CalledProcessError as e:
            print("failed: ", e.cmd, e.args)
            if cache_dir and args.retry > 0:
                print("clean: ", cache_dir, flush=True)
                print("start retrying...", flush=True)
                if args.dry:
                    pass
                else:
                    force_rm_dir(cache_dir)
                build()
            else:
                exit(1)
