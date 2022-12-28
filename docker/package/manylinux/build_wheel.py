import os
import subprocess
import tempfile
from pathlib import Path
import getpass
import uuid


def get_arg_env(env_var_name: str, mode="run"):
    val = os.getenv(env_var_name)
    assert val, f"system environment variable {env_var_name} found empty"
    if mode == "run":
        return f"--env {env_var_name}={val}"
    elif mode == "build":
        return f"--build-arg {env_var_name}={val}"
    else:
        raise f"{mode} not supported"


def get_proxy_build_args():
    proxy_build_args = []
    if os.getenv("HTTP_PROXY"):
        for v in ["HTTP_PROXY", "HTTPS_PROXY"]:
            proxy_build_args.append(get_arg_env(v, mode="build"))
    if os.getenv("http_proxy"):
        for v in ["http_proxy", "https_proxy"]:
            proxy_build_args.append(get_arg_env(v, mode="build"))
    return " ".join(proxy_build_args)


def get_proxy_env_args():
    proxy_build_args = []
    if os.getenv("HTTP_PROXY"):
        for v in ["HTTP_PROXY", "HTTPS_PROXY"]:
            proxy_build_args.append(get_arg_env(v))
    if os.getenv("http_proxy"):
        for v in ["http_proxy", "https_proxy"]:
            proxy_build_args.append(get_arg_env(v))
    return " ".join(proxy_build_args)


def build_img(
    cuda_version,
    oneflow_src_dir,
    use_aliyun_mirror,
    use_tuna,
    use_system_proxy,
    img_tag,
    dry,
):
    cudnn_version = 7
    if str(cuda_version).startswith("11"):
        cudnn_version = 8
    cuda_version_img = cuda_version
    if cuda_version == "11.2":
        cuda_version_img = "11.2.2"
    if cuda_version == "11.1":
        cuda_version_img = "11.1.1"
    if cuda_version == "11.0":
        cuda_version_img = "11.0.3"
    from_img = f"nvidia/cuda:{cuda_version_img}-cudnn{cudnn_version}-devel-centos7"
    tuna_build_arg = ""
    if use_tuna:
        tuna_build_arg = '--build-arg use_tuna_yum=1 --build-arg pip_args="-i https://mirrors.aliyun.com/pypi/simple"'
    if use_aliyun_mirror:
        tuna_build_arg += ' --build-arg bazel_url="https://oneflow-static.oss-cn-beijing.aliyuncs.com/deps/bazel-3.4.1-linux-x86_64"'

    proxy_build_arg = get_proxy_build_args() if use_system_proxy else ""
    cmd = f"docker build -f docker/package/manylinux/Dockerfile {proxy_build_arg} {tuna_build_arg} --build-arg from={from_img} -t {img_tag} ."
    print(cmd)
    if dry == False:
        subprocess.check_call(cmd, cwd=oneflow_src_dir, shell=True)


def common_cmake_args(cache_dir=None, extra_oneflow_cmake_args=None):
    assert cache_dir
    ret = ""
    if (
        not extra_oneflow_cmake_args
        or "-DCMAKE_BUILD_TYPE" not in extra_oneflow_cmake_args
    ):
        ret += " -DCMAKE_BUILD_TYPE=Release"
    if not extra_oneflow_cmake_args or "-DBUILD_RDMA" not in extra_oneflow_cmake_args:
        ret += " -DBUILD_RDMA=ON"
    third_party_install_dir = os.path.join(cache_dir, "build-third-party-install")
    ret += f" -DTHIRD_PARTY_DIR={third_party_install_dir}"
    return ret


def get_build_dir_arg(cache_dir, oneflow_src_dir):
    return ""
    build_dir_real = os.path.join(cache_dir, "build")
    build_dir_mount = os.path.join(oneflow_src_dir, "build")
    return f"-v {build_dir_real}:{build_dir_mount}"


def force_rm_dir(dir_to_clean):
    print("cleaning:", dir_to_clean)
    assert dir_to_clean
    clean_cmd = f"docker run --network=host --rm -v {dir_to_clean}:{dir_to_clean} -w {dir_to_clean} busybox rm -rf {dir_to_clean}/*"
    subprocess.check_call(clean_cmd, shell=True)


def create_tmp_bash_and_run(docker_cmd, img, bash_cmd, bash_args, bash_wrap, dry):
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as wrapper_f:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
            w_name = "/host" + wrapper_f.name
            f_name = "/host" + f.name
            bash_cmd = "PATH=/opt/python/cp37-cp37m/bin:$PATH\n" + bash_cmd
            f.write(bash_cmd)
            f.flush()
            wrapped = f"""
{bash_wrap}
bash {bash_args} {f_name}
"""
            wrapper_f.write(wrapped)
            wrapper_f.flush()

            print("=" * 5 + f"bash_cmd: {f_name}" + "=" * 5)
            print(bash_cmd)
            print("=" * 5 + f"bash_cmd: {f_name}" + "=" * 5)

            print("=" * 5 + f"wrapped: {w_name}" + "=" * 5)
            print(wrapped)
            print("=" * 5 + f"wrapped: {w_name}" + "=" * 5)

            docker_cmd = f"{docker_cmd} -v /tmp:/host/tmp {img}"
            cmd = f"{docker_cmd} bash {bash_args} {w_name}"
            print(cmd)
            if dry:
                print("dry run, skipping")
            else:
                subprocess.check_call(cmd, shell=True)


def get_common_docker_args(
    oneflow_src_dir=None,
    cache_dir=None,
    current_dir=None,
    house_dir=None,
    use_system_proxy=True,
    inplace=False,
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
    proxy_env_arg = get_proxy_env_args() if use_system_proxy else ""
    inplace_attr = ""
    if inplace == False:
        inplace_attr = ":ro"
    cache_dir_args = " ".join(
        [
            f"-v {os.path.join(cache_dir, 'ccache')}:/root/.ccache",
            f"-v {os.path.join(cache_dir, 'local')}:/root/.local",
            f"-v {os.path.join(cache_dir, 'cache')}:/root/.cache",
        ]
    )
    return f"{cache_dir_args} -v {oneflow_src_dir}:{oneflow_src_dir}{inplace_attr} {proxy_env_arg} {pwd_arg} {house_dir_arg} {cache_dir_arg} {build_dir_arg} -w {current_dir} --shm-size=8g"


def get_python_dir(inplace=True, oneflow_src_dir=None, cache_dir=None):
    if inplace:
        assert oneflow_src_dir
        return os.path.join(oneflow_src_dir, "python")
    else:
        assert cache_dir
        return os.path.join(cache_dir, "python")


def build_third_party(
    img_tag,
    oneflow_src_dir,
    cache_dir,
    extra_oneflow_cmake_args,
    extra_docker_args,
    bash_args,
    bash_wrap,
    dry,
    use_system_proxy,
    inplace,
):
    third_party_build_dir = os.path.join(cache_dir, "build-third-party")
    oneflow_python_dir = get_python_dir(
        inplace=inplace, oneflow_src_dir=oneflow_src_dir, cache_dir=cache_dir
    )
    if inplace:
        inplace_arg = ""
        oneflow_python_dir_cmd = ""
    else:
        inplace_arg = f"-DONEFLOW_PYTHON_DIR={oneflow_python_dir}"
        oneflow_python_dir_cmd = f"""
        rm -rf {oneflow_python_dir}
        cp -r {oneflow_src_dir}/python {oneflow_python_dir}
        cd {oneflow_python_dir}
        git init
        git clean -nXd
        git clean -fXd
        cd -
        """
    cmake_cmd = " ".join(
        [
            "cmake",
            common_cmake_args(
                cache_dir=cache_dir, extra_oneflow_cmake_args=extra_oneflow_cmake_args
            ),
            "-DTHIRD_PARTY=ON -DONEFLOW=OFF",
            extra_oneflow_cmake_args,
            oneflow_src_dir,
            inplace_arg,
        ]
    )

    bash_cmd = f"""set -ex
export ONEFLOW_PYTHON_DIR={oneflow_python_dir}
{oneflow_python_dir_cmd}
export PATH="$PATH:$(dirname {get_python_bin('3.6')})"
export PYTHON_BIN_PATH={get_python_bin('3.6')}
$PYTHON_BIN_PATH -m pip install -i https://mirrors.aliyun.com/pypi/simple --user -r {os.path.join(oneflow_src_dir, "ci/fixed-dev-requirements.txt")}
$PYTHON_BIN_PATH -c "from __future__ import print_function;import numpy; print(numpy.get_include());"
{cmake_cmd}
cmake --build . -j `nproc` --target oneflow_deps
"""
    common_docker_args = get_common_docker_args(
        oneflow_src_dir=oneflow_src_dir,
        cache_dir=cache_dir,
        current_dir=third_party_build_dir,
        use_system_proxy=use_system_proxy,
        inplace=inplace,
    )
    docker_cmd = (
        f"docker run --network=host {extra_docker_args} --rm {common_docker_args}"
    )
    create_tmp_bash_and_run(docker_cmd, img_tag, bash_cmd, bash_args, bash_wrap, dry)


def get_python_bin(version):
    assert version in ["3.5", "3.6", "3.7", "3.8", "3.9"]
    py_ver = "".join(version.split("."))
    py_abi = f"cp{py_ver}-cp{py_ver}"
    if version in ["3.5", "3.6", "3.7"]:
        py_abi = f"{py_abi}m"
    py_root = f"/opt/python/{py_abi}"
    py_bin = f"{py_root}/bin/python"
    return py_bin


def build_oneflow(
    img_tag,
    oneflow_src_dir,
    cache_dir,
    extra_oneflow_cmake_args,
    extra_docker_args,
    python_version,
    skip_wheel,
    package_name,
    house_dir,
    bash_args,
    bash_wrap,
    dry,
    use_system_proxy,
    enter_bash,
    skip_audit,
    inplace,
):
    oneflow_build_dir = os.path.join(cache_dir, "build-oneflow")
    python_bin = get_python_bin(python_version)
    oneflow_python_dir = get_python_dir(
        inplace=inplace, oneflow_src_dir=oneflow_src_dir, cache_dir=cache_dir
    )
    if inplace:
        inplace_arg = ""
    else:
        inplace_arg = f"-DONEFLOW_PYTHON_DIR={oneflow_python_dir}"
    cmake_cmd = " ".join(
        [
            "cmake",
            common_cmake_args(
                cache_dir=cache_dir, extra_oneflow_cmake_args=extra_oneflow_cmake_args
            ),
            "-DTHIRD_PARTY=OFF -DONEFLOW=ON",
            extra_oneflow_cmake_args,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=1",
            f"-DPython3_EXECUTABLE={python_bin}",
            f"-DCODEGEN_PYTHON_EXECUTABLE={get_python_bin('3.6')}",
            oneflow_src_dir,
            inplace_arg,
        ]
    )
    common_docker_args = get_common_docker_args(
        oneflow_src_dir=oneflow_src_dir,
        cache_dir=cache_dir,
        current_dir=oneflow_build_dir,
        house_dir=house_dir,
        use_system_proxy=use_system_proxy,
        inplace=inplace,
    )
    docker_cmd = (
        f"docker run --network=host --rm {common_docker_args} {extra_docker_args}"
    )
    if enter_bash:
        docker_cmd += " -it"
    bash_cmd = f"""set -ex
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
export ONEFLOW_SRC_DIR={oneflow_src_dir}
export ONEFLOW_CMAKE_CMD="{cmake_cmd}"
{python_bin} -m pip install -i https://mirrors.aliyun.com/pypi/simple --user -r {os.path.join(oneflow_src_dir, "ci/fixed-dev-requirements.txt")}
"""
    if enter_bash:
        bash_cmd += "\nbash"
    else:
        bash_cmd += f"""
cd {oneflow_python_dir}
git clean -nXd -e \!oneflow/include -e \!oneflow/include/**
git clean -fXd -e \!oneflow/include -e \!oneflow/include/**
cd -
{cmake_cmd}
cmake --build . -j `nproc`
"""
    if skip_wheel or enter_bash:
        pass
    else:
        bash_cmd += f"""
cd {oneflow_python_dir}
{python_bin} setup.py bdist_wheel -d /tmp/tmp_wheel --package_name {package_name}
cd -
"""
    if skip_wheel == False:
        if skip_audit:
            bash_cmd += f"""
    cp /tmp/tmp_wheel/*.whl {house_dir}
    """
        else:
            bash_cmd += f"""
    auditwheel repair /tmp/tmp_wheel/*.whl --wheel-dir {house_dir}
    """
    return create_tmp_bash_and_run(
        docker_cmd, img_tag, bash_cmd, bash_args, bash_wrap, dry
    )


def is_img_existing(tag):
    returncode = subprocess.run(
        f"docker image inspect {tag}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    if returncode == 0:
        return True
    else:
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_img_tag", type=str, required=False, default=None,
    )
    parser.add_argument(
        "--container_name", type=str, required=False, default=None,
    )
    parser.add_argument(
        "--cache_dir", type=str, required=False, default=None,
    )
    default_wheel_house_dir = os.path.join(os.getcwd(), "wheelhouse")
    parser.add_argument(
        "--wheel_house_dir", type=str, required=False, default=default_wheel_house_dir,
    )
    parser.add_argument("--python_version", type=str, required=True)
    parser.add_argument(
        "--cuda_version", type=str, required=False, default="10.2",
    )
    parser.add_argument(
        "--package_name", type=str, required=False, default="oneflow",
    )
    parser.add_argument(
        "--extra_oneflow_cmake_args", action="append", nargs="+", default=[]
    )
    parser.add_argument(
        "--extra_docker_args", type=str, required=False, default="",
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
        "--skip_audit", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--build_img", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--use_tuna", default=False, action="store_true", required=False
    )
    parser.add_argument("--dry", default=False, action="store_true", required=False)
    parser.add_argument(
        "--use_system_proxy", default=False, action="store_true", required=False
    )
    parser.add_argument("--mlir", default=False, action="store_true", required=False)
    parser.add_argument("--gcc4", default=False, action="store_true", required=False)
    parser.add_argument("--gcc7", default=False, action="store_true", required=False)
    parser.add_argument("--gcc9", default=False, action="store_true", required=False)
    parser.add_argument(
        "--use_aliyun_mirror", default=False, action="store_true", required=False
    )
    parser.add_argument("--cpu", default=False, action="store_true", required=False)
    parser.add_argument("--bash", default=False, action="store_true", required=False)
    parser.add_argument("--inplace", default=False, action="store_true", required=False)
    parser.add_argument(
        "--shared_lib", default=False, action="store_true", required=False
    )
    parser.add_argument("--retry", default=0, type=int)
    args = parser.parse_args()
    if args.skip_img:
        "Arg skip_img is deprecated. Setting it has no effect. If you want to build image, use --build_img"
    if args.skip_wheel:
        args.skip_audit = True
    print("args.extra_oneflow_cmake_args", args.extra_oneflow_cmake_args)
    assert args.package_name
    extra_oneflow_cmake_args = " ".join(
        [" ".join(l) for l in args.extra_oneflow_cmake_args]
    )
    if (not args.gcc4) and (not args.gcc7) and (not args.gcc9):
        args.gcc7 = True
    cuda_versions = []
    if args.use_aliyun_mirror:
        extra_oneflow_cmake_args += " -DTHIRD_PARTY_MIRROR=aliyun"
    if args.shared_lib:
        extra_oneflow_cmake_args += " -DBUILD_SHARED_LIBS=ON"
    if args.cpu:
        extra_oneflow_cmake_args += " -DBUILD_CUDA=OFF"
        cuda_versions = ["10.2"]
    else:
        extra_oneflow_cmake_args += " -DBUILD_CUDA=ON"
    cuda_versions = args.cuda_version.split(",")
    cuda_versions = [v.strip() for v in cuda_versions]
    if args.mlir:
        extra_oneflow_cmake_args += " -DWITH_MLIR=ON"
    else:
        extra_oneflow_cmake_args += " -DWITH_MLIR=Off"
    for cuda_version in cuda_versions:

        cache_dir = None

        def build():
            img_tag = None
            img_prefix = f"oneflow-manylinux2014-cuda{cuda_version}"
            user = getpass.getuser()
            versioned_img_tag = f"{img_prefix}:0.1"
            if cuda_version in ["11.0", "11.1"]:
                versioned_img_tag = f"{img_prefix}:0.2"
            enforced_oneflow_cmake_args = ""
            enforced_oneflow_cmake_args += " -DBUILD_TESTING=ON"
            if float(cuda_version) >= 11:
                assert (
                    "CUDNN_STATIC" not in extra_oneflow_cmake_args
                ), "CUDNN_STATIC will be set to OFF if cuda_version > 11"
                enforced_oneflow_cmake_args += " -DCUDNN_STATIC=OFF"
            extra_docker_args = args.extra_docker_args
            if not args.container_name:
                args.container_name = f"manylinux-build-run-by-{getpass.getuser()}"
            assert args.container_name
            subprocess.call(
                f"docker rm -f {args.container_name}", shell=True,
            )
            extra_docker_args += f" --name {args.container_name}"
            user_img_tag = f"{img_prefix}:{user}"
            inc_img_tag = f"oneflowinc/{versioned_img_tag}"
            img_tag = inc_img_tag
            if args.build_img:
                img_tag = user_img_tag
            elif args.custom_img_tag:
                img_tag = args.custom_img_tag
            else:
                if is_img_existing(versioned_img_tag):
                    img_tag = versioned_img_tag
                elif is_img_existing(inc_img_tag):
                    img_tag = inc_img_tag
                else:
                    raise ValueError(
                        f"img not found, please run 'docker pull {inc_img_tag}'"
                    )
            assert img_tag is not None
            print("using", img_tag)
            if args.build_img:
                build_img(
                    cuda_version,
                    args.oneflow_src_dir,
                    args.use_aliyun_mirror,
                    args.use_tuna,
                    args.use_system_proxy,
                    img_tag,
                    args.dry,
                )
            bash_args = ""
            bash_wrap = ""
            if args.gcc4:
                bash_wrap = "gcc --version"
            elif args.gcc7:
                bash_wrap = """
source scl_source enable devtoolset-7
gcc --version
"""
            elif args.gcc9:
                bash_wrap = """
source scl_source enable devtoolset-9
gcc --version
"""
            else:
                raise ValueError("either one in gcc4, gcc7, gcc9 must be enabled")

            global cache_dir
            if args.cache_dir:
                cache_dir = args.cache_dir
            else:
                cache_dir = os.path.join(os.getcwd(), "manylinux2014-build-cache")
                sub_dir = cuda_version
                if args.mlir:
                    sub_dir += "-mlir"
                if args.gcc4:
                    sub_dir += "-gcc4"
                if args.gcc7:
                    sub_dir += "-gcc7"
                if args.gcc9:
                    sub_dir += "-gcc9"
                if args.cpu:
                    assert len(cuda_versions) == 1
                    sub_dir += "-cpu"
                if args.shared_lib:
                    sub_dir += "-shared"
                cache_dir = os.path.join(cache_dir, sub_dir)
            if args.build_img:
                return
            if args.skip_third_party == False:
                build_third_party(
                    img_tag,
                    args.oneflow_src_dir,
                    cache_dir,
                    extra_oneflow_cmake_args + enforced_oneflow_cmake_args,
                    extra_docker_args,
                    bash_args,
                    bash_wrap,
                    args.dry,
                    args.use_system_proxy,
                    args.inplace,
                )
            print(cuda_version.split("."))
            cuda_version_literal = "".join(cuda_version.split(".")[:2])
            assert len(cuda_version_literal) == 3
            python_versions = args.python_version.split(",")
            python_versions = [pv.strip() for pv in python_versions]
            for python_version in python_versions:
                print("building for python version:", python_version)
                build_oneflow(
                    img_tag,
                    args.oneflow_src_dir,
                    cache_dir,
                    extra_oneflow_cmake_args + enforced_oneflow_cmake_args,
                    extra_docker_args,
                    python_version,
                    args.skip_wheel,
                    args.package_name,
                    args.wheel_house_dir,
                    bash_args,
                    bash_wrap,
                    args.dry,
                    args.use_system_proxy,
                    args.bash,
                    args.skip_audit,
                    args.inplace,
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
