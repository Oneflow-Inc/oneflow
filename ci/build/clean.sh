set -ex
tmp_dir=${ONEFLOW_CI_TMP_DIR:-"$HOME/ci-tmp"}
docker run --rm \
    -v $tmp_dir:/ci-tmp \
    -w $tmp_dir:/ci-tmp busybox rm -rf /ci-tmp/wheelhouse
docker run --rm -v $PWD:/p -w /p busybox rm -rf tmp_wheel
docker run --rm -v $PWD:/p -w /p busybox rm -rf build
