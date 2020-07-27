set -ex
docker run --rm \
    -v $HOME/ci-tmp:/ci-tmp \
    -w $HOME/ci-tmp:/ci-tmp busybox rm -rf /ci-tmp/wheelhouse
