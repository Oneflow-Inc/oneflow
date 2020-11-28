set -ex
test_img_dir="$(dirname "${BASH_SOURCE[0]}")"
test_img_dir="$(realpath "${test_img_dir}")"
cd $test_img_dir
docker build --rm \
    -t oneflow-test:$USER .
