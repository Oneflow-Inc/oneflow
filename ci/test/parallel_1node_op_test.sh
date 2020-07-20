set -xeu

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}

bash $src_dir/ci/test/1node_op_test.sh
