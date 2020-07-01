set -xeu

pids=()

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}

ENABLE_USER_OP=True bash $src_dir/ci/test/1node_op_test.sh &
pids+=($!)

ENABLE_USER_OP=False bash $src_dir/ci/test/1node_op_test.sh &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done
