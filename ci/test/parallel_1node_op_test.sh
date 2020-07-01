set -xeu

pids=()

docker run --shm-size=8g --rm \
    -w $PWD -v $PWD:$PWD -v /dataset:/dataset -v /model_zoo:/model_zoo \
    --env ENABLE_USER_OP=True \
    oneflow-test bash ci/test/exe_test.sh &
pids+=($!)

docker run --shm-size=8g --rm \
    -w $PWD -v $PWD:$PWD -v /dataset:/dataset -v /model_zoo:/model_zoo \
    --env ENABLE_USER_OP=True \
    oneflow-test bash ci/test/1node_op_test.sh &
pids+=($!)

docker run --shm-size=8g --rm \
    -w $PWD -v $PWD:$PWD -v /dataset:/dataset -v /model_zoo:/model_zoo \
    --env ENABLE_USER_OP=False \
    oneflow-test bash ci/test/1node_op_test.sh
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done
