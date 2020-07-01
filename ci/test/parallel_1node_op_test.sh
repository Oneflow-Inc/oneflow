set -xeu

pids=()

docker run --shm-size=8g --rm \
    -w $PWD -v $PWD:$PWD -v /dataset:/dataset -v /model_zoo:/model_zoo \
    --env ENABLE_USER_OP=True \
    oneflow-test bash ci/test/1node_op_test.sh &
pid_with_user_op=($!)

docker run --shm-size=8g --rm \
    -w $PWD -v $PWD:$PWD -v /dataset:/dataset -v /model_zoo:/model_zoo \
    --env ENABLE_USER_OP=False \
    oneflow-test bash ci/test/1node_op_test.sh
pid_without_user_op=($!)

wait $pid_with_user_op
wait $pid_without_user_op
