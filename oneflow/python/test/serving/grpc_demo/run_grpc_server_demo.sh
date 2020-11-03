SERVER_ADDR="192.168.1.14"
SERVER_PORT=8000

SAVED_MODEL_DIR="./resnet50_models"

MODEL_VERSION=1

if [ ! -d ${SAVED_MODEL_DIR} ] 
then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_v15_of_best_model_val_top1_77318.tgz
    tar zxf resnet_v15_of_best_model_val_top1_77318.tgz

    python3 test_resnet_save_and_load.py
fi

python3 grpc_server.py \
    --server_address $SERVER_ADDR \
    --server_port $SERVER_PORT \
    --saved_model_path $SAVED_MODEL_DIR \
    --model_version $MODEL_VERSION
