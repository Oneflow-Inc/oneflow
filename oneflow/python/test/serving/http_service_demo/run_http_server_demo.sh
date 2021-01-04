SERVER_ADDR="192.168.1.14"
SERVER_PORT=8000

MODEL_LOAD_DIR="resnet_v15_of_best_model_val_top1_77318"

if [ ! -d ${MODEL_LOAD_DIR} ] 
then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_v15_of_best_model_val_top1_77318.tgz
    tar zxf resnet_v15_of_best_model_val_top1_77318.tgz
fi

python3 http_server_demo.py \
    --server_address $SERVER_ADDR \
    --server_port $SERVER_PORT \
    --model_load_dir $MODEL_LOAD_DIR
