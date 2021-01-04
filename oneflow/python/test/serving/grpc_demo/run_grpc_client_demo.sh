SERVER_ADDR="192.168.1.14"
SERVER_PORT=8000
IMAGES_PATH="/dataset/http_service_demo_client_images/"

python3 grpc_client.py \
    --server_address $SERVER_ADDR \
    --server_port $SERVER_PORT \
    --test_images_path $IMAGES_PATH

