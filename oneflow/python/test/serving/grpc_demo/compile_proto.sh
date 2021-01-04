python3 -m grpc_tools.protoc \
    --proto_path=. ./prediction_service.proto \
    --python_out=. \
    --grpc_python_out=.
