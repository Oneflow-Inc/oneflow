cmd="python model_server.py"
cmd+=" --port=9887"
cmd+=" --model_config_prototxt=resnet50_model_config.prototxt"

set -x
sh -c "$cmd"
