docker run --shm-size=8g --rm -it -w $PWD -v $PWD:$PWD -v /dataset:/dataset -v /model_zoo:/model_zoo \
    -v $HOME:$HOME \
    oneflow-test:$USER \
    bash
