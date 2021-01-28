set -x
function install_all {
    /opt/python/cp36-cp36m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp37-cp37m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp38-cp38/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && rm /tmp/dev-requirements.txt
}
export pip_args="-i https://mirrors.aliyun.com/pypi/simple"
install_all
if [ $? -eq 0 ]; then
    export pip_args="-i https://mirrors.aliyun.com/pypi/simple"
    install_all
fi
if [ $? -eq 0 ]; then
    export pip_args=""
    install_all
fi
