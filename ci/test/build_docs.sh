set -ex
pip3 install --user ci_tmp/*.whl
cp -r oneflow/docs /docs
cd /docs
make html
