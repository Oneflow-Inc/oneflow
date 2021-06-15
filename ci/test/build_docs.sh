set -ex
cp -r docs /docs
cd /docs
make html SPHINXOPTS="-W --keep-going"
