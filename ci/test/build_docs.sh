set -ex
cp -r docs /docs
cd /docs
make html SPHINXOPTS="-W --keep-going"
make html_cn SPHINXOPTS="-W --keep-going"
