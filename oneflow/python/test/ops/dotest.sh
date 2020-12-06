set -x
set -e


for i in {1..200}; do echo "$i"; python3 1node_test.py test_deconv2d.py; done

