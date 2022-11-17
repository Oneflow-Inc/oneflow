#!/bin/zsh
LOG=./logs
EXP_NAME="gcn"
O_DIR=${LOG}/${EXP_NAME}
mkdir -p ${O_DIR}

for round in 1 2 3 4 5 6 7 8 9 10; do
    echo "round ${round}"
    python train.py > ${O_DIR}/gcn.cora.${round}.log 2>&1
done