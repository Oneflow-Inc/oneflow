#!/bin/bash
set -xe

export PYTHONUNBUFFERED=1

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"/test_tmp_dir"}


rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r $src_dir/oneflow/python/benchmarks $test_tmp_dir
cd $test_tmp_dir/benchmarks

export ONEFLOW_DRY_RUN=1

python3 bert_benchmark/run_pretraining.py \
    --gpu_num_per_node=1 \
    --node_num=1 \
    --learning_rate=1e-4 \
    --weight_decay_rate=0.01 \
    --batch_size_per_device=24 \
    --iter_num=5 \
    --loss_print_every_n_iter=1 \
    --data_dir="/dataset/bert/bert_seq_len_128_repeat1024" \
    --data_part_num=1 \
    --seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_hidden_layers=12 \
    --num_attention_heads=12 \
    --max_position_embeddings=512 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --hidden_size_per_head=64
