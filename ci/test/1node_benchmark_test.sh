set -xe

rm -rf /benchmarks
cp -r oneflow/python/benchmarks /benchmarks
cd /benchmarks

python3 cnn_benchmark/of_cnn_benchmarks.py \
    --gpu_num_per_node=1 \
    --model="vgg16" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --learning_rate=0.01 \
    --optimizer="sgd" \
    --loss_print_every_n_iter=1 \
    --data_dir="/dataset/imagenet_227/train/32"

python3 cnn_benchmark/of_cnn_benchmarks.py \
    --gpu_num_per_node=1 \
    --model="alexnet" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --learning_rate=0.01 \
    --optimizer="sgd" \
    --loss_print_every_n_iter=1 \
    --data_dir="/dataset/imagenet_227/train/32"

python3 cnn_benchmark/of_cnn_benchmarks.py \
    --gpu_num_per_node=1 \
    --model="resnet50" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --gpu_image_decoder=True \
    --learning_rate=0.01 \
    --optimizer="sgd" \
    --loss_print_every_n_iter=1 \
    --data_dir="/dataset/imagenet_227/train/32"

python3 cnn_benchmark/of_cnn_benchmarks.py \
    --gpu_num_per_node=1 \
    --model="resnet50" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --learning_rate=0.01 \
    --optimizer="sgd" \
    --loss_print_every_n_iter=1 

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
