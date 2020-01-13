rm log -rf
#export CUDA_VISIBLE_DEVICES=1,2,3

python train_piece_4gpu.py --total_batch_num=100 --base_lr=0.0000625 --weight_l2=0.0005 --gpu_num_per_node=4 --batch_size=256 --data_part_num=1 --gt_max_len=90 --raw_data=1 --num_of_batches_in_snapshot=1000 --shuffle=0 --model_load_dir=../yolov/of_model/yolov3_model_python/

#python train_piece.py --total_batch_num=10 --base_lr=0.001 --gpu_num_per_node=1 --batch_size=64 --data_part_num=1 --gt_max_len=90 --raw_data=1 --num_of_batches_in_snapshot=10 --shuffle=0 --model_load_dir=../yolov/of_model/yolov3_model_python/


#python train_piece.py --total_batch_num=500200 --base_lr=0.001 --gpu_num_per_node=1 --batch_size=64 --data_part_num=1 --gt_max_len=90 --raw_data=1 --num_of_batches_in_snapshot=10 --shuffle=0 --model_load_dir=../yolov/of_model/yolov3_model_python/

