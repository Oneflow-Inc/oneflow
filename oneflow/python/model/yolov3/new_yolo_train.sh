rm log -rf
data_dir=../yolov/jpeg_record/ #raw_record_y1x1y2x2  #jpeg_record_y1x1y2x2/ #raw_record_y1x1y2x2
CUDA_VISIBLE_DEVICES=1,2,3

#gdb --args python new_yolo_train.py --total_batch_num=1 --base_lr=0.1 --class_num=1 --train_dir=$data_dir --gpu_num_per_node=1 --batch_size=2 --data_part_num=1 --gt_max_len=90 --raw_data=1 --num_of_batches_in_snapshot=10 --shuffle=0 --model_load_dir=of_model/yolov3_model_python/

python new_yolo_train.py --total_batch_num=100 --base_lr=0.1 --class_num=1 --train_dir=$data_dir --gpu_num_per_node=1 --batch_size=2 --data_part_num=1 --gt_max_len=90 --raw_data=1 --num_of_batches_in_snapshot=10 --shuffle=0

