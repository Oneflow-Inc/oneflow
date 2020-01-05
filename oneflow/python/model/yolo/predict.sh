rm log -rf
data_dir=hyd_of_record #raw_record_y1x1y2x2  #jpeg_record_y1x1y2x2/ #raw_record_y1x1y2x2
export CUDA_VISIBLE_DEVICES=1,2,3

python predict.py --total_batch_num=20 --base_lr=0.1 --class_num=1 --train_dir=$data_dir --gpu_num_per_node=1 --batch_size=1  --model_load_dir=../yolov/of_model/yolov3_model_python/

