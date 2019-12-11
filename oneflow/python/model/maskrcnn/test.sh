rm -r single_gpu_eval_dump
export CUDA_VISIBLE_DEVICES=3
python maskrcnn_eval.py -load="/model_zoo/detection/mask_rcnn_R_50_FPN_1x/snapshot/"  \
  -c="mask_rcnn_R_50_FPN_1x_eval.yaml"                                                \
  -cp=19237                                                                           \
  -bz=1                                                                               \
  -dataset_dir="/dataset/mscoco_2017"                                                 \
  -anno="annotations/instances_val2017.json"                                          \
  -imgd="val2017"
