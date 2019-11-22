rm -r distributed_eval_dump
python distributed_maskrcnn_eval.py                                                   \
  -load="/model_zoo/detection/mask_rcnn_R_50_FPN_1x/snapshot/"                        \
  -c="mask_rcnn_R_50_FPN_1x_eval.yaml"                                                \
  -cp=19237                                                                           \
  -bz=2                                                                               \
  -dataset_dir="/dataset/mscoco_2017"                                                 \
  -anno="annotations/sample_10_instances_val2017.json"                                \
  -imgd="val2017"                                                                     \
  -i=5                                                                                \
  -g=2
