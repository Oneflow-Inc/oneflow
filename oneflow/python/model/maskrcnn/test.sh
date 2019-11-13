rm -r eval_dump
export CUDA_VISIBLE_DEVICES=2
python maskrcnn_eval.py -load="/model_zoo/detection/mask_rcnn_R_50_FPN_1x/snapshot/" -c="mask_rcnn_R_50_FPN_1x_eval.yaml" -cp=19237
