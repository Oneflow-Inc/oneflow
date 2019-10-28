rm -r eval_dump
export CUDA_VISIBLE_DEVICES=2
python maskrcnn.py -box_head_post_processor -load="/model_zoo/detection/mask_rcnn_R_50_FPN_1x/snapshot/" -c="mask_rcnn_R_50_FPN_1x_eval.yaml"
