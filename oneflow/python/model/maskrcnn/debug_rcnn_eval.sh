export CUDA_VISIBLE_DEVICES=1
python maskrcnn.py -rcnn_eval -load="/model_zoo/detection/mask_rcnn_R_50_FPN_1x/snapshot/"
