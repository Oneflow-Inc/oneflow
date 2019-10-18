import numpy as np
#from box_head_inference import PostProcessor
from mask_head_inference import MaskPostProcessor
from bounding_box import BoxList
from coco import COCODataset
from coco_eval import do_coco_evaluation

from pycocotools.coco import COCO
ann_file = '/dataset/mscoco_2017/annotations/sample_2_instances_val2017.json'
cfg_file = 'e2e_mask_rcnn_R_50_FPN_1x_xuan.yaml'

coco = COCO(ann_file)
imgs = coco.loadImgs(coco.getImgIds())
mask_prob = np.load('data/mask_prob.npy')

#boxes = []
#for proposal,img in zip(proposals, imgs):
#  width, height = img['width'] * 600.0 / img['height'], 600.0
#  #bbox = BoxList(proposal, (img['width'], img['height']) , mode="xyxy")
#  bbox = BoxList(proposal, (width, height), mode="xyxy")
#  boxes.append(bbox)
#postprocessor = PostProcessor()
#res = postprocessor.forward((class_prob, box_regression), boxes)
results = []
def npy(field, i):
  return np.load('data/box_head_inference_{}_{}.npy'.format(field, i))
for i, img in enumerate(imgs):
  width, height = img['width'], img['height']
  width, height = img['width'] * 600.0 / img['height'], 600.0
  bbox = BoxList(npy('bbox', i), (width, height), mode="xyxy")
  for field in ['labels', 'scores']:
    bbox.add_field(field, npy(field, i))
  results.append(bbox)

mask_postprocessor = MaskPostProcessor()
predictions = mask_postprocessor.forward(mask_prob, results)

dataset = COCODataset(ann_file)
do_coco_evaluation(
    dataset,
    predictions,
    box_only=False,
    output_folder='./output',
    iou_types=['bbox', 'segm'],
    expected_results=(),
    expected_results_sigma_tol=4,
    )

