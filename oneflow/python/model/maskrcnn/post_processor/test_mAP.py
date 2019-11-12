import numpy as np
from box_head_inference import PostProcessor
from mask_head_inference import MaskPostProcessor
from bounding_box import BoxList
from coco import COCODataset
from coco_eval import do_coco_evaluation

from pycocotools.coco import COCO
ann_file = '/dataset/mscoco_2017/annotations/sample_1_instances_val2017.json'

coco = COCO(ann_file)
imgs = coco.loadImgs(coco.getImgIds())
# mask_prob = np.load('data/mask_prob.npy')

# # 2 img
# torch_path = "/home/xfjiang/rcnn_eval_fake_data/iter_0/"
# proposal_0 = np.load(torch_path + "rpn/proposals.image0.(500, 4).npy")
# proposal_1 = np.load(torch_path + "rpn/proposals.image1.(500, 4).npy")
# class_prob = np.load(torch_path + "box_head/class_prob.(1000, 81).npy")
# box_regression = np.load(torch_path + "roi_head/box_regression.(1000, 324).npy")
# image_sizes = np.load("/tmp/shared_with_jxf/maskrcnn_eval_input_data/image_size.npy")

# 1 img
torch_path = "/home/xfjiang/repos/maskrcnn-benchmark/inference_dump/iter_0/"
proposal_0 = np.load(torch_path + "rpn/proposals.image0.(500, 4).npy")
class_prob = np.load(torch_path + "box_head/class_prob.(500, 81).npy")
box_regression = np.load(torch_path + "roi_head/box_regression.(500, 324).npy")

boxes = [BoxList(proposal_0, (1066, 800), mode="xyxy")]

# boxes = []
# for proposal, image_size in zip([proposal_0], image_sizes):
#  bbox = BoxList(proposal, (image_size[1], image_size[0]), mode="xyxy")
#  boxes.append(bbox)
postprocessor = PostProcessor()
predictions = postprocessor.forward((class_prob, box_regression), boxes)

for item in predictions:
  print(len(item))

# results = []
# def npy(field, i):
#   return np.load('data/box_head_inference_{}_{}.npy'.format(field, i))
# for i, img in enumerate(imgs):
#   width, height = img['width'], img['height']
#   width, height = img['width'] * 600.0 / img['height'], 600.0
#   bbox = BoxList(npy('bbox', i), (width, height), mode="xyxy")
#   for field in ['labels', 'scores']:
#     bbox.add_field(field, npy(field, i))
#   results.append(bbox)

# mask_postprocessor = MaskPostProcessor()
# predictions = mask_postprocessor.forward(mask_prob, results)

dataset = COCODataset(ann_file)
do_coco_evaluation(
    dataset,
    predictions,
    box_only=False,
    output_folder='./output',
    iou_types=['bbox'],
    expected_results=(),
    expected_results_sigma_tol=4,
    )

