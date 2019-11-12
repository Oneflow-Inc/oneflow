import numpy as np

x = np.load("eval_dump/cls_probs.npy")
y = np.load("/home/xfjiang/rcnn_eval_fake_data/iter_0/box_head/class_prob.(1000, 81).npy")
print(np.max(np.abs(x-y)))

x = np.load("eval_dump/mask_prob.npy")
y = np.load("/home/xfjiang/rcnn_eval_fake_data/iter_0/mask_head/mask_prob.(85, 81, 28, 28).npy")
print(np.max(np.abs(x-y)))
