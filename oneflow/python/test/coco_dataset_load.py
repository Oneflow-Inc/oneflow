import oneflow as flow
import oneflow.core.data.data_pb2 as data_util
import numpy as np
import pickle as pkl
import os
import time
import argparse

from termcolor import colored

COCO_DATASET_DIR = "/dataset/mscoco_2017"
# COCO_ANNOTATIONS_FILE = "annotations/instances_val2017.json"
COCO_ANNOTATIONS_FILE = "annotations/instances_train2017.json"
# COCO_IMAGE_DIR = "val2017"
COCO_IMAGE_DIR = "train2017"
RANDOM_SEED = 297157
# batch_size == 2 data from pytorch
COMPARE_DATA = [
    "mock_data/iter_1.pkl",
    "mock_data/iter_2.pkl",
    "mock_data/iter_3.pkl",
    "mock_data/iter_4.pkl",
    "mock_data/iter_5.pkl",
    "mock_data/iter_6.pkl",
    "mock_data/iter_7.pkl",
    "mock_data/iter_8.pkl",
    "mock_data/iter_9.pkl",
    "mock_data/iter_10.pkl",
]

parser = argparse.ArgumentParser(description="flags for data loader")
# parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
# parser.add_argument("-i", "--iter_num", type=int, default=10, required=False)
# parser.add_argument(
#     "-m", "--multinode", default=False, action="store_true", required=False
# )
# parser.add_argument(
#     "-s", "--skip_scp_binary", default=False, action="store_true", required=False
# )
# parser.add_argument(
#     "-c",
#     "--scp_binary_without_uuid",
#     default=False,
#     action="store_true",
#     required=False,
# )
# parser.add_argument(
#     "-r", "--remote_by_hand", default=False, action="store_true", required=False
# )
parser.add_argument(
    "-d", "--dataset_dir", type=str, default=COCO_DATASET_DIR, required=False
)
parser.add_argument(
    "-a",
    "--annotation_file",
    type=str,
    default=COCO_ANNOTATIONS_FILE,
    required=False,
)
parser.add_argument(
    "-i", "--image_dir", type=str, default=COCO_IMAGE_DIR, required=False
)
parser.add_argument("-b", "--batch_size", type=int, default=2, required=False)
parser.add_argument(
    "-bc", "--batch_cache_size", type=int, default=3, required=False
)
# parser.add_argument(
#     "-s", "--save_results", default=False, action="store_true", required=False
# )
parser.add_argument(
    "-cp",
    "--compare_results",
    default=False,
    action="store_true",
    required=False,
)

args = parser.parse_args()


@flow.function
def coco_data_load_job():
    coco = flow.data.COCODataset(
        args.dataset_dir,
        args.annotation_file,
        args.image_dir,
        random_seed=RANDOM_SEED,
        shuffle=False,
        group_by_aspect_ratio=True,
    )
    data_loader = flow.data.DataLoader(
        coco, args.batch_size, args.batch_cache_size
    )
    data_loader.add_blob(
        "image",
        data_util.DataSourceCase.kImage,
        shape=(1344, 800, 3),
        dtype=flow.float,
        is_dynamic=True,
    )
    data_loader.add_blob(
        "image_id",
        data_util.DataSourceCase.kImageId,
        shape=(1,),
        dtype=flow.int64,
    )
    data_loader.add_blob(
        "image_size",
        data_util.DataSourceCase.kImageSize,
        shape=(2,),
        dtype=flow.int32,
    )
    data_loader.add_blob(
        "gt_bbox",
        data_util.DataSourceCase.kObjectBoundingBox,
        shape=(64, 4),
        dtype=flow.float,
        variable_length_axes=(0,),
        is_dynamic=True,
    )
    data_loader.add_blob(
        "gt_labels",
        data_util.DataSourceCase.kObjectLabel,
        shape=(64,),
        dtype=flow.int32,
        variable_length_axes=(0,),
        is_dynamic=True,
    )
    data_loader.add_blob(
        "gt_segm",
        data_util.DataSourceCase.kObjectSegmentation,
        shape=(64, 2, 256, 2),
        dtype=flow.double,
        variable_length_axes=(0, 1, 2),
        is_dynamic=True,
    )
    data_loader.add_blob(
        "gt_segm_mask",
        data_util.DataSourceCase.kObjectSegmentationAlignedMask,
        shape=(64, 1344, 800),
        dtype=flow.int8,
        variable_length_axes=(0,),
        is_dynamic=True,
    )
    data_loader.add_transform(flow.data.TargetResizeTransform(800, 1333))
    data_loader.add_transform(
        flow.data.ImageNormalizeByChannel((102.9801, 115.9465, 122.7717))
    )
    data_loader.add_transform(flow.data.ImageAlign(32))
    data_loader.add_transform(flow.data.SegmentationPolygonListToAlignedMask())
    data_loader.init()
    return (
        data_loader("image"),
        data_loader("image_id"),
        data_loader("image_size"),
        data_loader("gt_bbox"),
        data_loader("gt_labels"),
        data_loader("gt_segm"),
        data_loader("gt_segm_mask"),
    )


def compare(data, valid, name=None):
    success = False
    name = name or "unknown"
    if valid.shape != data.shape:
        print(
            colored(
                "shape not identical: {} vs {}".format(valid.shape, data.shape),
                "red",
            )
        )
        return

    if np.allclose(valid, data):
        if np.array_equal(valid, data):
            print(colored("{} identical".format(name), "green"))
        else:
            print(colored("{} allclose".format(name), "blue"))
        success = True
    else:
        print(colored("{} not close".format(name), "red"))

    return success


def compare_test_case():
    # flow.config.gpu_device_num(args.gpu_num_per_node)
    # flow.config.ctrl_port(9788)
    flow.config.exp_run_conf({"enable_experiment_run": False})
    flow.config.default_data_type(flow.float)

    for i in range(len(COMPARE_DATA)):
        image, image_id, image_size, gt_bbox, gt_labels, gt_segm, gt_segm_mask = (
            coco_data_load_job().get()
        )

        def get_output_path(name, iter):
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            return os.path.join(output_dir, "iter_{}_{}".format(iter, name))

        print(colored("=== iter_{} compare begin ===".format(i), "yellow"))

        with open(COMPARE_DATA[i], "rb") as f:
            data = pkl.load(f)

        # compare images
        contrast_image = np.transpose(data["images"], (0, 2, 3, 1))
        diff = image - contrast_image
        max_diff = np.max(diff)
        if max_diff <= 1.0:
            print(colored("image max diff is: {}".format(max_diff), "blue"))
        else:
            print(colored("image max diff is: {}".format(max_diff), "red"))
            np.save(get_output_path("images", i), image.ndarray())
            np.save(get_output_path("images_contrast", i), contrast_image)

        # compare image_size
        # of image_size format (height, width)
        # compare data image_size format (width, height)
        if not compare(
            image_size.ndarray(),
            np.concatenate(
                [data["image_size"][:, 1:2], data["image_size"][:, 0:1]], axis=1
            ),
            name="image_size",
        ):
            np.save(get_output_path("image_size", i), image_size.ndarray())

        # compare gt_bbox
        if not compare(
            gt_bbox.ndarray(),
            np.concatenate(data["gt_bbox"], axis=0),
            name="gt_bbox",
        ):
            np.save(get_output_path("gt_bbox", i), gt_bbox.ndarray())

        # compare gt_labels
        if not compare(
            gt_labels.ndarray(),
            np.concatenate(data["gt_labels"], axis=0),
            name="gt_labels",
        ):
            np.save(get_output_path("gt_labels", i), gt_labels.ndarray())

        # compare gt_segm
        def flat_polys(polys):
            poly_arrays = []
            for img_polys in polys:
                for obj_poly_list in img_polys:
                    for obj_poly in obj_poly_list:
                        poly_arrays.append(np.array(obj_poly).reshape(-1, 2))
            return np.concatenate(poly_arrays, axis=0)

        if not compare(
            gt_segm.ndarray(),
            flat_polys(data["gt_segm_poly"]),
            name="gt_segm_poly",
        ):
            np.save(get_output_path("gt_segm_poly", i), gt_segm.ndarray())

        # compare gt_segm_mask
        masks = []
        for m in data["gt_segm_mask"]:
            if len(m.shape) == 2:
                m = np.expand_dims(m, axis=0)

            assert len(m.shape) == 3
            pad_h = image.shape[1] - m.shape[1]
            pad_w = image.shape[2] - m.shape[2]
            masks.append(np.pad(m, ((0, 0), (0, pad_h), (0, pad_w))))

        if not compare(
            gt_segm_mask.ndarray(),
            np.concatenate(masks, axis=0),
            name="gt_segm_mask",
        ):
            np.save(get_output_path("gt_segm_mask", i), gt_segm_mask.ndarray())
            np.save(
                get_output_path("gt_segm_mask_contrast", i),
                np.concatenate(masks, axis=0),
            )

        print(colored("=== iter_{} compare done ===\n".format(i), "yellow"))


cur_step = 0
start_time_list = []


def profile_async_test_case():
    def profile_callback(results):
        global cur_step
        # (image, image_size, gt_bbox, gt_labels, gt_segm, gt_segm_mask) = results
        print(
            "{:<10}{:>12}".format(
                cur_step, time.time() - start_time_list[cur_step]
            )
        )
        cur_step += 1

    print("{:<10}{:>12}".format("iter", "time"))
    for i in range(10):
        coco_data_load_job().async_get(profile_callback)
        start_time_list.append(time.time())


def profile_test_case():
    print("{:<10}{:<25}{:>12}".format("iter", "image_id", "time"))
    start_time = time.time()
    for i in range(10):
        results = coco_data_load_job().get()
        image_id = np.squeeze(results[1].ndarray(), axis=1).tolist()
        step_time = time.time()
        print(
            "{:<10}{:<25}{:>12}".format(
                i, str(image_id), step_time - start_time
            )
        )
        start_time = step_time


if __name__ == "__main__":
    if args.compare_results:
        compare_test_case()
    else:
        profile_test_case()
