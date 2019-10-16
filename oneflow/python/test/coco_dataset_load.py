import oneflow as flow
import oneflow.core.data.data_pb2 as data_util
import numpy as np
import pickle as pkl
import argparse

from termcolor import colored
from datetime import datetime

COCO_DATASET_DIR = "/dataset/mscoco_2017"
COCO_ANNOTATIONS_FILE = "annotations/instances_val2017.json"
COCO_IMAGE_DIR = "val2017"
RANDOM_SEED = 297157
COMPARE_DATA = "data.pkl"  # batch_size == 2 data from pytorch

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
parser.add_argument(
    "-s", "--save_results", type=bool, default=False, required=False
)
parser.add_argument(
    "-cp", "--compare_results", type=bool, default=True, required=False
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
        data_util.DataSourceCase.kObjectSegmentationMask,
        shape=(64, 1344, 800),
        dtype=flow.int8,
        variable_length_axes=(0, 1, 2),
        is_dynamic=True,
    )
    data_loader.add_transform(flow.data.TargetResizeTransform(800, 1333, 32))
    data_loader.add_transform(flow.data.SegmentationPolygonListToMask())
    data_loader.init()
    return (
        data_loader("image"),
        data_loader("image_size"),
        data_loader("gt_bbox"),
        data_loader("gt_labels"),
        data_loader("gt_segm"),
        data_loader("gt_segm_mask"),
    )


def compare(data, valid, name=None):
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
        return
    else:
        print(colored("{} not close".format(name), "red"))


def main():
    # flow.config.gpu_device_num(args.gpu_num_per_node)
    # flow.config.ctrl_port(9788)
    flow.config.exp_run_conf({"enable_experiment_run": False})
    flow.config.default_data_type(flow.float)

    image, image_size, gt_bbox, gt_labels, gt_segm, gt_segm_mask = (
        coco_data_load_job().get()
    )

    if args.save_results:
        np.save("image", image.ndarray())
        np.save("image_size", image_size.ndarray())
        np.save("gt_bbox", gt_bbox.ndarray())
        np.save("gt_labels", gt_labels.ndarray())
        np.save("gt_segm", gt_segm.ndarray())
        np.save("gt_segm_mask", gt_segm_mask.ndarray())

    if args.compare_results:
        with open(COMPARE_DATA, "rb") as f:
            data = pkl.load(f)

        # compare image_size
        # of image_size format (height, width)
        # compare data image_size format (width, height)
        compare(
            image_size.ndarray(),
            np.concatenate(
                [data["image_size"][:, 1:2], data["image_size"][:, 0:1]], axis=1
            ),
            name="image_size",
        )

        # compare gt_bbox
        compare(
            gt_bbox.ndarray(),
            np.concatenate(data["gt_bbox"], axis=0),
            name="gt_bbox",
        )

        # compare gt_labels
        compare(
            gt_labels.ndarray(),
            np.concatenate(data["gt_labels"], axis=0),
            name="gt_labels",
        )

        # compare gt_segm
        def flat_polys(polys):
            poly_arrays = []
            for img_polys in polys:
                for obj_poly_list in img_polys:
                    for obj_poly in obj_poly_list:
                        poly_arrays.append(np.array(obj_poly).reshape(-1, 2))
            return np.concatenate(poly_arrays, axis=0)

        compare(gt_segm.ndarray(), flat_polys(data["gt_segm"]), name="gt_segm")

        # compare gt_segm_mask
        compare(
            gt_segm_mask.ndarray(),
            np.concatenate(
                [m.reshape(-1) for m in data["gt_segm_mask"]], axis=0
            ),
            name="gt_segm_mask",
        )


if __name__ == "__main__":
    main()
