import numpy as np
import oneflow as flow

coco_dict = dict()


def _coco(anno_file):
    global coco_dict

    if anno_file not in coco_dict:
        from pycocotools.coco import COCO

        coco_dict[anno_file] = COCO(anno_file)

    return coco_dict[anno_file]


def _make_coco_reader_op(
    annotation_file,
    image_dir,
    batch_size,
    stride_partition=True,
    shuffle=True,
    random_seed=-1,
    name=None,
):
    if name is None:
        name = "COCOReader"
    assert isinstance(name, str)

    return (
        flow.user_op_builder(name)
        .Op("COCOReader")
        .Output("image")
        .Output("image_id")
        .Output("image_size")
        .Output("gt_bbox")
        .Output("gt_label")
        .Output("gt_segm")
        .Output("gt_segm_index")
        .Attr("annotation_file", annotation_file, "AttrTypeString")
        .Attr("image_dir", image_dir, "AttrTypeString")
        .Attr("batch_size", batch_size, "AttrTypeInt64")
        .Attr("shuffle_after_epoch", shuffle, "AttrTypeBool")
        .Attr("random_seed", random_seed, "AttrTypeInt64")
        .Attr("group_by_ratio", True, "AttrTypeBool")
        .Attr("stride_partition", stride_partition, "AttrTypeBool")
        .Attr("empty_tensor_size", 32, "AttrTypeInt64")
        .Attr("tensor_init_bytes", 1048576, "AttrTypeInt32")
        .Build()
    )


def _of_coco_data_load(anno_file, image_dir, nthread, batch_size, stride_partition):
    flow.config.cpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def coco_load_fn():
        with flow.fixed_placement("cpu", "0:0-{}".format(nthread - 1)):
            coco_reader = _make_coco_reader_op(
                annotation_file=anno_file,
                image_dir=image_dir,
                batch_size=batch_size,
                stride_partition=stride_partition,
                shuffle=False,
            )

            (
                image,
                image_id,
                image_size,
                gt_bbox,
                gt_label,
                gt_segm,
                gt_segm_index,
            ) = coco_reader.InferAndTryRun().RemoteBlobList()

            decoded_image = flow.image_decode(image)
            image_list = flow.tensor_buffer_to_tensor_list(
                decoded_image, shape=(640, 800, 3), dtype=flow.uint8
            )
            bbox_list = flow.tensor_buffer_to_tensor_list(gt_bbox, shape=(128, 4), dtype=flow.float)
            label_list = flow.tensor_buffer_to_tensor_list(
                gt_label, shape=(128,), dtype=flow.int32
            )
            segm_list = flow.tensor_buffer_to_tensor_list(gt_segm, shape=(512, 2), dtype=flow.float)
            segm_index_list = flow.tensor_buffer_to_tensor_list(
                gt_segm_index, shape=(512, 3), dtype=flow.int32
            )

        return image_id, image_size, image_list, bbox_list, label_list, segm_list, segm_index_list

    image_id, image_size, image, gt_bbox, gt_label, gt_segm, gt_segm_index = coco_load_fn().get()
    return (
        image_id.ndarray(),
        image_size.ndarray(),
        image.ndarray_lists(),
        gt_bbox.ndarray_lists(),
        gt_label.ndarray_lists(),
        gt_segm.ndarray_lists(),
        gt_segm_index.ndarray_lists(),
    )


def _coco_load_imgs(anno_file):
    coco = _coco(anno_file)
    img_ids = coco.getImgIds()
    img_ids.sort()
    for i, img_id in enumerate(img_ids[:20]):
        img_h = coco.imgs[img_id]["height"]
        img_w = coco.imgs[img_id]["width"]
        group_id = int(img_h / img_w)
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        print(
            "index: {}, image_id: {}, group_id: {}, anno len: {}".format(
                i, img_id, group_id, len(anno_ids)
            )
        )


def test_coco_reader(test_case):
    image_id, image_size, image, bbox, label, segm, segm_index = _of_coco_data_load(
        "/dataset/mscoco_2017/annotations/instances_val2017.json",
        "/dataset/mscoco_2017/val2017",
        1,
        2,
        False,
    )
    print(type(image_id))
    print(image_id)
    print(image_size)
    print(len(image))
    print([len(img) for img in image])
    print([[im.shape for im in img] for img in image])
    print(len(bbox))
    print([len(box) for box in bbox])
    print([[b.shape for b in box] for box in bbox])
    print(bbox)
    print(label)
    print(segm)
    print(segm_index)

    _coco_load_imgs("/dataset/mscoco_2017/annotations/instances_val2017.json")
