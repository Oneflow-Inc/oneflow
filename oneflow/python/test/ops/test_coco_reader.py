import numpy as np
import oneflow as flow


def _make_coco_reader_op(
    annotation_file, image_dir, batch_size, shuffle=True, random_seed=-1, name=None
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
        .Output("gt_segm_offset")
        .Attr("annotation_file", annotation_file, "AttrTypeString")
        .Attr("image_dir", image_dir, "AttrTypeString")
        .Attr("batch_size", batch_size, "AttrTypeInt64")
        .Attr("shuffle_after_epoch", shuffle, "AttrTypeBool")
        .Attr("random_seed", random_seed, "AttrTypeInt64")
        .Attr("group_by_ratio", True, "AttrTypeBool")
        .Attr("stride_partition", False, "AttrTypeBool")
        .Attr("empty_tensor_size", 32, "AttrTypeInt64")
        .Attr("tensor_init_bytes", 1048576, "AttrTypeInt32")
        .Build()
    )


def _of_data_loader():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def coco_job_func():
        with flow.fixed_placement("cpu", "0:0"):
            coco_reader = _make_coco_reader_op(
                annotation_file="/dataset/mscoco_2017/annotations/instances_val2017.json",
                image_dir="/dataset/mscoco_2017/val2017",
                batch_size=2,
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
        return image_id
        # return image, image_id, image_size, gt_bbox, gt_label, gt_segm, gt_segm_index

    # image, image_id, image_size, gt_bbox, gt_label, gt_segm, gt_segm_index = coco_job_func().get()
    image_id = coco_job_func().get()
    print(image_id)


def test_coco_reader(test_case):
    _of_data_loader()
