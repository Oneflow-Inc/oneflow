import oneflow as flow
import oneflow.core.data.data_pb2 as data_util


def make_data_loader(
    batch_size,
    batch_cache_size=3,
    dataset_dir="/dataset/mscoco_2017",
    annotation_file="annotations/instances_train2017.json",
    image_dir="train2017",
    random_seed=123456,
    shuffle=False,
    group_by_aspect_ratio=True,
    random_flip_image=False,
):
    coco = flow.data.COCODataset(
        dataset_dir,
        annotation_file,
        image_dir,
        random_seed,
        shuffle,
        group_by_aspect_ratio,
    )
    data_loader = flow.data.DataLoader(coco, batch_size, batch_cache_size)
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
        shape=(128, 4),
        dtype=flow.float,
        variable_length_axes=(0,),
        is_dynamic=True,
    )
    data_loader.add_blob(
        "gt_labels",
        data_util.DataSourceCase.kObjectLabel,
        shape=(128,),
        dtype=flow.int32,
        variable_length_axes=(0,),
        is_dynamic=True,
    )
    data_loader.add_blob(
        "gt_segm_poly",
        data_util.DataSourceCase.kObjectSegmentation,
        shape=(128, 2, 256, 2),
        dtype=flow.double,
        variable_length_axes=(0, 1, 2),
        is_dynamic=True,
    )
    data_loader.add_blob(
        "gt_segm",
        data_util.DataSourceCase.kObjectSegmentationAlignedMask,
        shape=(128, 1344, 800),
        dtype=flow.int8,
        variable_length_axes=(0,),
        is_dynamic=True,
    )
    data_loader.add_transform(flow.data.TargetResizeTransform(800, 1333))
    if random_flip_image:
        data_loader.add_transform(flow.data.ImageRandomFlip())
    data_loader.add_transform(
        flow.data.ImageNormalizeByChannel((102.9801, 115.9465, 122.7717))
    )
    data_loader.add_transform(flow.data.ImageAlign(32))
    data_loader.add_transform(flow.data.SegmentationPolygonListToAlignedMask())
    data_loader.init()
    return data_loader
