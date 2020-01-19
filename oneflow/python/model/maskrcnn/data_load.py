import oneflow as flow
import oneflow.core.data.data_pb2 as data_util
import math


def roundup(x, align):
    return int(math.ceil(x / float(align)) * align)


def make_data_loader(cfg, is_train):
    if is_train:
        coco = flow.data.COCODataset(
            dataset_dir=cfg.DATASETS.TRAIN,
            annotation_file=cfg.DATASETS.ANNOTATION_TRAIN,
            image_dir=cfg.DATASETS.IMAGE_DIR_TRAIN,
            random_seed=cfg.DATASETS.RANDOM_SEED,
            shuffle=cfg.DATASETS.SHUFFLE,
            group_by_aspect_ratio=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        )
        data_loader = flow.data.DataLoader(
            coco, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS
        )

        aligned_min_size = roundup(cfg.INPUT.MIN_SIZE_TRAIN, cfg.DATALOADER.SIZE_DIVISIBILITY)
        aligned_max_size = roundup(cfg.INPUT.MAX_SIZE_TRAIN, cfg.DATALOADER.SIZE_DIVISIBILITY)

        data_loader.add_blob(
            "image",
            data_util.kImage,
            shape=(aligned_max_size, aligned_min_size, 3),
            dtype=flow.float,
            is_dynamic=True,
        )
        data_loader.add_blob("image_size", data_util.kImageSize, shape=(2,), dtype=flow.int32)
        data_loader.add_blob(
            "gt_bbox",
            data_util.kObjectBoundingBox,
            shape=(cfg.INPUT.MAX_BOXES_PER_IMAGE, 4),
            dtype=flow.float,
            variable_length_axes=(0,),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "gt_labels",
            data_util.kObjectLabel,
            shape=(cfg.INPUT.MAX_BOXES_PER_IMAGE,),
            dtype=flow.int32,
            variable_length_axes=(0,),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "gt_segm_poly",
            data_util.kObjectSegmentation,
            shape=(
                cfg.INPUT.MAX_BOXES_PER_IMAGE,
                cfg.INPUT.MAX_POLYGONS_PER_OBJECT,
                cfg.INPUT.MAX_POINTS_PER_POLYGON,
                2,
            ),
            dtype=flow.double,
            variable_length_axes=(0, 1, 2),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "gt_segm",
            data_util.kObjectSegmentationAlignedMask,
            shape=(cfg.INPUT.MAX_BOXES_PER_IMAGE, aligned_max_size, aligned_min_size),
            dtype=flow.int8,
            variable_length_axes=(0,),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "image_id",
            data_util.kImageId,
            shape=(1, ),
            dtype=flow.int64,
        )
        data_loader.add_transform(
            flow.data.TargetResizeTransform(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
        )
        if cfg.INPUT.MIRROR_PROB > 0.0:
            data_loader.add_transform(flow.data.ImageRandomFlip(probability=cfg.INPUT.MIRROR_PROB))
        data_loader.add_transform(
            flow.data.ImageNormalizeByChannel(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        data_loader.add_transform(flow.data.ImageAlign(cfg.DATALOADER.SIZE_DIVISIBILITY))
        data_loader.add_transform(flow.data.SegmentationPolygonListToAlignedMask())
    else:
        coco = flow.data.COCODataset(
            dataset_dir=cfg.DATASETS.TEST,
            annotation_file=cfg.DATASETS.ANNOTATION_TEST,
            image_dir=cfg.DATASET.IMAGE_DIR_TEST,
            random_seed=123456,
            shuffle=False,
            group_by_aspect_ratio=True,
        )
        data_loader = flow.data.DataLoader(
            coco, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS
        )

        aligned_min_size = roundup(cfg.INPUT.MIN_SIZE_TEST, cfg.DATALOADER.SIZE_DIVISIBILITY)
        aligned_max_size = roundup(cfg.INPUT.MAX_SIZE_TEST, cfg.DATALOADER.SIZE_DIVISIBILITY)

        data_loader.add_blob(
            "image",
            data_util.kImage,
            shape=(aligned_min_size, aligned_max_size, 3),
            dtype=flow.float,
            is_dynamic=True,
        )
        data_loader.add_blob("image_size", data_util.kImageSize, shape=(2,), dtype=flow.int32)
        data_loader.add_blob("image_id", data_util.kImageId, shape=(1,), dtype=flow.int64)
        data_loader.add_transform(
            flow.data.TargetResizeTransform(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        )
        data_loader.add_transform(
            flow.data.ImageNormalizeByChannel(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        data_loader.add_transform(flow.data.ImageAlign(cfg.DATALOADER.SIZE_DIVISIBILITY))

    data_loader.init()
    return data_loader
