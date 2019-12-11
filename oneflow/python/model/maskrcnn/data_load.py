import oneflow as flow
import oneflow.core.data.data_pb2 as data_util
import math


def roundup(x, align):
    return int(math.ceil(x / float(align)) * align)


def make_data_loader(config, training, load_cache_size=3):
    if training:
        train_cfg = config.TRAIN
        coco = flow.data.COCODataset(
            dataset_dir=train_cfg.DATASET.DATASET_DIR,
            annotation_file=train_cfg.DATASET.ANNOTATION,
            image_dir=train_cfg.DATASET.IMAGE_DIR,
            random_seed=train_cfg.DATASET.RANDOM_SEED,
            shuffle=train_cfg.DATASET.SHUFFLE,
            group_by_aspect_ratio=True,
        )
        data_loader = flow.data.DataLoader(
            coco, train_cfg.BATCH_SIZE, load_cache_size
        )

        aligned_target_size = roundup(
            train_cfg.INPUT.TARGET_SIZE, train_cfg.INPUT.IMAGE_ALIGN_SIZE
        )
        aligned_max_size = roundup(
            train_cfg.INPUT.MAX_SIZE, train_cfg.INPUT.IMAGE_ALIGN_SIZE
        )

        data_loader.add_blob(
            "image",
            data_util.kImage,
            shape=(aligned_max_size, aligned_target_size, 3),
            dtype=flow.float,
            is_dynamic=True,
        )
        data_loader.add_blob(
            "image_size",
            data_util.kImageSize,
            shape=(2,),
            dtype=flow.int32,
        )
        data_loader.add_blob(
            "gt_bbox",
            data_util.kObjectBoundingBox,
            shape=(train_cfg.INPUT.MAX_BOXES_PER_IMAGE, 4),
            dtype=flow.float,
            variable_length_axes=(0,),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "gt_labels",
            data_util.kObjectLabel,
            shape=(train_cfg.INPUT.MAX_BOXES_PER_IMAGE,),
            dtype=flow.int32,
            variable_length_axes=(0,),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "gt_segm_poly",
            data_util.kObjectSegmentation,
            shape=(
                train_cfg.INPUT.MAX_BOXES_PER_IMAGE,
                train_cfg.INPUT.MAX_POLYGONS_PER_OBJECT,
                train_cfg.INPUT.MAX_POINTS_PER_POLYGON,
                2,
            ),
            dtype=flow.double,
            variable_length_axes=(0, 1, 2),
            is_dynamic=True,
        )
        data_loader.add_blob(
            "gt_segm",
            data_util.kObjectSegmentationAlignedMask,
            shape=(
                train_cfg.INPUT.MAX_BOXES_PER_IMAGE,
                aligned_max_size,
                aligned_target_size,
            ),
            dtype=flow.int8,
            variable_length_axes=(0,),
            is_dynamic=True,
        )
        data_loader.add_transform(
            flow.data.TargetResizeTransform(
                train_cfg.INPUT.TARGET_SIZE, train_cfg.INPUT.MAX_SIZE
            )
        )
        if train_cfg.INPUT.MIRROR_PROB > 0.0:
            data_loader.add_transform(
                flow.data.ImageRandomFlip(probability=train_cfg.INPUT.MIRROR_PROB)
            )
        data_loader.add_transform(
            flow.data.ImageNormalizeByChannel(
                train_cfg.INPUT.PIXEL_MEAN, train_cfg.INPUT.PIXEL_STD
            )
        )
        data_loader.add_transform(
            flow.data.ImageAlign(train_cfg.INPUT.IMAGE_ALIGN_SIZE)
        )
        data_loader.add_transform(
            flow.data.SegmentationPolygonListToAlignedMask()
        )
    else:
        eval_cfg = config.EVAL
        coco = flow.data.COCODataset(
            dataset_dir=eval_cfg.DATASET.DATASET_DIR,
            annotation_file=eval_cfg.DATASET.ANNOTATION,
            image_dir=eval_cfg.DATASET.IMAGE_DIR,
            random_seed=123456,
            shuffle=False,
            group_by_aspect_ratio=True,
        )
        data_loader = flow.data.DataLoader(
            coco, eval_cfg.BATCH_SIZE, load_cache_size
        )

        aligned_target_size = roundup(
            eval_cfg.INPUT.TARGET_SIZE, eval_cfg.INPUT.IMAGE_ALIGN_SIZE
        )
        aligned_max_size = roundup(
            eval_cfg.INPUT.MAX_SIZE, eval_cfg.INPUT.IMAGE_ALIGN_SIZE
        )

        data_loader.add_blob(
            "image",
            data_util.kImage,
            shape=(aligned_target_size, aligned_max_size, 3),
            dtype=flow.float,
            is_dynamic=True,
        )
        data_loader.add_blob(
            "image_size",
            data_util.kImageSize,
            shape=(2,),
            dtype=flow.int32,
        )
        data_loader.add_blob(
            "image_id",
            data_util.kImageId,
            shape=(1,),
            dtype=flow.int64,
        )
        data_loader.add_transform(
            flow.data.TargetResizeTransform(
                eval_cfg.INPUT.TARGET_SIZE, eval_cfg.INPUT.MAX_SIZE
            )
        )
        data_loader.add_transform(
            flow.data.ImageNormalizeByChannel(
                eval_cfg.INPUT.PIXEL_MEAN, eval_cfg.INPUT.PIXEL_STD
            )
        )
        data_loader.add_transform(
            flow.data.ImageAlign(eval_cfg.INPUT.IMAGE_ALIGN_SIZE)
        )

    data_loader.init()
    return data_loader
