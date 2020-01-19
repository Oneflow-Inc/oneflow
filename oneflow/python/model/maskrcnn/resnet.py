import oneflow as flow
from collections import namedtuple
from registry import Registry

StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)

_STAGE_SPECS = Registry({"R-50-FPN": ResNet50FPNStagesTo5})


class ResNet(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, inputs):
        stem_channels = self.cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        num_groups = self.cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = self.cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = self.cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        stage_specs = _STAGE_SPECS[self.cfg.MODEL.BACKBONE.CONV_BODY]
        freeze_at = self.cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT

        features = []
        with flow.deprecated.variable_scope("body"):
            blob = self.build_stem(inputs, stem_channels)
            for i, stage_spec in enumerate(stage_specs, 1):
                stage2_relative_factor = 2 ** (stage_spec.index - 1)
                bottleneck_channels = (
                    stage2_bottleneck_channels * stage2_relative_factor
                )
                out_channels = stage2_out_channels * stage2_relative_factor
                blob = self.build_stage(
                    blob,
                    stage_index=stage_spec.index,
                    first_stage=(True if i == 1 else False),
                    block_count=stage_spec.block_count,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    trainable=(False if i < freeze_at else True),
                )
                if stage_spec.return_features:
                    features.append(blob)

        return features

    def build_stem(self, inputs, out_channels):
        with flow.deprecated.variable_scope("stem"):
            conv1 = flow.layers.conv2d(
                inputs=inputs,
                filters=out_channels,
                kernel_size=[7, 7],
                strides=[2, 2],
                padding="SAME",
                data_format="NCHW",
                dilation_rate=[1, 1],
                trainable=False,
                name="conv1",
                use_bias=False,
            )
            affine = flow.layers.affine_channel(
                conv1,
                axis=1,
                activation=flow.keras.activations.relu,
                trainable=False,
                name="bn1",
            )
            pool = flow.nn.max_pool2d(
                affine,
                ksize=[3, 3],
                strides=[2, 2],
                padding="SAME",
                data_format="NCHW",
                name="pool1",
            )
        return pool

    def build_stage(
        self,
        inputs,
        stage_index,
        first_stage,
        block_count,
        bottleneck_channels,
        out_channels,
        trainable,
    ):
        out = inputs
        for block_index in range(block_count):
            stride = 2 if (not first_stage and block_index == 0) else 1
            with flow.deprecated.variable_scope(
                "layer{}_{}".format(stage_index, block_index)
            ):
                out = self.build_block(
                    out,
                    bottleneck_channels,
                    out_channels,
                    stride=stride,
                    downsample=block_index == 0,
                    trainable=trainable,
                )

        return out

    def build_block(
        self,
        inputs,
        bottleneck_channels,
        out_channels,
        stride,
        downsample=False,
        trainable=False,
    ):
        if downsample:
            x = flow.layers.conv2d(
                inputs=inputs,
                filters=out_channels,
                kernel_size=1,
                strides=stride,
                padding="SAME",
                data_format="NCHW",
                dilation_rate=1,
                trainable=trainable,
                name="downsample_0",
                use_bias=False,
            )
            downsample_blob = flow.layers.affine_channel(
                x, axis=1, trainable=False, name="downsample_1"
            )

        stride_1x1, stride_3x3 = (
            (stride, 1) if self.cfg.MODEL.RESNETS.STRIDE_IN_1X1 else (1, stride)
        )

        conv1 = flow.layers.conv2d(
            inputs=inputs,
            filters=bottleneck_channels,
            kernel_size=1,
            strides=stride_1x1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
            trainable=trainable,
            name="conv1",
            use_bias=False,
        )
        affine1 = flow.layers.affine_channel(
            conv1,
            activation=flow.keras.activations.relu,
            axis=1,
            trainable=False,
            name="bn1",
        )

        conv2 = flow.layers.conv2d(
            inputs=affine1,
            filters=bottleneck_channels,
            kernel_size=3,
            strides=stride_3x3,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
            trainable=trainable,
            name="conv2",
            use_bias=False,
        )
        affine2 = flow.layers.affine_channel(
            conv2,
            activation=flow.keras.activations.relu,
            axis=1,
            trainable=False,
            name="bn2",
        )

        conv3 = flow.layers.conv2d(
            inputs=affine2,
            filters=out_channels,
            kernel_size=1,
            strides=1,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=1,
            trainable=trainable,
            name="conv3",
            use_bias=False,
        )
        affine3 = flow.layers.affine_channel(
            conv3, activation=None, axis=1, trainable=False, name="bn3"
        )

        add = flow.math.add(
            downsample_blob if downsample else inputs, affine3, name="add"
        )
        return flow.keras.activations.relu(add, name="output")
