import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from datetime import datetime
import argparse
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
        self.stem_out_channels = cfg.BACKBONE.RESNET_STEM_OUT_CHANNELS
        self.stage_specs = _STAGE_SPECS[cfg.BACKBONE.CONV_BODY]
        self.freeze_at = cfg.BACKBONE.FREEZE_CONV_BODY_AT

    def build(self, inputs):
        features = []
        with flow.deprecated.variable_scope("body"):
            # CHECK_POINT: stem out
            blob = self.build_stem(inputs)
            for i, stage_spec in enumerate(self.stage_specs, 1):
                stage_channel_relative_factor = 2 ** (stage_spec.index - 1)
                bottleneck_channels = (
                    self.stem_out_channels * stage_channel_relative_factor
                )
                out_channels = (
                    self.stem_out_channels * stage_channel_relative_factor * 4
                )
                # CHECK_POINT: stage out
                blob = self.build_stage(
                    blob,
                    stage_index=stage_spec.index,
                    first_stage=(True if i == 1 else False),
                    block_count=stage_spec.block_count,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    trainable=(False if i < self.freeze_at else True),
                )
                if stage_spec.return_features:
                    features.append(blob)

        return features

    def build_stem(self, inputs):
        with flow.deprecated.variable_scope("stem"):
            conv1 = flow.layers.conv2d(
                inputs=inputs,
                filters=self.cfg.BACKBONE.RESNET_STEM_OUT_CHANNELS,
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
            if first_stage:
                strides = [1, 1]
            else:
                strides = [2, 2] if block_index == 0 else [1, 1]
            with flow.deprecated.variable_scope(
                "layer{}_{}".format(stage_index, block_index)
            ):
                out = self.build_block(
                    out,
                    bottleneck_channels,
                    out_channels,
                    strides=strides,
                    downsample=block_index == 0,
                    trainable=trainable,
                )

        return out

    def build_block(
        self,
        inputs,
        bottleneck_channels,
        out_channels,
        strides,
        downsample=False,
        trainable=False,
    ):
        if downsample:
            x = flow.layers.conv2d(
                inputs=inputs,
                filters=out_channels,
                kernel_size=[1, 1],
                strides=strides,
                padding="SAME",
                data_format="NCHW",
                dilation_rate=[1, 1],
                trainable=trainable,
                name="downsample_0",
                use_bias=False,
            )
            downsample_blob = flow.layers.affine_channel(
                x, axis=1, trainable=False, name="downsample_1"
            )

        conv1 = flow.layers.conv2d(
            inputs=inputs,
            filters=bottleneck_channels,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="SAME",
            data_format="NCHW",
            dilation_rate=[1, 1],
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
            kernel_size=[3, 3],
            strides=strides,
            padding="SAME",
            data_format="NCHW",
            dilation_rate=[1, 1],
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
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="SAME",
            data_format="NCHW",
            dilation_rate=[1, 1],
            trainable=trainable,
            name="conv3",
            use_bias=False,
        )
        affine3 = flow.layers.affine_channel(
            conv3,
            activation=flow.keras.activations.relu,
            axis=1,
            trainable=False,
            name="bn3",
        )
        return flow.keras.activations.relu(
            (downsample_blob if downsample else inputs) + affine3
        )
