import oneflow as flow


class FPN(object):
    def __init__(self, cfg):
        self.inner_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.layer_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

    def build(self, features):
        layers = range(len(features))
        inner_blobs = [None] * len(layers)
        lateral_blobs = [None] * len(layers)
        topdown_blobs = [None] * len(layers)
        layer_blobs = [None] * len(layers)

        with flow.deprecated.variable_scope("fpn"):
            for i, feature in zip(layers[::-1], features[::-1]):
                lateral_name = "inner_lateral{}".format(i + 1)
                inner_name = "inner{}".format(i + 1)
                topdown_name = "inner_topdown{}".format(i + 1)
                layer_name = "layer{}".format(i + 1)
                if i == layers[0]:
                    lateral_blobs[i] = self.lateral(feature, lateral_name)
                    inner_blobs[i] = lateral_blobs[i] + topdown_blobs[i + 1]
                elif i == layers[-1]:
                    inner_blobs[i] = self.lateral(feature, inner_name)
                    topdown_blobs[i] = self.interpolate(inner_blobs[i], topdown_name)
                else:
                    lateral_blobs[i] = self.lateral(feature, lateral_name)
                    inner_blobs[i] = lateral_blobs[i] + topdown_blobs[i + 1]
                    topdown_blobs[i] = self.interpolate(inner_blobs[i], topdown_name)
                layer_blobs[i] = self.layer(inner_blobs[i], layer_name)

            layer_blobs.append(
                self.max_pool(layer_blobs[-1], "layer{}".format(len(layer_blobs) + 1))
            )

        return layer_blobs

    def lateral(self, x, name):
        return flow.layers.conv2d(
            x,
            filters=self.inner_channels,
            kernel_size=[1, 1],
            padding="SAME",
            data_format="NCHW",
            strides=[1, 1],
            dilation_rate=[1, 1],
            kernel_initializer=flow.kaiming_initializer(
                shape=(self.inner_channels, x.static_shape[1]) + (1, 1),
                distribution="random_uniform",
                mode="fan_in",
                nonlinearity="leaky_relu",
                negative_slope=1.0
            ),
            bias_initializer=flow.constant_initializer(0),
            name=name,
        )

    def interpolate(self, x, name):
        return flow.detection.upsample_nearest(
            x, name=name, scale=2, data_format="channels_first"
        )

    def layer(self, x, name):
        return flow.layers.conv2d(
            x,
            kernel_size=[3, 3],
            filters=self.layer_channels,
            padding="SAME",
            data_format="NCHW",
            strides=[1, 1],
            dilation_rate=[1, 1],
            kernel_initializer=flow.kaiming_initializer(
                shape=(self.layer_channels, x.static_shape[1]) + (3, 3),
                distribution="random_uniform",
                mode="fan_in",
                nonlinearity="leaky_relu",
                negative_slope=1.0
            ),
            bias_initializer=flow.constant_initializer(0),
            name=name,
        )

    def max_pool(self, x, name):
        return flow.nn.max_pool2d(
            x,
            ksize=[1, 1],
            strides=[2, 2],
            padding="SAME",
            data_format="NCHW",
            name="pool1",
        )
