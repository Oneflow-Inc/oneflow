from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.modules.batchnorm import _NormBase


class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _forward(self, x):
        axis = 1
        params_shape = [x.shape[axis]]
        weight = self.weight
        bias = self.bias
        nd_params_shape = [1] * len(x.shape)
        nd_params_shape[axis] = params_shape[0]
        mean = x.mean(2, keepdim=True)
        variance = x.var(2, keepdim=True)
        normalized = (x - mean) / flow.experimental.sqrt(variance + self.eps)
        if self.weight and params_shape[0] == self.weight.nelement():
            weight = self.weight.reshape(shape=nd_params_shape)
        if self.bias and params_shape[0] == self.bias.nelement():
            bias = self.bias.reshape(shape=nd_params_shape)
        if self.weight:
            normalized = normalized * weight
        if self.bias:
            normalized = normalized + bias
        return normalized

    def forward(self, x):
        self._check_input_dim(x)
        reshape_to_1d = x.reshape([x.shape[0], x.shape[1], -1])
        normalized_1d_out = self._forward(reshape_to_1d)
        reshape_back_to_nd = normalized_1d_out.reshape(list(x.shape))
        return reshape_back_to_nd


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
