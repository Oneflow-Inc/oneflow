from oneflow.compatible.single_client.ops.nn_ops import conv1d
from oneflow.compatible.single_client.ops.nn_ops import conv2d
from oneflow.compatible.single_client.ops.nn_ops import conv3d
from oneflow.compatible.single_client.ops.nn_ops import moments
from oneflow.compatible.single_client.ops.nn_ops import group_normalization
from oneflow.compatible.single_client.ops.nn_ops import instance_normalization1d
from oneflow.compatible.single_client.ops.nn_ops import instance_normalization2d
from oneflow.compatible.single_client.ops.nn_ops import instance_normalization3d
from oneflow.compatible.single_client.ops.nn_ops import batch_normalization
from oneflow.compatible.single_client.ops.nn_ops import layer_norm
from oneflow.compatible.single_client.ops.nn_ops import tf_conv2d
from oneflow.compatible.single_client.ops.nn_ops import bias_add
from oneflow.compatible.single_client.ops.nn_ops import fused_bias_add_gelu
from oneflow.compatible.single_client.ops.nn_ops import fused_bias_add_dropout
from oneflow.compatible.single_client.ops.nn_ops import max_pool1d
from oneflow.compatible.single_client.ops.nn_ops import avg_pool1d
from oneflow.compatible.single_client.ops.nn_ops import max_pool2d
from oneflow.compatible.single_client.ops.nn_ops import avg_pool2d
from oneflow.compatible.single_client.ops.nn_ops import max_pool3d
from oneflow.compatible.single_client.ops.nn_ops import avg_pool3d
from oneflow.compatible.single_client.ops.nn_ops import softmax
from oneflow.compatible.single_client.ops.nn_ops import logsoftmax
from oneflow.compatible.single_client.ops.nn_ops import softmax_grad
from oneflow.compatible.single_client.ops.nn_ops import sparse_cross_entropy
from oneflow.compatible.single_client.ops.nn_ops import (
    softmax_cross_entropy_with_logits,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    sparse_softmax_cross_entropy_with_logits,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    distributed_sparse_softmax_cross_entropy_with_logits,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    sigmoid_cross_entropy_with_logits,
)
from oneflow.compatible.single_client.ops.nn_ops import random_mask_like
from oneflow.compatible.single_client.ops.nn_ops import dropout
from oneflow.compatible.single_client.ops.nn_ops import deconv2d
from oneflow.compatible.single_client.ops.nn_ops import deconv2d_torch
from oneflow.compatible.single_client.ops.nn_ops import leaky_relu
from oneflow.compatible.single_client.ops.nn_ops import elu
from oneflow.compatible.single_client.ops.nn_ops import hard_sigmoid
from oneflow.compatible.single_client.ops.nn_ops import mish
from oneflow.compatible.single_client.ops.nn_ops import swish
from oneflow.compatible.single_client.ops.nn_ops import hardswish
from oneflow.compatible.single_client.ops.nn_ops import hardtanh
from oneflow.compatible.single_client.ops.nn_ops import relu6
from oneflow.compatible.single_client.ops.nn_ops import l1_loss
from oneflow.compatible.single_client.ops.nn_ops import bce_loss
from oneflow.compatible.single_client.ops.nn_ops import bce_with_logits_loss
from oneflow.compatible.single_client.ops.nn_ops import mse_loss
from oneflow.compatible.single_client.ops.nn_ops import margin_ranking_loss
from oneflow.compatible.single_client.ops.nn_ops import triplet_margin_loss
from oneflow.compatible.single_client.ops.nn_ops import pixel_shuffle
from oneflow.compatible.single_client.ops.nn_ops import pixel_shufflev2
from oneflow.compatible.single_client.ops.nn_ops import kldivloss
from oneflow.compatible.single_client.ops.domain_ops import (
    api_fused_self_attention_query_mul_key_and_value,
)
from oneflow.compatible.single_client.ops.math_ops import relu as relu
from oneflow.compatible.single_client.ops.math_ops import tril as tril
from oneflow.compatible.single_client.ops.math_ops import (
    fused_scale_tril as fused_scale_tril,
)
from oneflow.compatible.single_client.ops.math_ops import (
    fused_scale_tril_softmax_dropout as fused_scale_tril_softmax_dropout,
)
from oneflow.compatible.single_client.ops.loss_ops import ctc_greedy_decoder
from oneflow.compatible.single_client.nn.parameter import Parameter
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.sparse import Embedding
