/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Generated from oneflow/core/functional/functional_api.yaml. DO NOT EDIT!

#include <Python.h>

namespace oneflow {
namespace one {
namespace functional {

PyObject* add(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* amin(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sub(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* mul(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* mul_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* addcmul(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* addcmul_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* addcdiv(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* addcdiv_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* div(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* div_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* equal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* not_equal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* greater(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* greater_equal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* logical_and(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* logical_or(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* logical_not(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* logical_xor(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* less(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* less_equal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* pow(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* searchsorted(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* floor_divide(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* trunc_divide(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* max(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* min(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* median(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_max(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_min(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_sum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_nansum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_mean(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_all(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_any(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_prod(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_min_device_stage(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_max_device_stage(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_min_global_stage(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_max_global_stage(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* transpose(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* as_strided(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* select(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* swapaxes(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* swapdims(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* amax(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* permute(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* T(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* t(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reciprocal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reciprocal_no_nan(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* image_flip(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sin(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sin_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cos(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cosh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cosh_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fmod(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* log(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* log2(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* log10(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sqrt(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rsqrt(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* square(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sqrt_square_sum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* std(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* var(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rms_layer_norm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* relu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hann_window(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hardtanh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tan(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tan_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tanh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tanh_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* threshold(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* elu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* celu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gelu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gelu_with_approximate(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* glu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sigmoid(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sigmoid_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hardsigmoid(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hardshrink(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* softmax(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* log_softmax(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hardswish(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* leaky_relu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* normal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* normalization(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* normalization_add_relu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* eye(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* eye_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* erfinv(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* erfinv_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* arange(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* global_arange(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* flatten(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* argmax(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* argmin(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* argwhere(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* nonzero(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* broadcast_like(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cast(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* constant(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* global_constant(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* empty(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* global_empty(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* bernoulli(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* bernoulli_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* concat(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* bias_add(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* conv1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* conv2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* conv3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fake_quantization(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* quantization(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* min_max_observer(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* moving_average_min_max_observer(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* deconv1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* deconv2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* deconv3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* expand(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* repeat(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* repeat_interleave(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tile(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* roll(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* expand_dims(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* unsqueeze(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* squeeze(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* exp(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gather(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dim_gather(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* embedding_renorm_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* embedding(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* arg_sort(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gather_nd(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* scatternd(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tensor_scatter_nd_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* scatterndlike(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* matmul(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* mm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_mlp(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_matmul_bias_add_relu_dropout(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_matmul(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* matrix_vector_product(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tensordot(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* l1_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* mse_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* kl_div_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* nll_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* binary_cross_entropy_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* binary_cross_entropy_with_logits_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* binary_cross_entropy_with_logits_loss_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sparse_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* distributed_sparse_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sparse_softmax_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* softmax_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* softmax_cross_entropy_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* smooth_l1_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* combined_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* triplet_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* margin_ranking_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* ctc_loss(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* affine_grid(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* grid_sample(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* where(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* masked_fill(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* masked_fill_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* movedim(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tensor_split(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hsplit(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* vsplit(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* negative(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* layer_norm_affine(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* layer_norm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* group_norm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* avg_pool2d_nhwc(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* adaptive_avg_pool1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* adaptive_avg_pool2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* adaptive_avg_pool3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* max_pool1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* max_pool3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* prelu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reshape(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* view(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* contiguous(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* contiguous_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* slice_view_1d_contiguous(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* narrow(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* slice(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* slice_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* copy(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* to(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* flip(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_linear_1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_nearest_1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_nearest_2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_bilinear_2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_bicubic_2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_nearest_3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* upsample_trilinear_3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* abs(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* acos(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* acosh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* asin(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* asinh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* atan(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* atan2(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* atanh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* ceil(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* erf(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* erfc(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* expm1(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* floor(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* floor_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* lgamma(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* log1p(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* logsigmoid(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rint(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* round(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sign(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* sinh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* softplus(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* softshrink(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_hot(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* unsorted_segment_sum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tril(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* triu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* triu_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clamp(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clamp_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clamp_min(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clamp_min_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clamp_max(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clamp_max_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clip(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* clip_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* vector_norm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* matrix_norm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* norm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* inv(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* linalg_cross(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dropout(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dropout1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dropout2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dropout3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* pad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* silu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* mish(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* selu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* softsign(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* diag(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* diagonal(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* scatter(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* scatter_add(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tensor_setitem(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* avg_pool1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* avg_pool2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* avg_pool3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* minimum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* maximum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* stack(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* atleast_1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* atleast_2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* atleast_3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* hstack(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* vstack(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dstack(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* column_stack(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* row_stack(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* to_global(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* to_local(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* stream_touch(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* broadcast(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* local_all_reduce(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* local_reduce(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* select_top_n(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* identity(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* amp_white_identity(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* amp_black_identity(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reshape_like(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* reduce_sum_like(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rand(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* randn(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* randn_like(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* randint(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* randint_like(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* randperm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* unfold_tensor(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* unfold(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fold(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* split(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* unbind(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* chunk(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* split_like(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* pairwise_distance(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cosine_similarity(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* normalize(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_self_attention(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_scale_tril(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_bias_add_gelu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_bias_add_dropout(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_scale_mask_softmax(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_scale_mask_softmax_dropout(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_scale_tril_softmax_mask_scale(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_multi_head_attention_inference(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* send(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* recv(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_gather(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* ctc_greedy_decoder(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* nms(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* roi_align(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* meshgrid(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* index_select(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* decode_onerec(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* dot(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_dot_feature_interaction(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fused_cross_feature_interaction(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tensor_buffer_to_tensor(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* tensor_to_tensor_buffer(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gen_tensor_buffer(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* top_k(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* in_top_k(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cumsum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* cumprod(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_id_shuffle(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_embedding_shuffle(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_embedding_gradient_shuffle(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_lookup(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_fused_lookup(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_fused_lookup_grad(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_unique_key_value_pair(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_embedding_put(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_sgd_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_adam_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_adagrad_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* one_embedding_ftrl_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* einsum(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* pixel_shuffle(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* isnan(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* isinf(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* isfinite(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* roc_auc_score(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* pin_memory(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* fill_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rnn_tanh_cell(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rnn_relu_cell(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* lstm_cell(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gru_cell(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rnn_tanh(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* rnn_relu(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* lstm(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* gru(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* pack_padded_sequence(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* multi_tensor_sgd_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* multi_tensor_adam_update(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* trunc(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_norm_stats(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_norm_gather_stats_with_counts(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_norm_elemt(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_norm_backward_reduce(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* batch_norm_backward_elemt(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* adaptive_max_pool1d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* adaptive_max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* adaptive_max_pool3d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* exponential_(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* multinomial(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* deform_conv2d(PyObject* self, PyObject* args, PyObject* kwargs);

PyObject* bincount(PyObject* self, PyObject* args, PyObject* kwargs);

}  // namespace functional
}  // namespace one
}  // namespace oneflow
