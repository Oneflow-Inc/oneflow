#include <cstdint>
#include <string>
#include <vector>
#include "layers/convolution_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "memory/blob.h"

namespace caffe {
template <typename Dtype>
void ConvolutionLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new ConvolutionParam<Dtype>();

  ConvolutionProto convolution_proto;
  ParseProtoFromStringOrDie(proto_param_, &convolution_proto);

  CHECK(!convolution_proto.has_kernel_size() !=
    !(convolution_proto.has_kernel_h() && convolution_proto.has_kernel_w()))
    << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK((!convolution_proto.has_pad() && convolution_proto.has_pad_h()
    && convolution_proto.has_pad_w())
    || (!convolution_proto.has_pad_h() && !convolution_proto.has_pad_w()))
    << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!convolution_proto.has_stride() && convolution_proto.has_stride_h()
    && convolution_proto.has_stride_w())
    || (!convolution_proto.has_stride_h()
    && !convolution_proto.has_stride_w()))
    << "Stride is stride OR stride_h and stride_w are required.";

  if (convolution_proto.has_kernel_size()) {
    param->kernel_h_ = param->kernel_w_ = convolution_proto.kernel_size();
  } else {
    param->kernel_h_ = convolution_proto.kernel_h();
    param->kernel_w_ = convolution_proto.kernel_w();
  }
  CHECK_GT(param->kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(param->kernel_w_, 0) << "Filter dimensions cannot be zero.";

  if (!convolution_proto.has_pad_h()) {
    param->pad_h_ = param->pad_w_ = convolution_proto.pad();
  } else {
    param->pad_h_ = convolution_proto.pad_h();
    param->pad_w_ = convolution_proto.pad_w();
  }

  if (!convolution_proto.has_stride_h()) {
    param->stride_h_ = param->stride_w_ = convolution_proto.stride();
  } else {
    param->stride_h_ = convolution_proto.stride_h();
    param->stride_w_ = convolution_proto.stride_w();
  }

  param->out_channels_ = convolution_proto.num_output();  // TODO: change name in proto
  CHECK_GT(param->out_channels_, 0);

  param->group_ = convolution_proto.group();
  CHECK_EQ(param->out_channels_ % param->group_, 0)
    << "Number of outputs should be multiples of group";
  param->bias_term_ = convolution_proto.bias_term();

  param_ = param;
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(ConvolutionData, data, data_param);
  GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
  auto model_param = param->mutable_model_param();
  GET_CONCRETE_POINTER(ConvolutionModel, model, model_param);

  const Shape& in_shape = data->in->shape();
  data->in_diff->set_shape(in_shape);

  CHECK_EQ(in_shape.num_axes(), 4)
    << "Input must have 4 axes, corresponding to (num, channels, height, width)";

  param->in_channels_ = in_shape.channels();
  CHECK_EQ(param->in_channels_ % param->group_, 0);
  param->num_ = in_shape.num();
  param->in_height_ = in_shape.height();
  param->in_width_ = in_shape.width();

  // weight shape
  std::vector<int64_t> weight_shape{
    param->out_channels_,
    param->in_channels_ / param->group_,
    param->kernel_h_,
    param->kernel_w_ };
  // weight & weight_diff
  model->weight->set_shape(weight_shape);
  model->weight_diff->set_shape(weight_shape);

  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  param->is_1x1_
    = param->kernel_w_ == 1 && param->kernel_h_ == 1
    && param->stride_h_ == 1 && param->stride_w_ == 1
    && param->pad_h_ == 0 && param->pad_w_ == 0;

  // infer the height and width of output
  param->out_height_
    = (param->in_height_ + 2 * param->pad_h_ - param->kernel_h_) / param->stride_h_ + 1;
  param->out_width_
    = (param->in_width_ + 2 * param->pad_w_ - param->kernel_w_) / param->stride_w_ + 1;
  // output shape
  std::vector<int64_t> out_shape{
    param->num_, param->out_channels_, param->out_height_, param->out_width_ };
  data->out->set_shape(out_shape);
  data->out_diff->set_shape(out_shape);

  param->conv_out_spatial_dim_ = param->out_height_ * param->out_width_;
  param->kernel_dim_ = param->in_channels_ * param->kernel_h_ * param->kernel_w_;
  param->weight_offset_
    = param->out_channels_ * param->kernel_dim_ / param->group_ / param->group_;
  param->col_offset_
    = param->kernel_dim_ * param->conv_out_spatial_dim_ / param->group_;
  param->output_offset_
    = param->out_channels_ * param->conv_out_spatial_dim_ / param->group_;

  // bias & bias_diff & bias_multiplier
  if (param->bias_term_) {
    std::vector<int64_t> bias_shape{ 1, param->out_channels_ };
    model->bias->set_shape(bias_shape);
    model->bias_diff->set_shape(bias_shape);

    std::vector<int64_t> bias_multiplier_shape{
      1, param->out_height_ * param->out_width_ };
    model->bias_multiplier->set_shape(bias_multiplier_shape);
  }

  // col_buf
  std::vector<int64_t> col_buffer_shape{
    1, param->kernel_dim_, param->out_height_, param->out_width_ };
  data->col_buf->set_shape(col_buffer_shape);
  
  // Finally, align the blob shapes in this->param_->prototype_data_ with |data_param|
  param_->mutable_data_param()->AlignBlobShapes(*data_param);

  // To this end, we complete:
  // (1) Init the blob shapes in this->param_->prototype_model_
  // (2) Init the output blob shapes in |data_param|
  // (3) Init the input&outout blob shapes in this->param_->prototype_data_
}
INSTANTIATE_CLASS(ConvolutionLayer);
REGISTER_LAYER_CLASS(Convolution);
}  // namespace caffe
