//#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "layers/cudnn_convolution_layer.h"
#include "layers/layer_factory.h"
namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new CuDNNConvolutionParam<Dtype>();

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
  }
  else {
    param->kernel_h_ = convolution_proto.kernel_h();
    param->kernel_w_ = convolution_proto.kernel_w();
  }
  CHECK_GT(param->kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(param->kernel_w_, 0) << "Filter dimensions cannot be zero.";

  if (!convolution_proto.has_pad_h()) {
    param->pad_h_ = param->pad_w_ = convolution_proto.pad();
  }
  else {
    param->pad_h_ = convolution_proto.pad_h();
    param->pad_w_ = convolution_proto.pad_w();
  }

  if (!convolution_proto.has_stride_h()) {
    param->stride_h_ = param->stride_w_ = convolution_proto.stride();
  }
  else {
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
void CuDNNConvolutionLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(CuDNNConvolutionData, data, data_param);
  GET_CONCRETE_POINTER(CuDNNConvolutionParam, param, param_);
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

  // Set the indexing parameters.
  param->bias_offset_ = (param->out_channels_ / param->group_);

  cudnn::createFilterDesc<Dtype>(&(param->filter_desc_),
    param->out_channels_ / param->group_, param->in_channels_ / param->group_,
    param->kernel_h_, param->kernel_w_);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  cudnn::createTensor4dDesc<Dtype>(&(param->in_descs_));
  cudnn::createTensor4dDesc<Dtype>(&(param->out_descs_));
  cudnn::createConvolutionDesc<Dtype>(&(param->conv_descs_));

  // Tensor descriptor for bias.
  if (param->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&(param->bias_desc_));
  }

  int32_t in_dim_ = param->in_channels_ * param->in_height_ * param->in_width_;
  int32_t out_dim_ = param->out_channels_ * param->conv_out_spatial_dim_;

  param->in_offset_ = in_dim_ / param->group_;
  param->out_offset_ = out_dim_ / param->group_;
  const int height = param->in_height_;
  const int width = param->in_width_;
  const int height_out = param->out_height_;
  const int width_out = param->out_width_;
  const int pad_h = param->pad_h_;
  const int pad_w = param->pad_w_;
  const int stride_h = param->stride_h_;
  const int stride_w = param->stride_w_;


  cudnn::setTensor4dDesc<Dtype>(&(param->in_descs_),
    param->num_,
    param->in_channels_ / param->group_, height, width,
    param->in_channels_ * height * width,
    height * width, width, 1);
  cudnn::setTensor4dDesc<Dtype>(&(param->out_descs_),
    param->num_,
    param->out_channels_ / param->group_, height_out, width_out,
    param->out_channels_ * param->conv_out_spatial_dim_,
    param->conv_out_spatial_dim_, width_out, 1);
  cudnn::setConvolutionDesc<Dtype>(&(param->conv_descs_), param->in_descs_,
    param->filter_desc_, pad_h, pad_w,
    stride_h, stride_w);



  // Tensor descriptor for bias.
  if (param->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&(param->bias_desc_),
      1, param->out_channels_ / param->group_, 1, 1);
  }

  // Finally, align the blob shapes in this->param_->prototype_data_ with |data_param|
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
  // To this end, we complete:
  // (1) Init the blob shapes in this->param_->prototype_model_
  // (2) Init the output blob shapes in |data_param|
  // (3) Init the input&outout blob shapes in this->param_->prototype_data_
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);
REGISTER_LAYER_CLASS(CuDNNConvolution);
}   // namespace caffe
//#endif

#if 0

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
// Check that handles have been setup before destroying.
GET_CONCRETE_POINTER(ConvolutionParam, param, param_);

for (int i = 0; i < in_descs_.size(); i++) {
  cudnnDestroyTensorDescriptor(in_descs_[i]);
  cudnnDestroyTensorDescriptor(out_descs_[i]);
  cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
}
if (param->bias_term_) {
  cudnnDestroyTensorDescriptor(bias_desc_);
}
cudnnDestroyFilterDescriptor(filter_desc_);

// for (int g = 0; g < param->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
// for (int g = 0; g < param->group_ ; g++) {
//   cudaStreamDestroy(stream_[g]);
//   cudnnDestroy(handle_[g]);
// }

cudaFree(workspaceData);
delete[] fwd_algo_;
delete[] bwd_filter_algo_;
delete[] bwd_data_algo_;
delete[] workspace_fwd_sizes_;
delete[] workspace_bwd_data_sizes_;
delete[] workspace_bwd_filter_sizes_;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::InitFromInputShape(
DataParam<Dtype>* data_param) {
ConvolutionLayer<Dtype>::InitFromInputShape(data_param);

GET_CONCRETE_POINTER(ConvolutionParam, param, param_);

// Initialize CUDA streams and cuDNN.
// stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
// handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
cudaStream_t stream;

CUDA_CHECK(cudaStreamCreate(&stream));
CUDNN_CHECK(cudnnCreate(&handle));
CUDNN_CHECK(cudnnSetStream(handle, stream));

int in_size = 1;
// Initialize algorithm arrays
fwd_algo_ = new cudnnConvolutionFwdAlgo_t[in_size];
bwd_filter_algo_ = new cudnnConvolutionBwdFilterAlgo_t[in_size];
bwd_data_algo_ = new cudnnConvolutionBwdDataAlgo_t[in_size];

// initialize size arrays
workspace_fwd_sizes_ = new size_t[in_size];
workspace_bwd_filter_sizes_ = new size_t[in_size];
workspace_bwd_data_sizes_ = new size_t[in_size];

// workspace data
workspaceSizeInBytes = 0;
workspaceData = NULL;
// workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];
workspace = new void*[param->group_];

for (size_t i = 0; i < in_size; ++i) {
  // initialize all to default algorithms
  fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
  bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
  bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
  // default algorithms don't require workspace
  workspace_fwd_sizes_[i] = 0;
  workspace_bwd_data_sizes_[i] = 0;
  workspace_bwd_filter_sizes_[i] = 0;
}

// for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
//   CUDA_CHECK(cudaStreamCreate(&stream_[g]));
//   CUDNN_CHECK(cudnnCreate(&handle_[g]));
//   CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
//   workspace[g] = NULL;
// }
for (int g = 0; g < param->group_; g++) {
  workspace[g] = NULL;
}

// Set the indexing parameters.
bias_offset_ = (param->out_channels_ / param->group_);

// Create filter descriptor.
// const int* kernel_shape_data = this->kernel_shape_.cpu_data();
// const int kernel_h = kernel_shape_data[0];
// const int kernel_w = kernel_shape_data[1];

cudnn::createFilterDesc<Dtype>(&filter_desc_,
  param->out_channels_ / param->group_, param->in_channels_ / param->group_,
  param->kernel_h_, param->kernel_w_);

// Create tensor descriptor(s) for data and corresponding convolution(s).
for (int i = 0; i < in_size; i++) {
  cudnnTensorDescriptor_t in_desc;
  cudnn::createTensor4dDesc<Dtype>(&in_desc);
  in_descs_.push_back(in_desc);
  cudnnTensorDescriptor_t out_desc;
  cudnn::createTensor4dDesc<Dtype>(&out_desc);
  out_descs_.push_back(out_desc);
  cudnnConvolutionDescriptor_t conv_desc;
  cudnn::createConvolutionDesc<Dtype>(&conv_desc);
  conv_descs_.push_back(conv_desc);
}

// Tensor descriptor for bias.
if (param->bias_term_) {
  cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
}

int32_t in_dim_ = param->in_channels_ * param->in_height_ * param->in_width_;
int32_t out_dim_ = param->out_channels_ * param->conv_out_spatial_dim_;

in_offset_ = in_dim_ / param->group_;
out_offset_ = out_dim_ / param->group_;
const int height = param->in_height_;
const int width = param->in_width_;
const int height_out = param->out_height_;
const int width_out = param->out_width_;
// const int* pad_data = this->pad_.cpu_data();
const int pad_h = param->pad_h_;
const int pad_w = param->pad_w_;
// const int* stride_data = this->stride_.cpu_data();
const int stride_h = param->stride_h_;
const int stride_w = param->stride_w_;

// Specify workspace limit for kernels directly until we have a
// planning strategy and a rewrite of Caffe's GPU memory management
size_t workspace_limit_bytes = 8 * 1024 * 1024;

for (int i = 0; i < in_size; i++) {
  cudnn::setTensor4dDesc<Dtype>(&in_descs_[i],
    param->num_,
    param->in_channels_ / param->group_, height, width,
    param->in_channels_ * height * width,
    height * width, width, 1);
  cudnn::setTensor4dDesc<Dtype>(&out_descs_[i],
    param->num_,
    param->out_channels_ / param->group_, height_out, width_out,
    param->out_channels_ * param->conv_out_spatial_dim_,
    param->conv_out_spatial_dim_, width_out, 1);
  cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], in_descs_[i],
    filter_desc_, pad_h, pad_w,
    stride_h, stride_w);

  // choose forward and backward algorithms + workspace(s)
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
    in_descs_[i],
    filter_desc_,
    conv_descs_[i],
    out_descs_[i],
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
    workspace_limit_bytes,
    &fwd_algo_[i]));

  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
    in_descs_[i],
    filter_desc_,
    conv_descs_[i],
    out_descs_[i],
    fwd_algo_[i],
    &(workspace_fwd_sizes_[i])));

  // choose backward algorithm for filter
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
    in_descs_[i], out_descs_[i], conv_descs_[i], filter_desc_,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
    workspace_limit_bytes, &bwd_filter_algo_[i]));

  // get workspace for backwards filter algorithm
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
    in_descs_[i], out_descs_[i], conv_descs_[i], filter_desc_,
    bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

  // choose backward algo for data
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle,
    filter_desc_, out_descs_[i], conv_descs_[i], in_descs_[i],
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
    workspace_limit_bytes, &bwd_data_algo_[i]));

  // get workspace size
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
    filter_desc_, out_descs_[i], conv_descs_[i], in_descs_[i],
    bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]));
}

// reduce over all workspace sizes to get a maximum to allocate / reallocate
size_t total_workspace_fwd = 0;
size_t total_workspace_bwd_data = 0;
size_t total_workspace_bwd_filter = 0;

for (size_t i = 0; i < in_size; i++) {
  total_workspace_fwd = std::max(total_workspace_fwd,
    workspace_fwd_sizes_[i]);
  total_workspace_bwd_data = std::max(total_workspace_bwd_data,
    workspace_bwd_data_sizes_[i]);
  total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
    workspace_bwd_filter_sizes_[i]);
}
// get max over all operations
size_t max_workspace = std::max(total_workspace_fwd,
  total_workspace_bwd_data);
max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
// ensure all groups have enough workspace
// size_t total_max_workspace = max_workspace *
// (this->group_ * CUDNN_STREAMS_PER_GROUP);
size_t total_max_workspace = max_workspace * param->group_;

// this is the total amount of storage needed over all groups + streams
if (total_max_workspace > workspaceSizeInBytes) {
  DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
  workspaceSizeInBytes = total_max_workspace;

  // free the existing workspace and allocate a new (larger) one
  cudaFree(this->workspaceData);

  cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
  if (err != cudaSuccess) {
    // force zero memory path
    for (int i = 0; i < in_size; i++) {
      workspace_fwd_sizes_[i] = 0;
      workspace_bwd_filter_sizes_[i] = 0;
      workspace_bwd_data_sizes_[i] = 0;
      fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
      bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    }

    // NULL out all workspace pointers
    // for (int g = 0; g < (param->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
    for (int g = 0; g < param->group_; g++) {
      workspace[g] = NULL;
    }
    // NULL out underlying data
    workspaceData = NULL;
    workspaceSizeInBytes = 0;
  }

  // if we succeed in the allocation, set pointer aliases for workspaces
  // for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
  for (int g = 0; g < param->group_; g++) {
    workspace[g] = reinterpret_cast<char *>(workspaceData)+g*max_workspace;
  }
}

// Tensor descriptor for bias.
if (param->bias_term_) {
  cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
    1, param->out_channels_ / param->group_, 1, 1);
}

cudaStreamDestroy(stream);
cudnnDestroy(handle);
}
#endif 

