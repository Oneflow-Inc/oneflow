#ifndef _LAYERS_CONVOLUTION_LAYER_H_
#define _LAYERS_CONVOLUTION_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "math/im2col.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class ConvolutionData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  Blob<Dtype>* col_buf{ nullptr };
  explicit ConvolutionData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
    DATA_REGISTER_BLOB(layer_name, col_buf, BlobType::kOther);
  }
};

template <typename Dtype>
class ConvolutionModel : public ModelParam<Dtype> {
public:
  Blob<Dtype>* weight{ nullptr };
  Blob<Dtype>* weight_diff{ nullptr };
  Blob<Dtype>* bias{ nullptr };
  Blob<Dtype>* bias_diff{ nullptr };
  Blob<Dtype>* bias_multiplier{ nullptr };

  explicit ConvolutionModel(const std::string& layer_name) {
    MODEL_REGISTER_BLOB(layer_name, weight, BlobType::kModel);
    MODEL_REGISTER_BLOB(layer_name, weight_diff, BlobType::kModelDiff);
    MODEL_REGISTER_BLOB(layer_name, bias, BlobType::kModel);
    MODEL_REGISTER_BLOB(layer_name, bias_diff, BlobType::kModelDiff);
    MODEL_REGISTER_BLOB(layer_name, bias_multiplier, BlobType::kTemp);
  }
};

template <typename Dtype>
class ConvolutionParam : public LayerParam<Dtype> {
public:
  // Parameters init from proto
  int32_t kernel_h_, kernel_w_;
  int32_t pad_h_, pad_w_;
  int32_t stride_h_, stride_w_;
  int32_t out_channels_;  // num_output_;
  int32_t group_;
  bool bias_term_;

  // Parameters init from input's shape
  int32_t in_channels_;
  int32_t num_;
  int32_t in_height_, in_width_;

  // Facility parameters inferred from above
  bool is_1x1_;
  int32_t out_height_, out_width_;
  int32_t conv_out_spatial_dim_;
  int32_t kernel_dim_;
  int32_t weight_offset_;
  int32_t col_offset_;
  int32_t output_offset_;

  explicit ConvolutionParam() {}
};

template <typename Dtype>
class ConvolutionLayer : public BaseLayer<Dtype> {
public:
  explicit ConvolutionLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new ConvolutionData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new ConvolutionModel<Dtype>(layer_name_);
  }

  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  inline void conv_im2col_gpu(cudaStream_t cuda_stream, const Dtype* data,
    Dtype* col_buff) const {
    GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
    im2col_gpu(
      cuda_stream, data, param->in_channels_, param->in_height_,
      param->in_width_, param->kernel_h_, param->kernel_w_, param->pad_h_,
      param->pad_w_, param->stride_h_, param->stride_w_, col_buff);
  }

  inline void conv_col2im_gpu(cudaStream_t cuda_stream, const Dtype* col_buff,
    Dtype* data) const {
    GET_CONCRETE_POINTER(ConvolutionParam, param, param_);
    col2im_gpu(
      cuda_stream, col_buff, param->in_channels_, param->in_height_,
      param->in_width_, param->kernel_h_, param->kernel_w_, param->pad_h_,
      param->pad_w_, param->stride_h_, param->stride_w_, data);
  }

  void forward_gpu_gemm(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* input, const Dtype* weights, const Blob<Dtype>* col_buffer_,
    Dtype* output, bool skip_im2col) const;
  void forward_gpu_bias(cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* bias, const Dtype* bias_multiplier_, Dtype* output) const;
  void weight_gpu_gemm(cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* input, const Dtype* output, const Blob<Dtype>* col_buffer_,
    Dtype* weights) const;
  void backward_gpu_gemm(cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* output, const Dtype* weights, const Blob<Dtype>* col_buffer_,
    Dtype* input) const;

  ConvolutionLayer(const ConvolutionLayer& other) = delete;
  ConvolutionLayer& operator=(const ConvolutionLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_CONVOLUTION_LAYER_H_