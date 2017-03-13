#ifndef _LAYERS_VISION_LAYERS_H_
#define _LAYERS_VISION_LAYERS_H_
#include <glog/logging.h>

#include <string>
#include <vector>

#include "layers/layer.h"
#include "math/im2col.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionLayer(const std::string& param_str): Layer(param_str) {
      LOG(INFO) << "Creating ConvolutionLayer...";
      ParseProtoFromStringOrDie(param_str_, &convolution_param_);
    }
  // NOTE(xcdu): load parameter for test
  explicit ConvolutionLayer(const std::string& param_str,
    const std::shared_ptr<ConvolutionParameter> convolution_param):
    Layer(param_str), convolution_param_(*convolution_param) {
    LOG(INFO) << "Creating ConvolutionLayer...";
  }
  virtual ~ConvolutionLayer() {}
  virtual void SetParameterNames() {
    parameter_names_.assign({ "weight", "weight_diff", "bias", "bias_diff",
      "bias_multiplier", "col_buffer" });
  }
  virtual void LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Forward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* outputs);
  virtual void Backward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* inputs);

 protected:
  ConvolutionParameter convolution_param_;

  int32_t kernel_h_, kernel_w_;
  int32_t stride_h_, stride_w_;
  int32_t num_;
  int32_t channels_;
  int32_t pad_h_, pad_w_;
  int32_t height_, width_;
  int32_t group_;
  int32_t num_output_;
  int32_t height_out_, width_out_;
  bool bias_term_;
  bool is_1x1_;

  int32_t conv_out_channels_;
  int32_t conv_in_channels_;
  int32_t conv_out_spatial_dim_;
  int32_t conv_in_height_;
  int32_t conv_in_width_;
  int32_t kernel_dim_;
  int32_t weight_offset_;
  int32_t col_offset_;
  int32_t output_offset_;

  inline void compute_output_shape() {
    height_out_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  }

  inline void conv_im2col_gpu(cudaStream_t cuda_stream, const Dtype* data,
    Dtype* col_buff) {
    im2col_gpu(cuda_stream, data, conv_in_channels_, conv_in_height_,
    conv_in_width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
    col_buff);
  }

  inline void conv_col2im_gpu(cudaStream_t cuda_stream, const Dtype* col_buff,
    Dtype* data) {
    col2im_gpu(cuda_stream, col_buff, conv_in_channels_, conv_in_height_,
      conv_in_width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
      stride_w_, data);
  }

  void forward_gpu_gemm(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* input, const Dtype* weights, const Blob<Dtype>* col_buffer_,
    Dtype* output, bool skip_im2col);
  void forward_gpu_bias(cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* bias, const Dtype* bias_multiplier_, Dtype* output);
  void weight_gpu_gemm(cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* input, const Dtype* output, const Blob<Dtype>* col_buffer_,
    Dtype* weights);
  void backward_gpu_gemm(cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const Dtype* output, const Dtype* weights, const Blob<Dtype>* col_buffer_,
    Dtype* input);

  ConvolutionLayer(const ConvolutionLayer& other) = delete;
  ConvolutionLayer& operator=(const ConvolutionLayer& other) = delete;
};

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const std::string& param_str)
    : Layer(param_str) {
      LOG(INFO) << "Creating PoolingLayer...";
      ParseProtoFromStringOrDie(param_str_, &pooling_param_);
    }
  // for test
  explicit PoolingLayer(const std::string& param_str,
    const std::shared_ptr<PoolingParameter> pooling_param) :
    Layer(param_str), pooling_param_(*pooling_param) {
    LOG(INFO) << "Creating PoolingLayer...";
  }
  virtual ~PoolingLayer() {}
  virtual void SetParameterNames() {
    parameter_names_.assign({ "idx" });
  }
  virtual void LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Forward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* outputs);
  virtual void Backward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* inputs);

 protected:
  PoolingParameter pooling_param_;

  int32_t kernel_h_, kernel_w_;
  int32_t stride_h_, stride_w_;
  int32_t pad_h_, pad_w_;
  int32_t channels_;
  int32_t height_, width_;
  int32_t pooled_height_, pooled_width_;
  bool global_pooling_;

  PoolingLayer(const PoolingLayer& other) = delete;
  PoolingLayer& operator=(const PoolingLayer& other) = delete;
};

template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const std::string& param_str)
  : Layer(param_str) {
    LOG(INFO) << "Creating InnerProductLayer...";
    ParseProtoFromStringOrDie(param_str_, &innerproduct_param_);
  }
  // Note(xcdu): for test
  explicit InnerProductLayer(const std::string& param_str,
    const std::shared_ptr<InnerProductParameter> innerproduct_param):
    Layer(param_str), innerproduct_param_(*innerproduct_param) {
    LOG(INFO) << "Creating InnerProductLayer...";
  }
  virtual ~InnerProductLayer() {}
  virtual void SetParameterNames() {
    parameter_names_.assign({ "weight", "weight_diff", "bias", "bias_diff",
      "bias_multiplier" });
  }
  virtual void LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Forward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* outputs);
  virtual void Backward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* inputs);

 protected:
  InnerProductParameter innerproduct_param_;
  int32_t M_;
  int32_t K_;
  int32_t N_;
  bool bias_term_;

  InnerProductLayer(const InnerProductLayer& other) = delete;
  InnerProductLayer& operator=(const InnerProductLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_VISION_LAYERS_H_
