#ifndef _LAYERS_COMMON_LAYERS_H_
#define _LAYERS_COMMON_LAYERS_H_
#include <glog/logging.h>

#include <string>
#include <vector>

#include "layers/layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const std::string& param_str)
    : Layer(param_str) {
      LOG(INFO) << "Creating SoftmaxLayer...";
      ParseProtoFromStringOrDie(param_str_, &softmax_param_);
    }
  explicit SoftmaxLayer(const std::string& param_str,
    const std::shared_ptr<SoftmaxParameter> softmax_param) :
    Layer(param_str), softmax_param_(*softmax_param) {
    LOG(INFO) << "Creating ConvolutionLayer...";
  }
  virtual ~SoftmaxLayer() {}
  virtual void SetParameterNames() {
    parameter_names_.assign({ "sum_multiplier", "scale" });
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
  SoftmaxParameter softmax_param_;

  int32_t outer_num_;
  int32_t inner_num_;
  int32_t softmax_axis_;

  SoftmaxLayer(const SoftmaxLayer& other) = delete;
  SoftmaxLayer& operator=(const SoftmaxLayer& other) = delete;
};

}  // namespace caffe
#endif  // _LAYERS_COMMON_LAYERS_H_
