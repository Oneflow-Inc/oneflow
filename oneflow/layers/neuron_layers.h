#ifndef _LAYERS_NEURON_LAYERS_H_
#define _LAYERS_NEURON_LAYERS_H_
#include <glog/logging.h>

#include <string>
#include <vector>

#include "layers/layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class ReLULayer : public Layer<Dtype> {
 public:
  explicit ReLULayer(const std::string& param_str)
    : Layer(param_str) {
      LOG(INFO) << "Creating ReLULayer...";
    }
  virtual ~ReLULayer() {}
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
  ReLULayer(const ReLULayer& other) = delete;
  ReLULayer& operator=(const ReLULayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_NEURON_LAYERS_H_
