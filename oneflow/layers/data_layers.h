#ifndef _LAYERS_DATA_LAYERS_H_
#define _LAYERS_DATA_LAYERS_H_
#include <glog/logging.h>
#include "layers/layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(const std::string& param_str)
  : Layer(param_str) {
    LOG(INFO) << "Creating DataLayer...";
    ParseProtoFromStringOrDie(param_str_, &data_param_);
  }
  virtual ~DataLayer() {}
  virtual void LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  virtual void Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);

 protected:
  DataParameter data_param_;

  DataLayer(const DataLayer& other) = delete;
  DataLayer& operator=(const DataLayer& other) = delete;
};

template <typename Dtype>
class DataPreloadLayer : public Layer<Dtype> {
 public:
  explicit DataPreloadLayer(const std::string& param_str)
  : Layer(param_str) {
    LOG(INFO) << "Creating DataLayer...";
    ParseProtoFromStringOrDie(param_str_, &data_param_);
  }
  virtual ~DataPreloadLayer() {}
  virtual void LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs,
    uint32_t preload_size);
  virtual void Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);

 protected:
  DataParameter data_param_;
  uint32_t preload_size_;

  DataPreloadLayer(const DataPreloadLayer& other) = delete;
  DataPreloadLayer& operator=(const DataPreloadLayer& other) = delete;
};

template <typename Dtype>
class BoxingLayer : public Layer<Dtype> {
 public:
  explicit BoxingLayer(const std::string& param_str)
    : Layer(param_str) {
      LOG(INFO) << "Creating BoxingLayer...";
    }
  virtual ~BoxingLayer() {}
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
   int32_t parallel_num_;

  BoxingLayer(const BoxingLayer& other) = delete;
  BoxingLayer& operator=(const BoxingLayer& other) = delete;
};

enum class DataCopyType {
  kD2H = 0,
  kH2D
};

template <typename Dtype>
class DataCopyLayer : public Layer<Dtype> {
 public:
  explicit DataCopyLayer(const std::string& param_str)
    : Layer(param_str) {
      LOG(INFO) << "Creating DataCopyLayer...";
    }
  virtual ~DataCopyLayer() {}
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
   DataCopyType type_;

  DataCopyLayer(const DataCopyLayer& other) = delete;
  DataCopyLayer& operator=(const DataCopyLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_DATA_LAYERS_H_