#ifndef _LAYER_LAYER_H_
#define _LAYER_LAYER_H_
#include <cublas_v2.h>
#include <cuda.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/shape.h"
#include "dag/blob_meta.h"
#include "device/device_alternate.h"
#include "memory/device_register.h"

namespace caffe {
template <typename Dtype>
class TaskDag;
template <typename Dtype>
class DeviceRegister;
class BlobMeta;
template <typename Dtype>
class Layer {
 public:
  explicit Layer(const std::string& param_str)
    : param_str_(param_str) {}
  virtual ~Layer() {
    for (auto& it : shape_dict_) {
      delete it.second;
    }
  }

  void Setup(const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs);
  void AddShapes();
  const Shape& get_shape(const std::string& parameter_name) const;
  void ParameterReshape(const std::string& parameter_name,
    std::vector<int64_t> shape);
  const std::string& GetParameterString() const;
  const std::vector<std::string>& GetParameterNames() const;
  void PrintModelSize();
  const size_t GetModelShapeCount() const;
  void AllocateModel(void* data_ptr,
    const std::unique_ptr<DeviceRegister<Dtype>>& device_register,
    std::string name, int64_t global_id);

  virtual void SetParameterNames() {}
  virtual void LayerSetup(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs) {}
  virtual void Reshape(
    const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
    std::vector<std::shared_ptr<BlobMeta>>* outputs) {}
  virtual void Forward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {}
  virtual void Backward(
    cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
    const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
    std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
    std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {}

 protected:
  std::string param_str_;
  std::vector<std::string> parameter_names_;
  std::unordered_map<std::string, Shape*> shape_dict_;
  Layer(const Layer& other) = delete;
  Layer& operator=(const Layer& other) = delete;
};

template <typename Dtype>
inline const Shape&
Layer<Dtype>::get_shape(const std::string& parameter_name) const {
  auto it = shape_dict_.find(parameter_name);
  CHECK(it != shape_dict_.end());
  return *(it->second);
}
template <typename Dtype>
inline const std::vector<std::string>& Layer<Dtype>::GetParameterNames() const {
  return parameter_names_;
}
template <typename Dtype>
inline const std::string& Layer<Dtype>::GetParameterString() const {
  return param_str_;
}
template <typename Dtype>
inline void Layer<Dtype>::PrintModelSize() {
  for (auto& it : shape_dict_) {
    DLOG(INFO) << it.first << "_shape: " << it.second->shape_string();
  }
}

template <typename Dtype>
inline const size_t Layer<Dtype>::GetModelShapeCount() const {
  size_t sum = 0;
  for (auto& it : shape_dict_) {
    sum += it.second->count();
  }
  return sum;
}
template <typename Dtype>
inline void Layer<Dtype>::ParameterReshape(const std::string& parameter_name,
  std::vector<int64_t> shape) {
  auto it = shape_dict_.find(parameter_name);
  CHECK(it != shape_dict_.end());
  it->second->Reshape(shape);
}
}  // namespace caffe
#endif  // _LAYER_LAYER_H_
