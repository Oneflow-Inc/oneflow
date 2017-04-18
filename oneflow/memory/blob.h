#ifndef _MEMORY_BLOB_H_
#define _MEMORY_BLOB_H_
#include <cuda.h>
#include <string>
#include "common/common.h"
#include "common/shape.h"
#include "memory/memory_manager.h"
#include "proto/oneflow.pb.h"
#include <functional>
#include <google/protobuf/repeated_field.h>
#include <type_traits>

namespace oneflow {
template <typename Dtype>
class Blob {
 public:
  Blob();
  Blob(const Shape& shape, DeviceType device_type); 
  Blob(void* data_ptr, const Shape& shape, DeviceType device_type);
  ~Blob();

  void set_shape(const std::vector<int64_t>& shape) {
  }
  void set_shape(const Shape& shape) {
  }
  void Alloc(DeviceType device_type){}  // TODO(jiyuan): allocate

  const Shape& shape() const;
  Shape& mutable_shape();
  size_t byte_size() const;
  DeviceType device_type() const;
  const Dtype* data() const;
  Dtype* mutable_data() const;
  void set_data_ptr(void* ptr, bool is_own);  // for temporary lenet test

  // TODO(xcdu): data_ptr can not be reset after constructed function, for none
  //             of L-value pointer can be accessed to.
  void Reallocate(DeviceType device_type) {
  }

  using SerializationAcceptor =
    std::function<void(const std::string& blobName, 
    const std::string& data, const int64_t pos)>;
  void Serialize(
    SerializationAcceptor acceptor,
    const std::vector<std::string>& store_layer_names,
    const std::vector<int64_t>& store_layer_shapes,
    const std::vector<int64_t>& layer_seek_pos) const;

  bool Deserialize(
    size_t offset,
    const BlobProto& proto,
    const std::string& load_layer_name,
    int64_t load_layer_shape);

  BlobProto::DataType TypeMetaToDataType(int32_t id) const;

  inline int32_t id() const { return id_; }

 private:
  void* data_ptr_;
  Shape shape_;
  size_t byte_size_;
  DeviceType device_type_;
  bool own_memory_;

  int32_t id_;

  const int32_t device_id_;
  int32_t device_local_id_;
  int64_t mem_shift_;

  void Allocate();
  void PreAllocate();
  void Release();

  Blob(const Blob& other) = delete;
  Blob& operator=(const Blob& other) = delete;
};

template <typename Dtype>
inline const Shape& Blob<Dtype>::shape() const {
  return shape_;
}
template <typename Dtype>
inline Shape& Blob<Dtype>::mutable_shape() {
  return shape_;
}

template <typename Dtype>
inline size_t Blob<Dtype>::byte_size() const {
  return byte_size_;
}

template <typename Dtype>
inline DeviceType Blob<Dtype>::device_type() const {
  return device_type_;
}

template <typename Dtype>
inline const Dtype* Blob<Dtype>::data() const{
  return static_cast<const Dtype*>(data_ptr_);
}

template <typename Dtype>
inline Dtype* Blob<Dtype>::mutable_data() const {
  return static_cast<Dtype*>(data_ptr_);
}

template <typename Dtype>
inline void Blob<Dtype>::set_data_ptr(void* ptr, bool is_own) {
  if (own_memory_) {
    CUDA_CHECK(cudaFree(data_ptr_));
  }
  own_memory_ = is_own;
  data_ptr_ = ptr;
}


}  // namespace oneflow
#endif  // _MEMORY_BLOB_H_
