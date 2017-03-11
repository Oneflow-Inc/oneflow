#include "memory/blob.h"
#include <cuda.h>
#include <string>
#include "common/common.h"
#include "common/shape.h"
#include "context/one.h"
#include "context/id_map.h"
#include "device/device_alternate.h"
#include "memory/blob_copydetail.h"

namespace caffe {
template <typename Dtype>
Blob<Dtype>::Blob():
  own_memory_(false), data_ptr_(nullptr),
  device_type_(DeviceType::kUnknown), device_id_(-2), device_local_id_(-2),
  id_(0) {
  // type id is different in g++ from MSVC.
  if (typeid(Dtype) == typeid(float)) id_ = 0;
  else if (typeid(Dtype) == typeid(int32_t)) id_ = 1;
  else if (typeid(Dtype) == typeid(std::string)) id_ = 2;
  else if (typeid(Dtype) == typeid(int64_t)) id_ = 3;
  else if (typeid(Dtype) == typeid(double)) id_ = 4;
}
template<typename Dtype>  // test
Blob<Dtype>::Blob(const Shape& shape, DeviceType device_type): Blob() {
  byte_size_ = shape_.count()*sizeof(Dtype);
  Allocate();
}
template <typename Dtype>  // test
Blob<Dtype>::Blob(void *data_ptr, const Shape& shape, DeviceType device_type)
  : Blob() {
  byte_size_ = shape_.count()*sizeof(Dtype);
}
#if 0
template <typename Dtype>
Blob<Dtype>::Blob(const Shape& shape, MemoryType type,
  const int32_t device_id)
  : shape_(shape), type_(type), own_memory_(true),
  device_id_(device_id),
  device_local_id_(-2) {
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  // FIXME(jiyuan): not depend on idmap
  // device_local_id_
  //   = id_map->local_id_from_device_id(device_id_);
  // cudaSetDevice(local_device_id_);
  byte_size_ = shape_.count() * sizeof(Dtype);
  Allocate();
}
template <typename Dtype>
Blob<Dtype>::Blob(void *data_ptr, const Shape& shape, MemoryType type,
  const int32_t device_id)
  : data_ptr_(data_ptr), shape_(shape), type_(type), own_memory_(false),
  device_id_(device_id),
  device_local_id_(-2) {
  byte_size_ = shape_.count() * sizeof(Dtype);
}

#endif

template <typename Dtype>
Blob<Dtype>::~Blob() {
  if (own_memory_) {
    Release();
  }
}

template <typename Dtype>
void Blob<Dtype>::Allocate() {
  own_memory_ = true;
  //switch (type_) {
  //case MemoryType::kHostPageableMemory:
  //  data_ptr_ = malloc(byte_size_);
  //  CHECK_NOTNULL(data_ptr_);
  //  break;
  //// NOTE(xcdu): 2015.11.8 use runtime API in place of driver API for driver
  //// API depends on cuContext.
  //// TODO(xcdu): 2015.11.8 some variables in device_context may not be needed
  //// any more. delete them in future.
  //case MemoryType::kHostPinnedMemory:
  //  // CUDA_DRIVER_CHECK(cuMemAllocHost(&data_ptr_, byte_size_));
  //  CUDA_CHECK(cudaMallocHost(&data_ptr_, byte_size_));
  //  break;
  //case MemoryType::kDeviceMemory:
  //  // CUDA_DRIVER_CHECK(cuMemAlloc((CUdeviceptr*)data_ptr_, byte_size_));
  //  CUDA_CHECK(cudaMalloc(&data_ptr_, byte_size_));
  //  break;
  //default:
  //  LOG(FATAL) << "Unknown memory location";
  //  break;
  //}
}

template <typename Dtype>
void Blob<Dtype>::PreAllocate() {
  //switch (type_) {
  //case MemoryType::kHostPageableMemory:
  //  LOG(FATAL) << "kHostPageableMemory not support preallocate";
  //  break;
  //case MemoryType::kHostPinnedMemory:
  //  LOG(FATAL) << "kHostPinnedMemory not support preallocate";
  //  break;
  //case MemoryType::kDeviceMemory:
  //  break;
  //default:
  //  LOG(FATAL) << "Unknown memory location";
  //  break;
  //}
}

template <typename Dtype>
void Blob<Dtype>::Release() {
  //switch (type_) {
  //case MemoryType::kHostPageableMemory:
  //  free(data_ptr_);
  //  break;
  //case MemoryType::kHostPinnedMemory:
  //  // CUDA_DRIVER_CHECK(cuMemFreeHost(data_ptr_));
  //  CUDA_CHECK(cudaFreeHost(data_ptr_));
  //  break;
  //case MemoryType::kDeviceMemory:
  //  // CUDA_DRIVER_CHECK(cuMemFree(CUdeviceptr(data_ptr_)));
  //  CUDA_CHECK(cudaFree(data_ptr_));
  //  break;
  //default:
  //  LOG(FATAL) << "Unknown memory location";
  //}
}

template <typename Dtype>
BlobProto::DataType Blob<Dtype>::TypeMetaToDataType(int32_t id) const {
  static_assert(sizeof(int) == 4,
    "int in this compiler does not equal to 4 bytes.");
  static std::map<int32_t, BlobProto::DataType> data_type_map{
    { 0, BlobProto_DataType_FLOAT },
    { 1, BlobProto_DataType_INT32 },
    { 2, BlobProto_DataType_STRING },    
    { 3, BlobProto_DataType_INT64 },   
    { 4, BlobProto_DataType_DOUBLE },
    //{ 5, BlobProto_DataType_BOOL },
    //{ 6, BlobProto_DataType_UINT8 },
    //{ 7, BlobProto_DataType_INT8 },
    //{ 8, BlobProto_DataType_UINT16 },
    //{ 9, BlobProto_DataType_INT16 },
    //{ 10, BlobProto_DataType_FLOAT16 },
    // BYTE does not have a type meta to proto mapping: we should
    // always use uint8_t when serializing. BYTE is kept for backward
    // compatibility.
  };
  const auto it = data_type_map.find(id);
  return (it == data_type_map.end()
    ? BlobProto_DataType_UNDEFINED : it->second);
}

template <typename Dtype>
void Blob<Dtype>::Serialize(
  SerializationAcceptor acceptor,
  const std::vector<std::string>& store_layer_names,
  const std::vector<int64_t>& store_layer_shapes,
  const std::vector<int64_t>& layer_seek_pos) const {

  CHECK(store_layer_names.size() == store_layer_shapes.size());

  //if one layer is too large(size_t> 1000000), we should cut it into chunks.
  for (size_t layerBegin = 0; layerBegin < store_layer_shapes.size();
    ++layerBegin) {
    BlobProto proto;
    proto.set_name(store_layer_names[layerBegin]);
    auto layer_shape = store_layer_shapes[layerBegin];
    proto.set_shape(layer_shape);
    const BlobProto::DataType& data_type = TypeMetaToDataType(this->id());
    proto.set_data_type(data_type);
    detail::CopyToProtoAsIs(data_type, store_layer_shapes[layerBegin],
      this->data() + layerBegin, proto);
    acceptor(store_layer_names[layerBegin], 
      proto.SerializeAsString(), layer_seek_pos[layerBegin]);
  }
}

template <typename Dtype>
bool Blob<Dtype>::Deserialize(
  size_t offset,
  const BlobProto& proto,
  const std::string& load_layer_name,
  int64_t load_layer_shape) {

  CHECK(proto.name() == load_layer_name);
  CHECK(proto.shape() == load_layer_shape);

  return detail::CopyFromProtoAsIs(proto.data_type(), load_layer_shape,
      proto, this->mutable_data() + offset);
}
INSTANTIATE_CLASS(Blob);
}  // namespace caffe
