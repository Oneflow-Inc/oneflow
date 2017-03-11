#include "task/device_register.h"
#include <string>
#include <unordered_map>
#include <memory>
#include "memory/blob.h"
#include "common/common.h"
#include "layers/base_layer.h"
#include "task/blob_index_builder.h"
#include "dag/register_info.h"
#include "net/network.h"
#include "net/network_memory.h"

/*
A DeviceRegister holds all the data (or model, depending on it is a data or model
DeviceRegister) required by a particular task. Take a convolution task as an 
example, which may contain several ConvolutionLayers, its DeviceRegister will 
include all the intermediate data Blobs. The blobs are looked up through their 
name in a dictionary of <name, blob>.
*/
namespace caffe {
template <typename Dtype>
DeviceRegister<Dtype>::DeviceRegister(void* data_ptr,
  const RegisterInfo& register_info)
  : total_byte_size_(0),
  blob_count_(0),
  data_ptr_(data_ptr),
  net_memory_(nullptr) {
  network_ = register_info.network();
  AddAllBlobs(register_info);
  if (network_) {
    RegisterNetworkMemory();
  }
  blob_index_builder_.reset(new BlobIndexBuilder<Dtype>());
}

template <typename Dtype>
DeviceRegister<Dtype>::~DeviceRegister() {
  if (network_) {
    net_memory_->Unregister();
    delete net_memory_;
  }
}

template <typename Dtype>
void DeviceRegister<Dtype>::AddAllBlobs(const RegisterInfo& register_info) {
  // 1, Add normal blobs
  auto all_blob_names = register_info.GetAllBlobNames();
  int32_t all_blob_num = all_blob_names.size();
  int64_t offset = 0;
  for (auto& blob_name : all_blob_names) {
    auto blob_shape = register_info.GetBlobShape(blob_name);
    add_non_envelope_blob(
      blob_name,
      blob_shape,
      static_cast<void*>(static_cast<Dtype*>(data_ptr_) + offset),
      register_info.device_type());
    offset += blob_shape.count();
  }
  
  // 2, Add envelope blobs
  if (register_info.has_envelope()) {
    auto envelope_names = register_info.GetEnvelopeNames();
    auto envelope_shape = register_info.GetEnvelopeShape();
    for (auto& envelope_name : envelope_names) {
      add_envelope_blob(
        envelope_name, envelope_shape, data_ptr_, register_info.device_type());
    }
  }
}

template <typename Dtype>
void DeviceRegister<Dtype>::RegisterNetworkMemory() {
  net_memory_ = GetNdspiRDMAInstance()->NewNetworkMemory();
  net_memory_->Reset(data_ptr_, total_byte_size_);
  net_memory_->Register();
}

template <typename Dtype>
void* DeviceRegister<Dtype>::data_ptr() const {
  return data_ptr_;
}

template <typename Dtype>
NetworkMemory* DeviceRegister<Dtype>::network_memory() const {
  CHECK(network_);
  return net_memory_;
}

template <typename Dtype>
MemoryDescriptor* DeviceRegister<Dtype>::memory_descriptor() const {
  CHECK(remote_);
  // TODO(jiyuan): initialize the content of |memory_descriptor_|.
  return memory_descriptor_;
}

template <typename Dtype>
bool DeviceRegister<Dtype>::is_local_network() const {
  return network_;
}

template <typename Dtype>
bool DeviceRegister<Dtype>::is_remote_network() const {
  return remote_;
}

template <typename Dtype>
const int32_t DeviceRegister<Dtype>::blob_count() const {
  return blob_count_;
}

template <typename Dtype>
int64_t DeviceRegister<Dtype>::total_byte_size() const {
  return total_byte_size_;
}

template <typename Dtype>
inline const bool
DeviceRegister<Dtype>::contains_blob_name(const std::string& name) const {
  auto it = blob_dict_.find(name);
  if (it != blob_dict_.end()) return true;
  return false;
}

template <typename Dtype>
inline const std::unique_ptr<Blob<Dtype>>&
DeviceRegister<Dtype>::get_blob_by_name(const std::string& name) const {
  auto it = blob_dict_.find(name);
  CHECK(it != blob_dict_.end()) << "Blob name: "
    << name << " does not exist!";
  return it->second;
}

template <typename Dtype>
void DeviceRegister<Dtype>::add_non_envelope_blob(const std::string& name,
    const Shape& shape,
    void* blob_ptr,
    DeviceType device_type) {
  add_blob(name, shape, blob_ptr, device_type);
  blob_count_++;
  total_byte_size_ += shape.count() * sizeof(Dtype);
}

template <typename Dtype>
void DeviceRegister<Dtype>::add_envelope_blob(const std::string& name,
  const Shape& shape,
  void* blob_ptr,
  DeviceType device_type) {
  add_blob(name, shape, blob_ptr, device_type);
}

template <typename Dtype>
void DeviceRegister<Dtype>::add_blob(const std::string& name,
  const Shape& shape,
  void* blob_ptr,
  DeviceType device_type) {
  CHECK_EQ(blob_dict_.count(name), 0);
  std::unique_ptr<Blob<Dtype>> blob(
    new Blob<Dtype>(blob_ptr, shape, device_type));
  blob_dict_.insert(std::make_pair(name, std::move(blob)));
}


template <typename Dtype>
BlobIndex<Dtype> DeviceRegister<Dtype>::get_blob_index(
  const TaskParamCreator<Dtype>* task_param_creator) const {
  return blob_index_builder_->GetBlobIndex(this, task_param_creator);
}

INSTANTIATE_CLASS(DeviceRegister);
}  // namespace caffe
