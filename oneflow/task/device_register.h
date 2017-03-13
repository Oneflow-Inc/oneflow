#ifndef _MEMORY_DEVICE_REGISTER_H_
#define _MEMORY_DEVICE_REGISTER_H_
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "memory/blob.h"
#include "layers/base_layer.h"
#include "task/blob_index_builder.h"
/*
A DeviceRegister holds all the data (or model, depending on it is a data or
as an model DeviceRegister) required by a particular task. Take a convolution
task example, which may contain several ConvolutionLayers, its DeviceRegister
will include all the intermediate data Blobs. The blobs are looked up through
their name in a dictionary of <name, blob>.

A task may have several registers, we already collect all the RegisterInfo of
each task. DeviceRegister is created according to a RegisterInfo.

DeviceRegister should provide some indexing structure facilitating the
construction of DataParam and ModelParam, so that we don't need to look up the
dictionary from name to blob every time while preparing DataParam and
ModelParam for the operators inside a task.
*/

namespace caffe {
template <typename Dtype>
class TaskParamCreator;

class RegisterInfo;

class NetworkMemory;

class MemoryDescriptor;

template <typename Dtype>
class DeviceRegister {
 public:
   DeviceRegister(void* data_ptr, const RegisterInfo& register_info);
  ~DeviceRegister();

  BlobIndex<Dtype> get_blob_index(
    const TaskParamCreator<Dtype>* task_param_creator) const;

  void* data_ptr() const;
  int64_t total_byte_size() const;
  const int32_t blob_count() const;
  const std::unique_ptr<Blob<Dtype>>& get_blob_by_name(
    const std::string& name) const;
  const bool contains_blob_name(const std::string& name) const;

  bool is_local_network() const;
  NetworkMemory* network_memory() const;
  bool is_remote_network() const;
  MemoryDescriptor* memory_descriptor() const;

private:
  void* data_ptr_;
  int64_t total_byte_size_;
  int32_t blob_count_;  // not include the envelope blob

  bool network_;
  NetworkMemory* net_memory_;

  bool remote_{ false };
  MemoryDescriptor* memory_descriptor_;

  std::unordered_map<std::string, std::unique_ptr<Blob<Dtype>>> blob_dict_;
  std::shared_ptr<BlobIndexBuilder<Dtype>> blob_index_builder_;

  // Allocate normal blobs and envelope blobs
  void AddAllBlobs(const RegisterInfo& register_info);

  void RegisterNetworkMemory();

  void add_non_envelope_blob(const std::string& name,
    const Shape& shape,
    void* blob_ptr,
    DeviceType device_type);

  void add_envelope_blob(const std::string& name,
    const Shape& shape,
    void* blob_ptr,
    DeviceType device_type);

  void add_blob(const std::string& name,
    const Shape& shape,
    void* blob_ptr,
    DeviceType device_type);

  DeviceRegister(const DeviceRegister& other) = delete;
  DeviceRegister& operator=(const DeviceRegister& other) = delete;
};
}  // namespace caffe
#endif  // _MEMORY_DEVICE_REGISTER_H_
