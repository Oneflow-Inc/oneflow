#ifndef _TASK_BLOB_INDEX_BUILDER_H_
#define _TASK_BLOB_INDEX_BUILDER_H_

#include <stdint.h>
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "memory/blob.h"

namespace oneflow {
template <typename Dtype>
class TaskParamCreator;

template <typename Dtype>
class DeviceRegister;

template <typename Dtype>
class DataModelBaseParam;

template <typename Dtype>
class BlobIndex {
public:
  BlobIndex() = default;

  void set_blob(int32_t index, Blob<Dtype>* blob) {
    blob_index_.push_back({ index, blob });
  }

  const std::vector<std::pair<int32_t, Blob<Dtype>*>>& get_index() const {
    return blob_index_;
  }

private:
  // In the pair, the key indicates the blob's index in TaskParam
  std::vector<std::pair<int32_t, Blob<Dtype>*>> blob_index_;
};

template <typename Dtype>
class BlobIndexBuilder {
public:
  BlobIndexBuilder() = default;
  ~BlobIndexBuilder() = default;

  BlobIndex<Dtype> GetBlobIndex(
    const DeviceRegister<Dtype>* device_register,
    const TaskParamCreator<Dtype>* task_param_creator);
public:

private:
  // From task_id to blob_index
  std::unordered_map<int32_t, BlobIndex<Dtype>> task_to_blob_index_;

  BlobIndex<Dtype> CreateBlobIndex();
};

}  // namespace oneflow
#endif  // _TASK_BLOB_INDEX_BUILDER_H_
