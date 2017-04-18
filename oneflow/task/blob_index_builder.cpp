#include <string>
#include <vector>
#include "task/blob_index_builder.h"
#include "task/device_register.h"
#include "task/task_param_creator.h"
#include "network/network_memory.h"

namespace oneflow {

template <typename Dtype>
BlobIndex<Dtype> BlobIndexBuilder<Dtype>::GetBlobIndex(
  const DeviceRegister<Dtype>* device_register,
  const TaskParamCreator<Dtype>* task_param_creator) {
  // First check whether the prep info already exists,
  // if yes directly return. Otherwise, get online prep info
  // and store it in task_param_prep_infos_
  auto task_id = task_param_creator->task_id();
  if (task_to_blob_index_.count(task_id)) {
    return task_to_blob_index_[task_id];
  }

  // Create param_pref_info on the fly and insert it into the dictionary
  task_to_blob_index_[task_id]
    = CreateBlobIndex();

  return task_to_blob_index_[task_id];

}

template <typename Dtype>
BlobIndex<Dtype> BlobIndexBuilder<Dtype>::CreateBlobIndex() {
  //FixMe xiaoshu
  BlobIndex<Dtype> blob_index;
  return blob_index;
}

INSTANTIATE_CLASS(BlobIndex);
INSTANTIATE_CLASS(BlobIndexBuilder);
}  // namespace oneflow
