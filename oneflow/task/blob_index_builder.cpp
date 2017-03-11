#include <string>
#include <vector>
#include "dag/task_dag.h"
#include "task/blob_index_builder.h"
#include "task/device_register.h"
#include "task/task_param_creator.h"
#include "net/network_memory.h"

namespace caffe {

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
    = CreateBlobIndex(device_register, task_param_creator);

  return task_to_blob_index_[task_id];

}

template <typename Dtype>
BlobIndex<Dtype> BlobIndexBuilder<Dtype>::CreateBlobIndex(
  const DeviceRegister<Dtype>* device_register,
  const TaskParamCreator<Dtype>* task_param_creator) {

  BlobIndex<Dtype> blob_index;
  // For each blob needed by this task, get its name in TaskDag, if it is in 
  // current DeviceRegister, add it into the ParamPrepInfo.
  auto& layer_blobs = task_param_creator->layer_blobs_in_execution();
  for (auto& layer_blob : layer_blobs) {
    auto register_blob
      = task_param_creator->register_blob_from_layer_blob(layer_blob);
    if (device_register->contains_blob_name(register_blob)) {
      int32_t index = task_param_creator->index_of_layer_blob(layer_blob);
      if (task_param_creator->is_net_receiver()) {
        // Net receiver
        if (strings::EndsWith(layer_blob, "envelope")) {
          // Envelope blob
          if (device_register->is_local_network()) {
            Blob<Dtype>* local_network_memory
              = reinterpret_cast<Blob<Dtype>*>(device_register->network_memory());
            blob_index.set_blob(index, local_network_memory);
          }
          if (device_register->is_remote_network()) {
            Blob<Dtype>* remote_memory_descriptor
              = reinterpret_cast<Blob<Dtype>*>(device_register->memory_descriptor());
            blob_index.set_blob(index, remote_memory_descriptor);
          }
        } else {
          // Net receiver, not envelope blob
          blob_index.set_blob(index,
            device_register->get_blob_by_name(register_blob).get());
        }
      } else {
        // Not net receiver
        blob_index.set_blob(index,
          device_register->get_blob_by_name(register_blob).get());
      }
    }
  }
  return blob_index;
}

INSTANTIATE_CLASS(BlobIndex);
INSTANTIATE_CLASS(BlobIndexBuilder);
}  // namespace caffe
