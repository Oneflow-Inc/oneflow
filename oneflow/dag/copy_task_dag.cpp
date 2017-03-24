#include "dag/copy_task_dag.h"
#include <string>
#include <utility>
#include <vector>
#include "common/common.h"
#include "context/one.h"
#include "dag/node_meta.h"
#include "dag/actor_dag.h"
#include "dag/compute_task_dag.h"
#include "dag/boxing_task_dag.h"
#include "dag/dag_builder.h"
#include "task/job_manager.h"
#include "layers/base_layer.h"
#include "layers/layer_factory.h"

namespace oneflow {
template <typename Dtype>
CopyTaskDag<Dtype>::CopyTaskDag(const DagBuilder<Dtype>& dag_builder,
  TaskType type, int32_t task_id, PathType path_type,
  const std::string& actor_name, bool is_forward) : TaskDag(
  dag_builder, type, task_id, path_type, actor_name, is_forward) {
}
template <typename Dtype>
CopyTaskDag<Dtype>::~CopyTaskDag() {}

template <typename Dtype>
void CopyTaskDag<Dtype>::InitH2D() {
  bool in_copy = strings::Contains(name_, "in_copy");
  is_h2d_ = (is_forward_ && in_copy) || (!is_forward_ && !in_copy);
}
template <typename Dtype>
void CopyTaskDag<Dtype>::BuildForward() {
  // To build Copy TaskDag, we need to memorize the compute pipe it serves.
  // For in_copy actor, the number of blobs to be copied and the shapes of the
  // blobs depend on the input blobs consumed by the computation task.
  // For out_copy actor, the number of blobs to be copied and the shapes of the
  // blobs depend on the output blobs produced by the computation task.
  // In the forward pass, the in_copy actor performs copy_h2d, while in the
  // backward pass, the in_copy actor performs copy_d2h. Similarly, in the
  // forward pass, the out_copy actor performs copy_d2h, while in the backward
  // pass, the out_copy actor performs copy_h2d. While allocating memory for
  // copy actor, note that, for copy_h2d, the actor does not own the host
  // memory, however, for copy_d2h, the actor does own the host memory. For
  // both copy_h2d and copy_d2h, the actor does not own the device memory.

  // While building CopyActor, we only need to know the number of blobs needed
  // to be copied. While setting up CopyActor, we will need to know the shape
  // of the blobs.
  InitH2D();

  auto actor_dag = dag_builder_.actor_dag();
  const std::string forward_in_copy_prefix = "forward_in_copy";

  std::vector<std::string> logical_blobs = GetLogicalBlobsNeedCopied();
  std::string pipe_name = actor_dag->GetPipeNameFromActor(name_);
  std::string layer_type = "Copy";
  std::string layer_name = pipe_name;
  std::string proto_str = BuildProtoString(logical_blobs.size());
  auto input_task_blobs = BuildInputTaskBlobs(logical_blobs);
  auto output_task_blobs = BuildOutputTaskBlobs(logical_blobs);

  auto layer_node = AddOpNode(layer_name, layer_type, proto_str);
  auto layer = layer_node->op()->layer();

  // Create input & output blob nodes. In copy task, each blob occurs twice, one
  // as in and the other as out. We let the blob node on device use the same
  // name as that in LogicalDag, while add a prefix to the blob name on host
  // to distinguish the name.
  std::vector<DataNode<BlobMeta>*> input_nodes;
  auto input_vars = layer->GetInputVars();
  CHECK(input_vars.size() == logical_blobs.size());
  int32_t idx = 0;
  for (auto& input_var : input_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
    auto task_blob = input_task_blobs[idx];
    auto logical_blob = logical_blobs[idx];
    input_nodes.push_back(AddDataNode(task_blob));
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, true);
    ++idx;
  }
  std::vector<DataNode<BlobMeta>*> output_nodes;
  auto output_vars = layer->GetOutputVars();
  CHECK(output_vars.size() == logical_blobs.size());
  idx = 0;
  for (auto& output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    auto task_blob = output_task_blobs[idx];
    auto logical_blob = logical_blobs[idx];
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    output_nodes.push_back(AddDataNode(task_blob));
    idx++;
  }
  for (auto& input_node : input_nodes) {
    layer_node->AddParent(input_node);
  }
  for (auto& output_node : output_nodes) {
    output_node->AddParent(layer_node);
  }
}

template <typename Dtype>
std::vector<std::string> CopyTaskDag<Dtype>::GetLogicalBlobsNeedCopied() const {
  auto actor_dag = dag_builder_.actor_dag();
  std::vector<std::string> compute_actor_names;
  if (is_h2d_) {
    compute_actor_names = actor_dag->GetSucceedingOpNodeNames(name_);
  } else {
    compute_actor_names = actor_dag->GetPrecedingOpNodeNames(name_);
  }
  CHECK_EQ(compute_actor_names.size(), 1);
  auto compute_actor_name = compute_actor_names[0];
  auto compute_actor_node = actor_dag->GetOpNode(compute_actor_name);
  auto compute_task_id = compute_actor_node->op()->task_id();

  // Get the raw pointer and convert it to a pointer to compute task_dag
  auto raw_task_dag = dag_builder_.GetTaskDag(compute_task_id);
  auto compute_task_dag
    = std::dynamic_pointer_cast<ComputeTaskDag<Dtype>>(raw_task_dag);
  CHECK_NOTNULL(compute_task_dag.get());
  CHECK(compute_task_dag->task_type() == TaskType::kComputeTask);

  // Lookup the ComputeTaskDag it serves to get the blobs needed to be copied
  std::vector<std::string> logical_blobs_copied;
  if (is_h2d_) {
    logical_blobs_copied = compute_task_dag->GetInputLogicalBlobs();
  } else {
    logical_blobs_copied = compute_task_dag->GetOutputLogicalBlobs();
  }
  CHECK_GE(logical_blobs_copied.size(), 1);
  return logical_blobs_copied;
}

template <typename Dtype>
std::string CopyTaskDag<Dtype>::BuildProtoString(int32_t blob_num) const {
  CopyProto copy_proto;
  copy_proto.set_num(blob_num);
  CopyType copy_type = is_h2d_ ? ForwardH2D : ForwardD2H;
  copy_proto.set_copy_type(copy_type);
  std::string copy_proto_str = copy_proto.DebugString();
  return copy_proto_str;
}

template <typename Dtype>
std::vector<std::string> CopyTaskDag<Dtype>::BuildInputTaskBlobs(
  const std::vector<std::string>& logical_blobs_copied) const {
  if (is_h2d_) {
    std::vector<std::string> input_blobs;
    for (auto& blob : logical_blobs_copied) {
      input_blobs.push_back("host/" + blob);
    }
    return input_blobs;
  } else {
    return logical_blobs_copied;
  }
}

template <typename Dtype>
std::vector<std::string> CopyTaskDag<Dtype>::BuildOutputTaskBlobs(
  const std::vector<std::string>& logical_blobs_copied) const {
  if (is_h2d_) {
    return logical_blobs_copied;
  } else {
    std::vector<std::string> output_blobs;
    for (auto& blob : logical_blobs_copied) {
      output_blobs.push_back("host/" + blob);
    }
    return output_blobs;
  }
}

template <typename Dtype>
void CopyTaskDag<Dtype>::AddProducedRegisterInfos() {
  CHECK_EQ(op_name_to_node_.size(), 1) << "Only a single copy layer";
  auto begin = op_name_to_node_.begin();
  auto op_name = begin->first;
  auto op_node = begin->second;
  auto op_meta = op_node->op();
  auto layer = op_meta->layer();

  EnvelopeFlag envelope_flag{ EnvelopeFlag::kOutEnvelope };
  DeviceType device_type = is_h2d_ ? DeviceType::kGPU : DeviceType::kCPUPinned;
  RegisterType register_type;
  if (is_forward_) {
    register_type = RegisterType::kDataType;
  } else {
    register_type = RegisterType::kDataDiffType;
  }

  auto& id_map = oneflow::TheOne<Dtype>::id_map();

  int32_t group_local_id = id_map->new_group_local_id(task_id_);
  int64_t group_id
    = id_map->group_id_from_task_id_and_group_local_id(task_id_, group_local_id);
  RegisterInfo register_info(register_type, device_type, group_id, false);

  std::vector<std::string> layer_vars;
  if (is_forward_) {
    layer_vars = layer->GetOutputVars();
  } else {
    layer_vars = layer->GetInputVars();
  }
  AddBlobsToProducedRegisterInfo(
    op_name, layer_vars, &register_info, envelope_flag, null_filter_);
  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(register_info);
}

template <typename Dtype>
void CopyTaskDag<Dtype>::AddConsumedRegisterInfosInPath() {
  auto actor_dag = dag_builder_.actor_dag();
  auto producers = GetImmediateProducerNamesInPath();
  CHECK_EQ(producers.size(), 1);
  auto producer = producers[0];
  auto producer_task_id = actor_dag->GetTaskID(producer);
  auto producer_task_dag = dag_builder_.GetTaskDag(producer_task_id);
  int64_t group_id = producer_task_dag->GetImmediateProducedGroupIdInPath(name_);
  // 1, Add group_id to register_info_manager_'s consumed register
  register_info_manager_.AddConsumedGroupId(group_id);

  // 2, Update the producer's |consumer_to_my_produced_group_id_|
  producer_task_dag->RegisterConsumer(task_id_, group_id);

  // 3, Collect all the blobs should be obtained in the producer's dst register
  // 4, Update which blob belongs to which register (group_id)
  // 5, Set register_blob_from_layer_blob, the correspondence between the
  //    layer_blob name in current task_dag to the task_blob name in the producer
  //    task_dag.
  CHECK_EQ(op_name_to_node_.size(), 1) << "Only a single copy layer";
  auto begin = op_name_to_node_.begin();
  auto op_name = begin->first;
  auto op_node = begin->second;
  auto op_meta = op_node->op();
  auto layer = op_meta->layer();
  std::vector<std::string> layer_vars;
  if (is_forward_) {
    layer_vars = layer->GetInputVars();
  } else {
    layer_vars = layer->GetOutputVars();
  }
  AddBlobsToConsumedRegisterInfo(
    op_name, layer_vars, producer_task_dag, group_id, null_filter_);
}

template <typename Dtype>
RegisterInfo CopyTaskDag<Dtype>::CompleteConsumedRegisterInfoCrossPath(
  RegisterType consumer_register_type, int64_t produced_group_id) {
  CHECK(path_type_ == PathType::kModelStorePath);
  CHECK(is_forward_);
  RegisterInfo register_info(consumer_register_type, DeviceType::kGPU, 0, false);

  CHECK_EQ(op_name_to_node_.size(), 1) << "Only a single copy layer";
  auto begin = op_name_to_node_.begin();
  auto op_name = begin->first;
  auto op_node = begin->second;
  auto op_meta = op_node->op();
  auto layer = op_meta->layer();

  std::vector<std::string> layer_vars;
  if (is_forward_) {
    layer_vars = layer->GetInputVars();
  } else {
    layer_vars = layer->GetOutputVars();
  }
  AddBlobsToConsumedRegisterInfoCrossPath(
    op_name,
    layer_vars,
    &register_info,
    produced_group_id,
    EnvelopeFlag::kOnEnvelope);

  return register_info;
}

template <typename Dtype>
RegisterInfo CopyTaskDag<Dtype>::ReplaceProducedRegisterInfoCrossPath(
  RegisterType my_register_type, int64_t other_group_id) {
  CHECK(path_type_ == PathType::kModelLoadPath)
    << "So far, only kModelLoadPath requires such routine";
  CHECK(is_forward_);

  // While in AddProducedRegisterInfo
  // (1) add blob to the produced register info (map to group_id)
  // (2) add the RegisterInfo into RegisterInfoManager
  // We need to revert the above process

  // Remove my RegisterInfo which will be replaced by |other_group_id|, so that
  // in a later stage, the system will not allocate resource to the replaced
  // RegisterInfo.
  int64_t my_group_id
    = register_info_manager_.GetProducedGroupIdForNonBoxingTask(my_register_type);
  register_info_manager_.RemoveProducedRegisterInfoForNonBoxingTask(
    my_register_type, my_group_id);

  // The blobs in the RegisterInfo which will be replaced are with
  // EnvelopeFlag::kOutEnvelope, however, we require them to be EnvelopeFlag::
  // kOnEnvelope. We just drop the old RegisterInfo (|my_group_id|) and create
  // a new one to return to the caller.

  CHECK_EQ(op_name_to_node_.size(), 1) << "Only a single copy layer";
  auto begin = op_name_to_node_.begin();
  auto op_name = begin->first;
  auto op_node = begin->second;
  auto op_meta = op_node->op();
  auto layer = op_meta->layer();

  EnvelopeFlag envelope_flag{ EnvelopeFlag::kOnEnvelope };
  DeviceType device_type = is_h2d_ ? DeviceType::kGPU : DeviceType::kCPUPinned;
  RegisterInfo register_info(my_register_type, device_type, 0, false);

  std::vector<std::string> layer_vars;
  layer_vars = layer->GetOutputVars();
  CHECK_EQ(layer_vars.size(), 1) << "";

  for (auto layer_var : layer_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(op_name, layer_var);
    auto task_blob = task_blob_from_layer_blob(layer_blob);
    auto register_blob = task_blob;
    register_info.AddEmptyBlob(task_blob, envelope_flag);
    // Remove the connection between |layer_blob| and |my_group_id|. The reverse
    // process of |AddProducedBlobToRegister|.
    blob_info_manager_.RemoveProducedBlobFromRegister(layer_blob, my_group_id);

    blob_info_manager_.AddConsumedBlobToRegister(layer_blob, register_blob,
      other_group_id);
  }
  return register_info;
}

INSTANTIATE_CLASS(CopyTaskDag);
}  // namespace oneflow
