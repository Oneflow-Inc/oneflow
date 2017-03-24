#include "dag/compute_task_dag.h"
#include <string>
#include <utility>
#include <vector>
#include "common/common.h"
#include "common/split_util.h"
#include "context/one.h"
#include "dag/node_meta.h"
#include "dag/logical_dag.h"
#include "dag/actor_dag.h"
#include "dag/dag_builder.h"
#include "layers/base_layer.h"
#include "layers/loader_layer.h"
#include "oneflow.pb.h"
#include "path/base_path.h"
#include "path/data_path.h"
#include "path/model_load_path.h"
#include "path/model_store_path.h"
#include "path/model_update_path.h"
#include "dag/blob_info_manager.h"
#include "layers/base_layer.h"

namespace oneflow {
template <typename Dtype>
ComputeTaskDag<Dtype>::ComputeTaskDag(const DagBuilder<Dtype>& dag_builder,
  TaskType type, int32_t task_id, PathType path_type,
  const std::string& actor_name, bool is_forward) : TaskDag(
  dag_builder, type, task_id, path_type, actor_name, is_forward) {
}
template <typename Dtype>
ComputeTaskDag<Dtype>::~ComputeTaskDag() {}

template <typename Dtype>
std::string ComputeTaskDag<Dtype>::build_task_blob_from_logical_blob(
  const std::string& logical_blob) const {
  return logical_blob;
}

template <typename Dtype>
std::string ComputeTaskDag<Dtype>::build_task_blob_from_layer_blob(
  const std::string& layer_blob) const {
  return layer_blob;
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::BuildForward() {
  // Clone the sub-DAG from LogicalDag
  BuildFromLayerSet();
  if (dag_builder_.has_BP()) {
    // Need to add CopyD2D only when this TaskDag has a backward correspondence
    AddCopyD2DLayer();
  }
  // Check whether we need to insert a SplitLayer at appropriate position
  AddSplitLayersIfNecessary();
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::BuildFromLayerSet() {
  auto actor_dag = dag_builder_.actor_dag();
  auto layer_names = actor_dag->GetLayerNamesFromActor(name_);
  if (type_ == TaskType::kDataTask) {
    CHECK_EQ(layer_names.size(), 1);
  }
  for (auto& layer_name : layer_names) {
    BuildFromLayer(layer_name);
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::BuildFromLayer(const std::string& layer_name) {
  // For kComputeTaskDag, the initial form of TaskDag is exactly the same as a
  // sub-graph of LogicalDag, such as the layer name, the task_blob_name (same
  // as the logical_blob_name).
  auto logical_dag = dag_builder_.logical_dag();
  auto layer_node = logical_dag->GetOpNode(layer_name);

  auto layer_type = layer_node->op()->type();
  auto layer_param_str = RectifyProtoStrForModelParallelism(
    layer_name, layer_type, layer_node->op()->param_str());
  // TaskDag and LogicalDag do not share the same layer object
  auto op_node = AddOpNode(layer_name, layer_type, layer_param_str);
  auto layer = op_node->op()->layer();

  std::vector<DNode*> input_nodes;
  auto input_vars = layer->GetInputVars();
  for (auto input_var : input_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
    auto logical_blob = logical_dag->DagBlobFromLayerBlob(layer_blob);
    auto task_blob = build_task_blob_from_logical_blob(logical_blob);

    bool is_input = false;
    DNode* data_node;
    auto data_node_it = data_name_to_node_.find(task_blob);
    if (data_node_it == data_name_to_node_.end()) {
      input_nodes.push_back(AddDataNode(task_blob));
      is_input = true;
    } else {
      input_nodes.push_back(data_node_it->second);
      is_input = false;
    }
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, is_input);
    UpdateBlobToConsumer(task_blob, layer_name, layer_blob);
  }

  std::vector<DNode*> output_nodes;
  auto output_vars = layer->GetOutputVars();
  for (auto output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    auto logical_blob = logical_dag->DagBlobFromLayerBlob(layer_blob);
    auto task_blob = build_task_blob_from_logical_blob(logical_blob);
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    output_nodes.push_back(AddDataNode(task_blob));
  }
  AddEdges(op_node, input_nodes, output_nodes);
}

template <typename Dtype>
std::string ComputeTaskDag<Dtype>::RectifyProtoStrForModelParallelism(
  const std::string& layer_name, const std::string& layer_type,
  const std::string& param_proto) const {
  auto logical_dag = dag_builder_.logical_dag();
  CHECK(logical_dag->HasOpNode(layer_name));
  PlacementInfo placement_info = logical_dag->GetPlacementInfo(layer_name);
  if (placement_info.parallel_policy() != kModelParallelOnMultipleDevices) {
    // For layer not with model-parallelism, no need to rectify the proto string
    return param_proto;
  }

  // Figure out the index of device's logical id in the all devices allocated
  // to this layer for model-parallelism
  auto& device_set = placement_info.device_set();
  int32_t parallel_size = placement_info.device_set().size();
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t thread_id = id_map->thread_id_from_task_id(task_id_);
  int32_t logical_id = id_map->logical_id_from_device_id(thread_id);
  // |device_set| keeps the logical_ids of all the devices allocated to this
  // layer, the ids are sorted, so we can use binary_search to find
  // an id's idx
  auto pos
    = std::lower_bound(device_set.begin(), device_set.end(), logical_id);
  CHECK(pos != device_set.end() && *pos == logical_id)
    << "The logical_id must be in the device_set";
  int32_t index = pos - device_set.begin();

  // NOTE(jiyuan): We assume only InnerProductLayer may be model-parallelism
  // Figure out the number of output neurons
  if (layer_type != "InnerProduct") {
    return param_proto;
  }
  InnerProductProto ip_proto;
  ParseProtoFromStringOrDie(param_proto, &ip_proto);
  CHECK(ip_proto.has_num_output());
  int32_t num_output = ip_proto.num_output();

  std::vector<int64_t> split_dims;
  GetDimOfEachSplit(num_output, parallel_size, &split_dims);
  ip_proto.set_num_output(split_dims[index]);

  return ip_proto.DebugString();
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::UpdateBlobToConsumer(
  const std::string& dag_blob_name,
  const std::string& consumer_layer_name,
  const std::string& layer_blob_name) {
  auto consumer_to_blob_it =
    task_blob_to_consumer_to_layer_blob_.find(dag_blob_name);
  if (consumer_to_blob_it == task_blob_to_consumer_to_layer_blob_.end()) {
    std::unordered_map<std::string, std::string> consumer_layer_to_layer_blob;
    consumer_layer_to_layer_blob.insert(
      { consumer_layer_name, layer_blob_name });
    task_blob_to_consumer_to_layer_blob_.insert({
      dag_blob_name, consumer_layer_to_layer_blob });
  } else {
    auto consumer_it = consumer_to_blob_it->second.find(consumer_layer_name);
    if (consumer_it == consumer_to_blob_it->second.end()) {
      consumer_to_blob_it->second.insert(
        { consumer_layer_name, layer_blob_name });
    } else {
      LOG(FATAL) << "Duplicate operations";
    }
  }
}

// Add a Copy2D2 layer to move the content to the input_task_blobs.
template <typename Dtype>
void ComputeTaskDag<Dtype>::AddCopyD2DLayer() {
  auto input_task_blobs = blob_info_manager_.input_task_blobs();
  if (input_task_blobs.size() == 0) return;  // e.g., kDataTask

  // Get the corresponding logical_blobs to input_task_blobs
  std::vector<std::string> logical_blobs;
  for (auto &input_task_blob : input_task_blobs) {
    auto logical_blob
      = blob_info_manager_.logical_blob_from_task_blob(input_task_blob);
    logical_blobs.push_back(logical_blob);
  }

  auto copy_input_blobs = BuildCopyInputBlobNames(input_task_blobs);
  // Keep the outputs of CopyD2D unchanged
  auto copy_output_blobs = input_task_blobs;

  std::string layer_type = "Copy";
  std::string layer_name = "copy_d2d_" + name_;
  std::string proto_str = BuildCopyProtoString(input_task_blobs.size());

  auto layer_node = AddOpNode(layer_name, layer_type, proto_str);
  auto layer = layer_node->op()->layer();

  std::vector<DataNode<BlobMeta>*> input_nodes;
  auto input_vars = layer->GetInputVars();
  CHECK(input_vars.size() == copy_input_blobs.size());
  int32_t idx = 0;
  for (auto& input_var : input_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
    auto task_blob = copy_input_blobs[idx];
    auto logical_blob = logical_blobs[idx];

    input_nodes.push_back(AddDataNode(task_blob));
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, true);
    ++idx;
  }
  std::vector<DataNode<BlobMeta>*> output_nodes;
  auto output_vars = layer->GetOutputVars();
  CHECK(output_vars.size() == copy_output_blobs.size());
  idx = 0;
  for (auto& output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    auto task_blob = copy_output_blobs[idx];
    auto logical_blob = logical_blobs[idx];
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    blob_info_manager_.EraseInputTaskBlob(task_blob);
    // The output_node should already exist
    output_nodes.push_back(GetDataNode(task_blob));
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
std::string ComputeTaskDag<Dtype>::BuildCopyProtoString(int32_t blob_num) const {
  CopyProto copy_proto;
  copy_proto.set_num(blob_num);
  CopyType copy_type = ForwardD2D;
  copy_proto.set_copy_type(copy_type);
  std::string copy_proto_str = copy_proto.DebugString();
  return copy_proto_str;
}

template <typename Dtype>
std::vector<std::string> ComputeTaskDag<Dtype>::BuildCopyInputBlobNames(
  const std::vector<std::string>& task_blobs) const {
  std::vector<std::string> input_blobs;
  for (auto& blob : task_blobs) {
    input_blobs.push_back("device/" + blob);
  }
  return input_blobs;
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::AddSplitLayersIfNecessary() {
  CollectBlobsNeedSplit();
  RemoveExistingEdges();
  InsertSplitLayers();
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::CollectBlobsNeedSplit() {
  auto logical_dag = dag_builder_.logical_dag();
  for (auto& data_node_pair : data_name_to_node_) {
    auto data_name = data_node_pair.first;
    auto data_node = data_node_pair.second;
    auto succeeding_layer_names = GetSucceedingOpNodeNamesOfDataNode(data_name);
    if (succeeding_layer_names.size() > 1) {
      task_blobs_need_split_.push_back(data_name);
      task_blob_to_consumers_.insert({ data_name, succeeding_layer_names});
      auto preceeding_layer_names
        = logical_dag->GetPreceedingOpNodeNamesOfDataNode(data_name);
      CHECK_EQ(preceeding_layer_names.size(), 1);
      task_blob_to_producer_.insert({ data_name, preceeding_layer_names[0] });
    }
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::RemoveExistingEdges() {
  for (auto task_blob_need_split : task_blobs_need_split_) {
    auto blob_node = GetDataNode(task_blob_need_split);
    CHECK_GT(task_blob_to_consumers_.count(task_blob_need_split), 0);
    auto consumers = task_blob_to_consumers_[task_blob_need_split];
    for (auto consumer : consumers) {
      auto consumer_node = GetOpNode(consumer);
      auto erase_num = consumer_node->RemoveParent(blob_node);
      CHECK(task_blob_to_consumer_to_layer_blob_.count(task_blob_need_split));
      auto consumer_to_layer_blob
        = task_blob_to_consumer_to_layer_blob_[task_blob_need_split];
      CHECK(consumer_to_layer_blob.count(consumer) > 0);
      auto layer_blob = consumer_to_layer_blob[consumer];
      blob_info_manager_.RemoveLayerAndTaskBlobPair(layer_blob);
      CHECK_EQ(erase_num, 1);
    }
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::InsertSplitLayers() {
  for (auto& task_blob_need_split : task_blobs_need_split_) {
    InsertSplitLayer(task_blob_need_split);
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::InsertSplitLayer(
  const std::string& task_blob_split) {
  CHECK_GT(task_blob_to_producer_.count(task_blob_split), 0);
  auto producer_name = task_blob_to_producer_[task_blob_split];
  CHECK_GT(task_blob_to_consumers_.count(task_blob_split), 0);
  auto consumer_names = task_blob_to_consumers_[task_blob_split];
  auto logical_blob
    = blob_info_manager_.logical_blob_from_task_blob(task_blob_split);

  // Construct split layer name
  std::string layer_type = "Split";
  auto layer_name = "split_" + producer_name;
  SplitProto split_proto;
  split_proto.set_in(task_blob_split);
  split_proto.set_out_num(consumer_names.size());
  for (int32_t i = 0; i < consumer_names.size(); ++i) {
    split_proto.add_out("out/" + std::to_string(i));
  }
  std::string split_proto_str = split_proto.DebugString();
  auto layer_node = AddOpNode(layer_name, layer_type, split_proto_str);
  auto layer = layer_node->op()->layer();

  // Connect the split layer node to the blob node that needs splitting
  auto blob_node = GetDataNode(task_blob_split);
  layer_node->AddParent(blob_node);

  // Handle with the input blobs of SplitLayer
  auto input_vars = layer->GetInputVars();
  CHECK_EQ(input_vars.size(), 1);
  auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_vars[0]);
  blob_info_manager_.RegisterBlob(
    layer_blob, task_blob_split, logical_blob, false);

  // Handle with the output blobs of SplitLayer
  auto output_vars = layer->GetOutputVars();
  CHECK_EQ(output_vars.size(), consumer_names.size());

  auto consumer_to_layer_blob_it
    = task_blob_to_consumer_to_layer_blob_.find(task_blob_split);
  CHECK(consumer_to_layer_blob_it !=
    task_blob_to_consumer_to_layer_blob_.end());
  auto consumer_to_layer_blob = consumer_to_layer_blob_it->second;
  CHECK_EQ(consumer_to_layer_blob.size(), output_vars.size());

  int32_t idx = 0;
  for (auto output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    auto task_blob = build_task_blob_from_layer_blob(layer_blob);
    auto data_node = AddDataNode(task_blob);
    data_node->AddParent(layer_node);
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);

    // Get the consumer layer node
    auto consumer_name = consumer_names[idx];
    auto consumer_node = GetOpNode(consumer_name);
    consumer_node->AddParent(data_node);

    // Get the input blob's layer_blob_name for consumer layer
    auto consumer_layer_blob_it = consumer_to_layer_blob.find(consumer_name);
    CHECK(consumer_layer_blob_it != consumer_to_layer_blob.end());
    auto consumer_blob = consumer_layer_blob_it->second;
    blob_info_manager_.RegisterBlob(
      consumer_blob, task_blob, logical_blob, false);

    ++idx;
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::BuildBackward() {
  // (1) The backward TaskDag share the same operators (i.e., layers) with the
  // forward TaskDag. Therefore, while creating OpNode, it does not create new
  // layers, instead, directly uses the layers in forward TaskDag.
  // (2) The backward TaskDag has its own blobs distinct from the forward
  // TaskDag.
  auto& actor_dag = dag_builder_.actor_dag();
  auto forward_task_name = actor_dag->GetForwardTaskName(name_);
  auto forward_task_id = actor_dag->GetTaskID(forward_task_name);
  auto backward_task_id = actor_dag->GetTaskID(name_);
  auto forward_task_dag = dag_builder_.GetTaskDag(forward_task_id);

  // 'Clone' the forward_task_dag to this backward TaskDag by reversely
  // traversing the forward_task_dag. The forward-backward TaskDags pair share
  // the operators (i.e., layers).
  DagReverseIterator<TaskDag<Dtype>, true> dag_iterator(*forward_task_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    // Skip the CopyD2D layer
    if (forward_task_dag->IsFirstOpNode(current_node)) continue;
    auto layer_node
      = dynamic_cast<const OpNode<LayerMeta<Dtype>>*>(current_node);
    auto layer_name = current_node->node_name();
    auto layer_meta = layer_node->op();
    // Directly re-use the layer created in |forward_task_dag|
    auto op_node = AddBackwardOpNode(layer_name, layer_meta);
    auto op_meta = op_node->op();  // Essentially is the same as |layer_meta|
    auto layer = op_meta->layer();

    std::vector<DNode*> input_nodes;
    auto output_diffs = layer->GetOutputDiffs();
    // TODO(jiyuan): loss layer does not have an out_diff, therefore, the loss
    // layer in backward TaskDag usually does not have an input blob.
    for (auto output_diff : output_diffs) {
      // Get its corresponding layer_blob name in forward-direction
      auto output_var = strings::get_data_blob_name(output_diff);
      auto layer_diff = strings::full_blob_name_in_layer(layer_name, output_diff);
      auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
      auto task_blob = forward_task_dag->task_blob_from_layer_blob(layer_blob);
      auto logical_blob
        = forward_task_dag->logical_blob_from_task_blob(task_blob);
      auto task_diff = strings::get_diff_blob_name(task_blob);
      auto data_node_it = data_name_to_node_.find(task_diff);
      bool is_input = false;
      DNode* data_node;
      if (data_node_it == data_name_to_node_.end()) {
        data_node = AddDataNode(task_diff);
        is_input = true;
      } else {
        data_node = data_node_it->second;
        is_input = false;
      }
      blob_info_manager_.RegisterBlob(layer_diff, task_diff, logical_blob, is_input);
      input_nodes.push_back(data_node);
    }

    std::vector<DNode*> output_nodes;
    auto input_diffs = layer->GetInputDiffs();
    // NOTE(jiyuan): for loss layer, the label data does not have an input_diff
    for (auto input_diff : input_diffs) {
      // Get its corresponding layer_blob name in forward-direction
      auto input_var = strings::get_data_blob_name(input_diff);
      auto layer_diff = strings::full_blob_name_in_layer(layer_name, input_diff);
      auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
      auto task_blob = forward_task_dag->task_blob_from_layer_blob(layer_blob);
      auto logical_blob = forward_task_dag->logical_blob_from_task_blob(task_blob);
      auto task_diff = strings::get_diff_blob_name(task_blob);
      auto data_node = AddDataNode(task_diff);
      output_nodes.push_back(data_node);
      blob_info_manager_.RegisterBlob(layer_diff, task_diff, logical_blob, false);
      // TODO(jiyuan): if the SplitLayer is for label-path, do not include them
      // in the backward TaskDag.
    }

    auto output_vars = layer->GetOutputVars();
    for (auto& output_var : output_vars) {
      auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
      auto task_blob = forward_task_dag->task_blob_from_layer_blob(layer_blob);
      auto logical_blob
        = forward_task_dag->logical_blob_from_task_blob(task_blob);
      blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    }

    auto input_vars = layer->GetInputVars();
    for (auto& input_var : input_vars) {
      auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
      auto task_blob = forward_task_dag->task_blob_from_layer_blob(layer_blob);
      auto logical_blob
        = forward_task_dag->logical_blob_from_task_blob(task_blob);
      blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    }
    AddEdges(op_node, input_nodes, output_nodes);
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::RegisterNonInputOutputBlobs() {
  auto logical_dag = dag_builder_.logical_dag();
  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    // kOther variables are needed by both forward and backward task
    auto other_vars = layer->GetOtherVars();
    for (auto& other_var : other_vars) {
      auto layer_blob = strings::full_blob_name_in_layer(op_name, other_var);
      auto task_blob = layer_blob;
      auto logical_blob = logical_dag->DagBlobFromLayerBlob(layer_blob);
      blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    }
    // kModel and kTemp variables are needed by both forward and backward task
    auto model_vars = layer->GetModelVars();
    for (auto& model_var : model_vars) {
      auto layer_blob = strings::full_blob_name_in_layer(op_name, model_var);
      auto task_blob = layer_blob;
      auto logical_blob = logical_dag->DagBlobFromLayerBlob(layer_blob);
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blob, false);
    }
    auto temp_vars = layer->GetTempVars();
    for (auto& temp_var : temp_vars) {
      auto layer_blob = strings::full_blob_name_in_layer(op_name, temp_var);
      auto task_blob = layer_blob;
      auto logical_blob = logical_dag->DagBlobFromLayerBlob(layer_blob);
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blob, false);
    }

    if (!is_forward_) {
      // kModelDiff is needed only by backward task
      auto model_diffs = layer->GetModelDiffs();
      for (auto& model_diff : model_diffs) {
        auto model_var = strings::get_data_blob_name(model_diff);
        auto layer_diff = strings::full_blob_name_in_layer(op_name, model_diff);
        auto layer_blob = strings::full_blob_name_in_layer(op_name, model_var);
        auto task_diff = layer_diff;
        auto logical_blob = logical_dag->DagBlobFromLayerBlob(layer_blob);
        blob_info_manager_.RegisterBlob(
          layer_diff, task_diff, logical_blob, false);
      }
    }
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::AddProducedRegisterInfos() {
  // The rule of adding the produced (owned) RegisterInfo is:
  // (1) For forward direction, we need to collect the InputVars, OutputVars,
  // OtherVars into a kDataType RegisterInfo.
  // (2) For backward direction, we need to collect the InputDiffs, OutputDiffs,
  // OtherVars into a kDataDiffType RegisterInfo, ModelDiffs into kModelDiffType
  // RegisterInfo.

  if (is_forward_) {
    ForwardAddProducedRegisterInfos();
  } else {
    BackwardAddProducedRegisterInfos();
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ForwardAddProducedRegisterInfos() {
  switch (path_type_) {
  case PathType::kDataPath:
    DataPathForwardAddProducedRegisterInfos();
    break;
  case PathType::kModelUpdatePath:
    ModelUpdatePathForwardAddProducedRegisterInfos();
    break;
  case PathType::kModelLoadPath:
    ModelLoadPathForwardAddProducedRegisterInfos();
    break;
  case PathType::kModelStorePath:
    ModelStorePathForwardAddProducedRegisterInfos();
    break;
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::DataPathForwardAddProducedRegisterInfos() {
  // 1) If it is a data loader layer, the produced ReisterInfo needs to cover
  // the output blobs.
  // 2) If it is a general compute layer, the produced RegisterInfo needs to
  // cover the input and output blobs, unless the input blob of 'First' layer
  // (which is the output blob of a CopyH2D layer).
  // No matter which case it is, the following implementation can handle with it.

  // NOTE(jiyuan): may change to kInEnvelope if this RegisterInfo will be
  // consumed by other paths.
  EnvelopeFlag envelope_flag{ EnvelopeFlag::kOutEnvelope };
  RegisterType register_type{ RegisterType::kDataType };
  DeviceType device_type{ DeviceType::kGPU };

  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t group_local_id = id_map->new_group_local_id(task_id_);
  int64_t group_id
    = id_map->group_id_from_task_id_and_group_local_id(task_id_, group_local_id);

  RegisterInfo data_register_info(register_type, device_type, group_id, false);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    if (!IsFirstOpNode(op_node)) {
      // No matter whether the first layer is a CopyD2D type, its input blob
      // does not need to be added to the |data_register_info|, since the blob
      // is an output blob of a CopyH2D layer in another task.
      AddBlobsToProducedRegisterInfo(op_name, layer->GetInputVars(),
        &data_register_info, envelope_flag, null_filter_);
    }

    AddBlobsToProducedRegisterInfo(op_name, layer->GetOutputVars(),
      &data_register_info, envelope_flag, null_filter_);

    AddBlobsToProducedRegisterInfo(op_name, layer->GetOtherVars(),
      &data_register_info, envelope_flag, null_filter_);
  }
  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(
    data_register_info);
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ModelUpdatePathForwardAddProducedRegisterInfos() {
  if (is_placeholder_) return;
  EnvelopeFlag envelope_flag{ EnvelopeFlag::kOnEnvelope };
  RegisterType register_type{ RegisterType::kDataType };
  DeviceType device_type{ DeviceType::kGPU };

  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t group_local_id = id_map->new_group_local_id(task_id_);
  int64_t group_id
    = id_map->group_id_from_task_id_and_group_local_id(task_id_, group_local_id);
  RegisterInfo data_register_info(register_type, device_type, group_id, false);

  // If it is not a placeholder task (i.e., model_update_layer or
  // null_update_layer), so far, the produced RegisterInfo only knows the
  // envelope blob's name. At a later time, we need to extract the normal blobs'
  // names from the consumer task in DataPath and fill in the normal blobs's
  // info of this produced RegisterInfo.
  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    AddBlobsToProducedRegisterInfo(op_name, layer->GetOutputVars(),
      &data_register_info, envelope_flag, null_filter_);
  }
  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(
    data_register_info);
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ModelLoadPathForwardAddProducedRegisterInfos() {
  if (is_placeholder_) return;
  // Only can originate from the LoadPartialModelLayer
  CHECK(type_ == TaskType::kDataTask);
  // If it is the loader layer, produced RegisterInfo needs to cover the
  // output blob.

  EnvelopeFlag envelope_flag{ EnvelopeFlag::kOutEnvelope };
  RegisterType register_type{ RegisterType::kDataType };
  DeviceType device_type{ DeviceType::kCPU };

  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t group_local_id = id_map->new_group_local_id(task_id_);
  int64_t group_id
    = id_map->group_id_from_task_id_and_group_local_id(task_id_, group_local_id);
  RegisterInfo data_register_info(register_type, device_type, group_id, false);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    AddBlobsToProducedRegisterInfo(op_name, layer->GetOutputVars(),
      &data_register_info, envelope_flag, null_filter_);
  }
  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(
    data_register_info);
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ModelStorePathForwardAddProducedRegisterInfos() {
  if (is_placeholder_) return;
  // Only can originates from the StoreLayer
  CHECK(type_ == TaskType::kDataTask);
  // If it is the store layer, no produced RegisterInfo
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::BackwardAddProducedRegisterInfos() {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t data_diff_local_id = id_map->new_group_local_id(task_id_);
  int64_t data_diff_group_id = id_map->group_id_from_task_id_and_group_local_id(
    task_id_, data_diff_local_id);
  int32_t model_diff_local_id = id_map->new_group_local_id(task_id_);
  int64_t model_diff_group_id = id_map->group_id_from_task_id_and_group_local_id(
    task_id_, model_diff_local_id);

  RegisterType data_diff_register_type{ RegisterType::kDataDiffType };
  RegisterType model_diff_register_type{ RegisterType::kModelDiffType };
  DeviceType device_type{ DeviceType::kGPU };
  EnvelopeFlag data_diff_envelope_flag{ EnvelopeFlag::kOutEnvelope };
  EnvelopeFlag model_diff_envelope_flag{ EnvelopeFlag::kInEnvelope };


  RegisterInfo data_diff_register_info(data_diff_register_type, device_type,
    data_diff_group_id, false);
  RegisterInfo model_diff_register_info(model_diff_register_type, device_type,
    model_diff_group_id, false);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    // For kDataDiffType
    if (!IsFirstOpNode(op_node)) {
      AddBlobsToProducedRegisterInfo(op_name, layer->GetOutputDiffs(),
        &data_diff_register_info, data_diff_envelope_flag, null_filter_);
    }

    AddBlobsToProducedRegisterInfo(op_name, layer->GetInputDiffs(),
      &data_diff_register_info, data_diff_envelope_flag, null_filter_);

    //AddBlobsToProducedRegisterInfo(op_name, layer->GetOtherVars(),
    //  &data_diff_register_info, data_diff_envelope_flag, null_filter_);

    // For kModelDiffType
    AddBlobsToProducedRegisterInfo(op_name, layer->GetModelDiffs(),
      &model_diff_register_info, model_diff_envelope_flag, null_filter_);

    AddBlobsToProducedRegisterInfo(op_name, layer->GetTempVars(),
      &model_diff_register_info, model_diff_envelope_flag, null_filter_);
  }

  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(
    data_diff_register_info);
  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(
    model_diff_register_info);
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::AddConsumedRegisterInfosInPath() {
  // DataTask doesn't has consumed register info
  if (type_ == TaskType::kDataTask) {
    return;
  }

  if (is_forward_) {
    ForwardAddConsumedRegisterInfoInPath();
  } else {
    BackwardAddConsumedRegisterInfoInPath();
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ForwardAddConsumedRegisterInfoInPath() {
  // The input of first layer is from previous task. Note that there may exist
  // multiple 'first' layer. For forward task, the consumed input is input_vars.
  auto actor_dag = dag_builder_.actor_dag();
  auto producers = GetImmediateProducerNamesInPath();
  // It is possible that a ComputeTaskDag does not have a preceeding kCopyH2D
  // task. For example, the Placeholder layer in ModelUpdatePath does not have
  // producers in the path.
  if (producers.size() == 0) return;
  // CHECK_EQ(producers.size(), 1);
  auto producer = producers[0];
  auto producer_task_id = actor_dag->GetTaskID(producer);
  auto producer_task_dag = dag_builder_.GetTaskDag(producer_task_id);
  auto group_id = producer_task_dag->GetImmediateProducedGroupIdInPath(name_);

  register_info_manager_.AddConsumedGroupId(group_id);
  producer_task_dag->RegisterConsumer(task_id_, group_id);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    if (IsFirstOpNode(op_node)) {
      AddBlobsToConsumedRegisterInfo(op_name, layer->GetInputVars(),
        producer_task_dag, group_id, null_filter_);
    }
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::BackwardAddConsumedRegisterInfoInPath() {
  // Backward task depends on
  // 1) out_diffs (producer task's input_diffs <-> consumer's output_diffs,
  // in data_diff_register). Handled with ImmediateProducedGroupIdInPath
  // 2) forward_task's input_vars, output_vars, model_vars. (data_register,
  // model_register). Not an Immediate producer.
  // 3) model register. cross-path dependency, will be handled with elsewhere:
  // i.e., in |CompleteConsumedRegisterInfoCrossPath|.

  // Tackle with case 1)
  auto actor_dag = dag_builder_.actor_dag();
  auto producers = GetImmediateProducerNamesInPath();
  CHECK_EQ(producers.size(), 1);
  auto producer = producers[0];
  auto producer_task_id = actor_dag->GetTaskID(producer);
  auto producer_task_dag = dag_builder_.GetTaskDag(producer_task_id);
  auto group_id = producer_task_dag->GetImmediateProducedGroupIdInPath(name_);

  register_info_manager_.AddConsumedGroupId(group_id);
  producer_task_dag->RegisterConsumer(task_id_, group_id);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    if (IsFirstOpNode(op_node)) {
      AddBlobsToConsumedRegisterInfo(op_name, layer->GetOutputDiffs(),
        producer_task_dag, group_id, null_filter_);
    }
  }

  // Tackle with case 2)
  auto forward_task_name = actor_dag->GetForwardTaskName(name_);
  auto forward_task_id = actor_dag->GetTaskID(forward_task_name);
  auto forward_task_dag = dag_builder_.GetTaskDag(forward_task_id);
  std::shared_ptr<ComputeTaskDag<Dtype>> forward_compute_task_dag
    = std::dynamic_pointer_cast<ComputeTaskDag<Dtype>>(forward_task_dag);
  CHECK(forward_compute_task_dag);
  auto data_group_id
    = forward_compute_task_dag->GetProducedGroupIdByRegisterType(
    RegisterType::kDataType);

  std::unordered_set<int64_t> group_ids;
  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    std::vector<std::string> data_vars;
    auto input_vars = layer->GetInputVars();
    auto output_vars = layer->GetOutputVars();
    auto other_vars = layer->GetOtherVars();
    data_vars.insert(data_vars.begin(), input_vars.begin(), input_vars.end());
    data_vars.insert(data_vars.begin(), output_vars.begin(), output_vars.end());
    data_vars.insert(data_vars.begin(), other_vars.begin(), other_vars.end());

    if (data_vars.size() > 0) {
      AddBlobsToConsumedRegisterInfo(op_name, data_vars, forward_task_dag,
        data_group_id, null_filter_);
      group_ids.insert(data_group_id);
    }
  }

  for (auto group_id : group_ids) {
    register_info_manager_.AddConsumedGroupId(group_id);
    forward_task_dag->RegisterConsumer(task_id_, group_id);
  }
}

template <typename Dtype>
RegisterInfo ComputeTaskDag<Dtype>::CompleteConsumedRegisterInfoCrossPath(
  RegisterType consumed_register_type, int64_t produced_group_id) {
  // kDataTask will not come here.
  CHECK(type_ == TaskType::kComputeTask);

  // So far, we assume only the compute TaskDag in kDataPath will be able to play
  // the role of a non-placeholder consumer of cross-path dependency.
  CHECK(path_type_ == PathType::kDataPath);

  // The consumer requires the model register.
  CHECK(consumed_register_type == RegisterType::kModelType);

  // Init the RegisterInfo which is produced by ModelUpdatePath but is consumed
  // by this TaskDag in DataPath.
  RegisterInfo model_register_info(
    RegisterType::kModelType, DeviceType::kGPU, 0, false);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    AddBlobsToConsumedRegisterInfoCrossPath(
      op_name,
      layer->GetModelVars(),
      &model_register_info,
      produced_group_id,
      EnvelopeFlag::kInEnvelope);

    AddBlobsToConsumedRegisterInfoCrossPath(
      op_name,
      layer->GetTempVars(),
      &model_register_info,
      produced_group_id,
      EnvelopeFlag::kInEnvelope);
  }

  // The backward compute TaskDag also needs to consume the model register.
  // The forward and backward TaskDags share the same RegisterInfo.
  if (is_forward_) {
    auto actor_dag = dag_builder_.actor_dag();
    auto backward_name = actor_dag->GetBackwardTaskName(this->name_);
    auto backward_task_dag = dag_builder_.GetTaskDagByName(backward_name);
    int32_t backward_task_id = backward_task_dag->task_id();
    // NOTE(jiyuan): although current TaskDag (this) is not the producer of
    // the RegisterInfo with |produced_group_id|, in our design, we let the
    // current TaskDag (i.e., the forward TaskDag) hand over the
    // |produced_group_id| to the backward TaskDag.
    this->RegisterConsumer(backward_task_id, produced_group_id);
    backward_task_dag->CompleteConsumedRegisterInfoCrossPath(
      consumed_register_type, produced_group_id);
  }

  return model_register_info;
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ForwardSetup() {
  LOG(INFO) << "Setup: " << name_;
  if (path_type_ == PathType::kDataPath && type_ == TaskType::kDataTask) {
    ForwardSetupPrepareDataTask();
    TaskDag<Dtype>::ForwardSetupInternal();
  } else {
    TaskDag<Dtype>::ForwardSetup();
  }
}

template <typename Dtype>
void ComputeTaskDag<Dtype>::ForwardSetupPrepareDataTask() {
  // We need to figure out the batch size.
  // Firstly, get the number of devices each data provider serves.
  // We have several ways to get the number: (1) traverse the
  // ActorDag; (2) Traverse the StageDag and figure out the number of pipes in
  // the first stage who processes 'data' blobs; (3) use the
  // max_data_parallel_num in StrategyDescriptor and the machine_num in
  // ResourceDescriptor. The third approach is not that general but is the
  // simplest, we use (3) currently.
  auto& strategy_descriptor =
    oneflow::TheOne<Dtype>::config_parser()->strategy_descriptor();
  int32_t device_num_per_data_provider
    = strategy_descriptor->device_num_per_data_provider();

  // Secondly, get the batch size for each device immediately following the
  // data provider. We get this value from StrategyDescriptor
  int32_t piece_size_each_device
    = strategy_descriptor->piece_size_each_device();

  // Thirdly, set the batch size for LoaderLayer
  int32_t piece_size_each_data_provider
    = piece_size_each_device * device_num_per_data_provider;

  auto loader_layers = GetFirstOpNames();
  CHECK_EQ(loader_layers.size(), 1);
  auto loader_node = GetOpNode(loader_layers[0]);
  auto &loader_meta = loader_node->op();
  auto& layer = loader_meta->mutable_layer();
  auto loader_layer = std::dynamic_pointer_cast<LoaderLayer<Dtype>>(layer);
  CHECK_NOTNULL(loader_layer.get());
  loader_layer->SetPieceSize(piece_size_each_data_provider);
}

template <typename Dtype>
std::vector<std::string>
ComputeTaskDag<Dtype>::GetInputLogicalBlobs() const {
  CHECK(type_ == TaskType::kComputeTask);
  auto actor_dag = dag_builder_.actor_dag();
  auto segment_dag = dag_builder_.segment_dag();
  auto segment_name = actor_dag->GetSegmentNameFromActor(name_);
  return segment_dag->GetInputBlobs(segment_name);
}

template <typename Dtype>
std::vector<std::string>
ComputeTaskDag<Dtype>::GetOutputLogicalBlobs() const {
  CHECK(type_ == TaskType::kComputeTask);
  auto actor_dag = dag_builder_.actor_dag();
  auto segment_dag = dag_builder_.segment_dag();
  auto segment_name = actor_dag->GetSegmentNameFromActor(name_);
  return segment_dag->GetOutputBlobs(segment_name);
}

INSTANTIATE_CLASS(ComputeTaskDag);
}  // namespace oneflow
