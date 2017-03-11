#include <utility>
#include <string>
#include <vector>
#include "common/common.h"
#include "context/one.h"
#include "dag/node_meta.h"
#include "dag/segment_dag.h"
#include "dag/actor_dag.h"
#include "dag/dag_builder.h"
#include "dag/boxing_task_dag.h"
#include "layers/base_layer.h"
#include "layers/layer_factory.h"
#include "task/job_manager.h"

namespace caffe {
template <typename Dtype>
BoxingTaskDag<Dtype>::BoxingTaskDag(const DagBuilder<Dtype>& dag_builder,
  TaskType type, int32_t task_id, PathType path_type,
  const std::string& actor_name, bool is_forward) : TaskDag(
  dag_builder, type, task_id, path_type, actor_name, is_forward) {}

template <typename Dtype>
BoxingTaskDag<Dtype>::~BoxingTaskDag() {}

// 1) Assumption
// For a particular segment-segment pair, we add a boxing layer for each
// blob. According to the assumption on SplitLayerWithinSegment and
// ConcatLayerWithinSegment, every blob will be consumed by exactly one
// segment, which means there are no overlapping blobs among all the envelopes
// out from a same segment, and there are no overlapping blobs among all the
// envelopes to a same segment.

// 2) Requirement to memory allocation
// While allocating memory for boxing task, ensure the memory of blobs in the 
// same envelope is contiguous.

// 3) RegisterInfo, ensure to include the middle_blob for backward boxing.

// 4) A same blob name in LogicalDag may have several corresponding blob name 
// in TaskDag. Note that, these blobs corresponding to the same blob name in 
// LogicalDag may not have the same shape. In other types of TaskDag, the blobs
// correspond to the same blob name in LogicalDag always have the same shape. 
// This leads to a special implementation of GetBlobShape()

template <typename Dtype>
void BoxingTaskDag<Dtype>::BuildForward() {
  auto actor_dag = dag_builder_.actor_dag();
  auto segment_dag = dag_builder_.segment_dag();
  auto boxing_pipe_name = actor_dag->GetPipeNameFromActor(name_);
  auto actor_boxing_info = actor_dag->GetForwardBoxingInfo(name_);
  auto segment_pairs = actor_boxing_info.GetSegmentPairs();

  // The boxing layers are added segment-segment-pair-wise. The boxing layers 
  // for the same segment-segment pair have the same properties (i.e., the same
  // number of inputs & outputs).
  for (auto& segment_pair : segment_pairs) {
    BoxingInfoElement boxing_info_elem
      = actor_boxing_info.GetBoxingInfoElement(segment_pair);
    auto envelope_names = segment_dag->FindDataNodesInBetween(
     segment_pair.first, segment_pair.second);
    CHECK_EQ(envelope_names.size(), 1);
    auto logical_blobs 
      = segment_dag->GetDataNode(envelope_names[0])->data()->blob_names();
    for (auto logical_blob : logical_blobs) {
      AddLayerForLogicalBlob(
        boxing_pipe_name, segment_pair, boxing_info_elem, logical_blob);
    }
  }
}

template <typename Dtype>
void BoxingTaskDag<Dtype>::AddLayerForLogicalBlob(
  const std::string& boxing_pipe_name,
  const SegmentSegmentPair& segment_pair,
  const BoxingInfoElement& boxing_info_elem,
  const std::string& logical_blob) {
  const std::string forward_in_boxing_prefix = "forward_in_boxing";
  bool is_in_boxing = strings::StartsWith(name_, forward_in_boxing_prefix);

  std::string layer_type = "Boxing";
  std::string layer_name = build_layer_name(boxing_pipe_name, logical_blob);

  BoxingProto boxing_proto;
  SetBoxingProtoValue(
    is_in_boxing, segment_pair, logical_blob, boxing_info_elem,
    &boxing_proto);
  std::string proto_str = boxing_proto.DebugString();

  auto layer_node = AddOpNode(layer_name, layer_type, proto_str);
  auto layer = layer_node->op()->layer();

  std::vector<DataNode<BlobMeta>*> input_nodes;
  std::vector<DataNode<BlobMeta>*> output_nodes;

  auto input_vars = layer->GetInputVars();
  CHECK(input_vars.size() == boxing_info_elem.in_num());
  for (auto input_var : input_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
    auto task_blob = strings::full_blob_name_in_dag(logical_blob, input_var);
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, true);
    input_nodes.push_back(AddDataNode(task_blob));
  }

  auto output_vars = layer->GetOutputVars();
  CHECK(output_vars.size() == boxing_info_elem.out_num());
  for (auto output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    auto task_blob = strings::full_blob_name_in_dag(logical_blob, output_var);
    blob_info_manager_.RegisterBlob(layer_blob, task_blob, logical_blob, false);
    output_nodes.push_back(AddDataNode(task_blob));
  }
  for (auto& input_node : input_nodes) {
    layer_node->AddParent(input_node);
  }
  for (auto& output_node : output_nodes) {
    output_node->AddParent(layer_node);
  }
}

template <typename Dtype>
void BoxingTaskDag<Dtype>::SetBoxingProtoValue(bool is_in_boxing,
  const SegmentSegmentPair& segment_pair,
  const std::string& blob_name,
  const BoxingInfoElement& boxing_info_elem,
  BoxingProto *boxing_proto) {
  CHECK_GE(boxing_info_elem.in_num(), 1);
  boxing_proto->set_in_num(boxing_info_elem.in_num());
  CHECK_GE(boxing_info_elem.out_num(), 1);
  boxing_proto->set_out_num(boxing_info_elem.out_num());

  auto first_segment = segment_pair.first;
  auto second_segment = segment_pair.second;
  auto segment_dag = dag_builder_.segment_dag();
  auto first_segment_node = segment_dag->GetOpNode(first_segment);
  auto first_parallel_policy
    = first_segment_node->op()->placement_info().parallel_policy();
  auto second_segment_node = segment_dag->GetOpNode(second_segment);
  auto second_parallel_policy
    = second_segment_node->op()->placement_info().parallel_policy();
  // FIXME(jiyuan): set correct value for in_model_update_path
  bool in_model_update_path = false;
  if (in_model_update_path) {
    // For boxing actor in model update path to sync model between data-parallel
    // nodes
    // No backward
    // Both in_boxing and out_boxing share the same logic
    boxing_proto->set_in_axis(3);
    boxing_proto->set_in_op(ADD);
    boxing_proto->set_backward_in_op(COPY);
    
    boxing_proto->set_out_axis(3);
    boxing_proto->set_out_op(SPLIT);
    boxing_proto->set_backward_out_op(CONCAT);
  } else if (first_parallel_policy == kDataParallelOnMultipleMachines
    && second_parallel_policy == kDataParallelOnMultipleDevices) {
    // For data loader task to compute task
    if (!segment_dag->IsFirstOpNode(first_segment)) {
      LOG(FATAL) << "Boxing node can not occur between two data-parallel layer";
    }
    // Only be valid for data-provider to compute
    if (is_in_boxing) {
      boxing_proto->set_in_axis(0);
      boxing_proto->set_in_op(CONCAT);
      boxing_proto->set_backward_in_op(SPLIT);

      boxing_proto->set_out_axis(0);
      boxing_proto->set_out_op(SPLIT);
      boxing_proto->set_backward_out_op(CONCAT);
    } else {
      boxing_proto->set_in_axis(0);
      boxing_proto->set_in_op(CONCAT);
      boxing_proto->set_backward_in_op(SPLIT);

      boxing_proto->set_out_axis(0);
      boxing_proto->set_out_op(COPY);
      boxing_proto->set_backward_out_op(ADD);
    }
  } else if (first_parallel_policy == kDataParallelOnMultipleDevices
    && second_parallel_policy == kModelParallelOnMultipleDevices) {
    // Not matter whether it is in_boxing or out_boxing, having the same
    // property
    boxing_proto->set_in_axis(0);
    boxing_proto->set_in_op(CONCAT);
    boxing_proto->set_backward_in_op(SPLIT);

    boxing_proto->set_out_axis(0);
    boxing_proto->set_out_op(COPY);
    boxing_proto->set_backward_out_op(ADD);
  } else if (first_parallel_policy == kModelParallelOnMultipleDevices
    && second_parallel_policy == kModelParallelOnMultipleDevices) {
    boxing_proto->set_in_axis(1);
    boxing_proto->set_in_op(CONCAT);
    boxing_proto->set_backward_in_op(SPLIT);

    boxing_proto->set_out_axis(0);
    boxing_proto->set_out_op(COPY);
    boxing_proto->set_backward_out_op(ADD);
  } else if (first_parallel_policy == kModelParallelOnMultipleDevices
    && second_parallel_policy == kDataParallelOnMultipleDevices) {
    // Model-parallel->Data parallel
    // For loss compute task
    boxing_proto->set_in_axis(1);
    boxing_proto->set_in_op(CONCAT);
    boxing_proto->set_backward_in_op(SPLIT);

    boxing_proto->set_out_axis(0);
    boxing_proto->set_out_op(SPLIT);
    boxing_proto->set_backward_out_op(CONCAT);
  } else {
    LOG(FATAL) << "This pair of parallel polices is not supported in "
      << "BoxingTaskDag : from |" << first_parallel_policy << "| to |"
      << second_parallel_policy << "|.";
  }
}

template <typename Dtype>
void BoxingTaskDag<Dtype>::UpdateRegisterInfo(
  const std::string& layer_blob,
  const std::string& task_blob,
  const std::string& second_segment,int32_t idx, int32_t output_num,
  std::unordered_map<std::string, std::vector<RegisterInfo>>* register_infos) {

  EnvelopeFlag envelope_flag{ EnvelopeFlag::kOutEnvelope };
  RegisterType register_type{ RegisterType::kDataType };
  DeviceType device_type{ DeviceType::kCPU };

  auto& id_map = caffe::TheOne<Dtype>::id_map();
  auto register_infos_it
    = register_infos->find(second_segment);
  if (register_infos_it == register_infos->end()) {
    // No register_infos exist for |segment_pair|
    std::vector<RegisterInfo> register_info_vec;
    register_info_vec.resize(output_num);
    for (int32_t i = 0; i < output_num; ++i) {
      register_info_vec[i].set_register_type(register_type);
      register_info_vec[i].set_device_type(device_type);
      register_info_vec[i].set_network(false);
      int32_t group_local_id = id_map->new_group_local_id(task_id_);
      int64_t group_id = id_map->group_id_from_task_id_and_group_local_id(
        task_id_, group_local_id);
      register_info_vec[i].set_group_id(group_id);
    }
    register_info_vec[idx].AddEmptyBlob(task_blob, envelope_flag);
    blob_info_manager_.AddProducedBlobToRegister(
      layer_blob, register_info_vec[idx].group_id());
    register_infos->insert({ second_segment, register_info_vec });
  } else {
    // register_infos already exist for |segment_pair|
    auto& register_info_vec = register_infos_it->second;
    CHECK(register_info_vec.size() == output_num);
    register_info_vec[idx].AddEmptyBlob(task_blob, envelope_flag);
    blob_info_manager_.AddProducedBlobToRegister(
      layer_blob, register_info_vec[idx].group_id());
  }
}

template <typename Dtype>
void BoxingTaskDag<Dtype>::RegisterNonInputOutputBlobs() {
  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    auto other_vars = layer->GetOtherVars();
    int32_t other_num = other_vars.size();
    CHECK(other_num == 1);
    for (auto& other_var : other_vars) {
      // Let |layer_blob|, |task_blob|, |logical_blob| have the same value
      auto layer_blob = strings::full_blob_name_in_layer(op_name, other_var);
      auto task_blob = layer_blob;
      auto logical_blob = layer_blob;
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blob, false);
    }
  }
}

template <typename Dtype>
void BoxingTaskDag<Dtype>::AddProducedRegisterInfos() {
  // Group the produced register_infos according to the consumer_segment.
  std::unordered_map<std::string, std::vector<RegisterInfo>>
    consumer_segment_to_produced_register_infos;

  auto segment_dag = dag_builder_.segment_dag();
  auto actor_dag = dag_builder_.actor_dag();
  BoxingInfo actor_boxing_info(false);
  if (is_forward_) {
    actor_boxing_info = actor_dag->GetForwardBoxingInfo(name_);
  } else {
    auto forward_name = actor_dag->GetForwardTaskName(name_);
    actor_boxing_info = actor_dag->GetForwardBoxingInfo(forward_name);
  }
  auto pipe_name = actor_dag->GetPipeNameFromActor(name_);

  // While building BoxingTaskDag, we create a layer for each blob in SegmentDag.
  // This layer will generate several output vars for the blob of SegmentDag. we
  // should group the output vars into the same RegisterInfo if they are from
  // the same SegmentSegmentPair and if they have the same indices in the 
  // ActorBoxingMeta.
  // For example, there are two blobs named with "blob_name1" and "blob_name2",
  // who are in the same SegmentSegmentPair. Assume there are three output vars
  // after the boxing layer for "blob_name1" and "blob_name2" respectively. 
  // Suppose the pipe_name is "pipe", then the layer names should be: "pipe_blob_name1"
  // and "pipe_blob_name2" respectively. The first layer "pipe_blob_name1" has 
  // three output vars: "pipe_blob_name1/0/out", "pipe_blob_name1/1/out", 
  // "pipe_blob_name1/2/out". The second layer "pipe_blob_name2" has three output
  // vars too: "pipe_blob_name2/0/out", "pipe_blob_name2/1/out", "pipe_blob_name2/2/out".
  // We should create three RegisterInfo objects which include:
  // (1) "pipe_blob_name1/0/out" and "pipe_blob_name2/0/out"
  // (2) "pipe_blob_name1/1/out" and "pipe_blob_name2/1/out"
  // (3) "pipe_blob_name1/2/out" and "pipe_blob_name2/2/out"

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();
    // Get the blob_name in SegmentDag from layer's name |op_name|
    auto blob_name = strings::RemovePrefix(op_name, pipe_name);  // "_blob_name"
    blob_name = strings::RemovePrefix(blob_name, "_");  // "blob_name"
    // Get the SegmentSegmentPair around |blob_name| in SegmentDag
    auto segment_pair = segment_dag->GetSegmentPairAroundBlob(blob_name);
    auto first_segment = segment_pair.first;
    auto second_segment = segment_pair.second;

    auto boxing_info_elem = actor_boxing_info.GetBoxingInfoElement(segment_pair);

    if (is_forward_) {
      auto output_vars = layer->GetOutputVars();
      int32_t output_num = output_vars.size();
      CHECK(output_num == boxing_info_elem.out_num());
      int32_t idx = 0;
      for (auto& output_var : output_vars) {
        auto layer_blob = strings::full_blob_name_in_layer(op_name, output_var);
        auto task_blob = task_blob_from_layer_blob(layer_blob);
        UpdateRegisterInfo(layer_blob, task_blob, second_segment, idx,
          output_num, &consumer_segment_to_produced_register_infos);
        ++idx;
      }
    } else {
      auto input_vars = layer->GetInputVars();  // revert the input/output
      int32_t input_num = input_vars.size();
      CHECK(input_num == boxing_info_elem.in_num());
      int32_t idx = 0;
      for (auto& input_var : input_vars) {
        auto layer_blob = strings::full_blob_name_in_layer(op_name, input_var);
        auto task_blob = task_blob_from_layer_blob(layer_blob);
        UpdateRegisterInfo(layer_blob, task_blob, first_segment, idx, input_num,
          &consumer_segment_to_produced_register_infos);
        ++idx;
      }
    }
  }

  for (auto& segment_infos_pair : consumer_segment_to_produced_register_infos) {
    auto consumer_segment = segment_infos_pair.first;
    auto& register_infos = segment_infos_pair.second;
    for (auto& register_info : register_infos) {
      register_info_manager_.AddProducedRegisterInfoForBoxingTask(
        register_info, consumer_segment);
    }
  }
}

template <typename Dtype>
void BoxingTaskDag<Dtype>::AddConsumedRegisterInfosInPath() {
  // In forward pass, input_vars is from preceding task.
  // In backward pass, output_vars is from preceding task.
  auto actor_dag = dag_builder_.actor_dag();
  auto segment_dag = dag_builder_.segment_dag();
  BoxingInfo actor_boxing_info(false);  // By default, not in_boxing
  if (is_forward_) {
    actor_boxing_info = actor_dag->GetForwardBoxingInfo(name_);
  } else {
    auto forward_name = actor_dag->GetForwardTaskName(name_);
    actor_boxing_info = actor_dag->GetForwardBoxingInfo(forward_name);
  }
  auto pipe_name = actor_dag->GetPipeNameFromActor(name_);

  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();
    auto blob_name = strings::RemovePrefix(op_name, pipe_name);
    blob_name = strings::RemovePrefix(blob_name, "_");
    auto segment_pair = segment_dag->GetSegmentPairAroundBlob(blob_name);
    auto first_segment = segment_pair.first;
    auto second_segment = segment_pair.second;
    auto boxing_info_element
      = actor_boxing_info.GetBoxingInfoElement(segment_pair);
    std::vector<std::string> blob_vars;
    std::vector<std::string> forward_producer_names;
    if (is_forward_) {
      blob_vars = layer->GetInputVars();
      forward_producer_names = boxing_info_element.GetOrderedInputs();
    } else {
      blob_vars = layer->GetOutputVars();
      forward_producer_names = boxing_info_element.GetOrderedOutputs();
    }
    CHECK_EQ(blob_vars.size(), forward_producer_names.size());
    int32_t producer_num = forward_producer_names.size();
    for (int32_t order = 0; order < producer_num; ++order) {
      std::string producer_name;
      if (is_forward_) {
        producer_name = forward_producer_names[order];
      } else {
        producer_name
          = actor_dag->GetBackwardTaskName(forward_producer_names[order]);
      }
      auto producer_task_id = actor_dag->GetTaskID(producer_name);
      auto producer_task_dag = dag_builder_.GetTaskDag(producer_task_id);
      int64_t group_id
        = producer_task_dag->GetImmediateProducedGroupIdInPath(name_);
      register_info_manager_.AddConsumedGroupId(group_id);
      producer_task_dag->RegisterConsumer(task_id_, group_id);
      AddBlobsToConsumedRegisterInfo(op_name, { blob_vars[order] },
        producer_task_dag, group_id, null_filter_);
    }
  }
}

template <typename Dtype>
int64_t BoxingTaskDag<Dtype>::GetImmediateProducedGroupIdInPath(
  const std::string& consumer_name) const {
  auto actor_dag = dag_builder_.actor_dag();

  BoxingInfo actor_boxing_info(false);
  std::string consumer_segment;
  SingleSideBoxingInfoElement single_side_info;
  int32_t order;
  if (is_forward_) {
    actor_boxing_info = actor_dag->GetForwardBoxingInfo(name_);
    consumer_segment
      = actor_boxing_info.SecondSegmentFromActorName(consumer_name);
    single_side_info
      = actor_boxing_info.GetSingleSideInfoFromSecondSegment(consumer_segment);
    order = single_side_info.GetOpOrder(consumer_name);
  } else {
    auto forward_name = actor_dag->GetForwardTaskName(name_);
    auto forward_producer_name = actor_dag->GetForwardTaskName(consumer_name);
    actor_boxing_info = actor_dag->GetForwardBoxingInfo(forward_name);
    consumer_segment
      = actor_boxing_info.FirstSegmentFromActorName(forward_producer_name);
    single_side_info
      = actor_boxing_info.GetSingleSideInfoFromFirstSegment(consumer_segment);
    order = single_side_info.GetOpOrder(forward_producer_name);
  }

  return register_info_manager_.GetProducedGroupIdForBoxingTask(
    consumer_segment, order);
}

template <typename Dtype>
std::string BoxingTaskDag<Dtype>::build_layer_name(const std::string& pipe_name,
  const std::string& blob_name) const {
  return strings::Join({ pipe_name, blob_name }, "_");
}
INSTANTIATE_CLASS(BoxingTaskDag);
}  // namespace caffe
