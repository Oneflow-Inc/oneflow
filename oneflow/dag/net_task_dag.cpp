#include "dag/net_task_dag.h"
#include <string>
#include <utility>
#include <vector>
#include "common/common.h"
#include "context/one.h"
#include "dag/node_meta.h"
#include "dag/segment_dag.h"
#include "dag/stage_dag.h"
#include "dag/pipe_dag.h"
#include "dag/actor_dag.h"
#include "dag/dag_builder.h"
#include "dag/boxing_task_dag.h"
#include "task/job_manager.h"
#include "layers/base_layer.h"
#include "layers/layer_factory.h"

namespace oneflow {
template <typename Dtype>
NetTaskDag<Dtype>::NetTaskDag(const DagBuilder<Dtype>& dag_builder,
  TaskType type, int32_t task_id, PathType path_type,
  const std::string& actor_name, bool is_forward) : TaskDag<Dtype>::TaskDag(
  dag_builder, type, task_id, path_type, actor_name, is_forward) {
  forward_is_sender_ = strings::Contains(actor_name, "out_net");
  is_net_receiver_ = (is_forward_ && !forward_is_sender_)
    || (!is_forward_ && forward_is_sender_);
}
template <typename Dtype>
NetTaskDag<Dtype>::~NetTaskDag() {}

template <typename Dtype>
void NetTaskDag<Dtype>::BuildForward() {
  // (1) Forward_out_net plays the role of 'sender', the 'sender' will pack the
  // separate blobs into an envelope owned by 'sender'. The 'receiver' at remote
  // machine will fetch the envelope of the 'sender' to its local envelope.
  // (2) In NetTaskDag, for a net_layer, besides the normal input/output blobs,
  // we also create an input envelope data node and an output envelope data node.
  // However, note that, not all the data nodes will be involved in an execution.
  // For example, for a 'sender' task, it only consumes the input blobs and
  // produces the output blobs, but the envelopes are not actually used.
  // On the other hand, for a 'receiver' task, it only consumes the input
  // envelope and produces an output envelope, while the normal input/output
  // blobs are not actually used.
  auto pipe_name = dag_builder_.actor_dag()->GetPipeNameFromActor(name_);
  auto logical_blobs = GetLogicalBlobsNeedTransferred();
  envelope_name_ = BuildEnvelopeName(logical_blobs);

  std::string layer_type = "Net";
  std::string layer_name = pipe_name;
  std::string proto_str = BuildProtoString(logical_blobs.size());
  auto input_task_blobs = BuildInputTaskBlobs(logical_blobs);
  auto output_task_blobs = BuildOutputTaskBlobs(logical_blobs);

  auto layer_node = AddOpNode(layer_name, layer_type, proto_str);
  auto layer = layer_node->op()->layer();

  std::vector<DataNode<BlobMeta>*> input_nodes;
  std::vector<DataNode<BlobMeta>*> output_nodes;
  auto input_vars = layer->GetInputVars();
  auto output_vars = layer->GetOutputVars();
  CHECK_EQ(input_vars.size(), input_task_blobs.size() + 1);  // + in_envelope
  CHECK_EQ(output_vars.size(), output_task_blobs.size() + 1); // + out_envelope
  int32_t idx = 0;
  for (auto input_var : input_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
    if (!strings::Contains(input_var, "envelope")) {
      auto task_blob = input_task_blobs[idx];
      input_nodes.push_back(AddDataNode(task_blob));
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blobs[idx], true);
      ++idx;
    } else {
      // For envelope blob, let the layer_blob and task_blob be same,
      // and let |envelope_name_| be the logical_blob name.
      auto task_blob = layer_blob;
      auto logical_blob = envelope_name_;
      input_nodes.push_back(AddDataNode(task_blob));
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blob, true);
    }
  }
  idx = 0;
  for (auto output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    if (!strings::Contains(output_var, "envelope")) {
      auto task_blob = output_task_blobs[idx];
      output_nodes.push_back(AddDataNode(task_blob));
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blobs[idx], false);
      ++idx;
    } else {
      // For envelope blob, let the layer_blob and task_blob be same,
      // and let |envelope_name_| be the logical_blob name.
      auto task_blob = layer_blob;
      auto logical_blob = envelope_name_;
      output_nodes.push_back(AddDataNode(task_blob));
      blob_info_manager_.RegisterBlob(
        layer_blob, task_blob, logical_blob, false);
    }
  }
  for (auto& input_node : input_nodes) {
    layer_node->AddParent(input_node);
  }
  for (auto& output_node : output_nodes) {
    output_node->AddParent(layer_node);
  }
}

template <typename Dtype>
std::vector<std::string> NetTaskDag<Dtype>::GetLogicalBlobsNeedTransferred(
  ) const {
  auto actor_dag = dag_builder_.actor_dag();
  auto pipe_dag = dag_builder_.pipe_dag();
  auto stage_dag = dag_builder_.stage_dag();
  auto segment_dag = dag_builder_.segment_dag();

  auto pipe_name = actor_dag->GetPipeNameFromActor(name_);
  StageStagePair stage_pair = pipe_dag->GetStagePairFromNetPipe(pipe_name);
  auto from_stage_name = stage_pair.first;
  auto to_stage_name = stage_pair.second;
  auto from_segment_name
    = stage_dag->GetOpNode(from_stage_name)->op()->segment_name();
  auto to_segment_name
    = stage_dag->GetOpNode(to_stage_name)->op()->segment_name();
  auto envelope_names
    = segment_dag->FindDataNodesInBetween(from_segment_name, to_segment_name);
  CHECK_EQ(envelope_names.size(), 1);
  auto logical_blobs
    = segment_dag->GetDataNode(envelope_names[0])->data()->blob_names();
  return logical_blobs;
}

template <typename Dtype>
std::string NetTaskDag<Dtype>::BuildProtoString(int32_t blob_num) const {
  int32_t in_num = blob_num;
  int32_t out_num = blob_num;

  NetProto net_proto;
  net_proto.set_forward_is_sender(forward_is_sender_);
  net_proto.set_in_envelope("in_envelope");
  net_proto.set_in_num(in_num);
  net_proto.set_out_envelope("out_envelope");
  net_proto.set_out_num(out_num);
  std::string net_proto_str = net_proto.DebugString();
  return net_proto_str;
}

template <typename Dtype>
std::vector<std::string> NetTaskDag<Dtype>::BuildInputTaskBlobs(
  const std::vector<std::string>& logical_blobs) const {
  std::vector<std::string> input_task_blobs;
  for (auto& logical_blob : logical_blobs) {
    input_task_blobs.push_back("net_from/" + logical_blob);
  }
  return input_task_blobs;
}

template <typename Dtype>
std::vector<std::string> NetTaskDag<Dtype>::BuildOutputTaskBlobs(
  const std::vector<std::string>& logical_blobs) const {
  std::vector<std::string> output_task_blobs;
  for (auto& logical_blob : logical_blobs) {
    output_task_blobs.push_back("net_to/" + logical_blob);
  }
  return output_task_blobs;
}

template <typename Dtype>
std::string NetTaskDag<Dtype>::BuildEnvelopeName(
  const std::vector<std::string>& blobs) const {
  std::string envelope = "envelope";
  for (auto& blob : blobs) {
    envelope += "_" + blob;
  }
  return envelope;
}

template <typename Dtype>
void NetTaskDag<Dtype>::AddProducedRegisterInfos() {
  CHECK_EQ(op_name_to_node_.size(), 1) << "Only a single net layer";
  auto begin = op_name_to_node_.begin();
  auto op_name = begin->first;
  auto op_node = begin->second;
  auto op_meta = op_node->op();
  auto layer = op_meta->layer();

  auto&& id_map = oneflow::TheOne<Dtype>::id_map();
  int32_t group_local_id = id_map->new_group_local_id(task_id_);
  int64_t group_id
    = id_map->group_id_from_task_id_and_group_local_id(task_id_, group_local_id);

  RegisterType register_type;
  if (is_forward_) {
    register_type = RegisterType::kDataType;
  } else {
    register_type = RegisterType::kDataDiffType;
  }
  DeviceType device_type{ DeviceType::kCPU };
  RegisterInfo register_info(register_type, device_type, group_id, true);
  bool is_sender = (is_forward_ && forward_is_sender_)    // forward, out_net
    || (!is_forward_ && !forward_is_sender_);             // backward, in_net
  std::vector<std::string> blob_vars
    = is_forward_ ? layer->GetOutputVars() : layer->GetInputVars();

  AddBlobsToProducedRegisterInfo(op_name, blob_vars, &register_info,
    EnvelopeFlag::kInEnvelope, is_not_envelope_);

  AddBlobsToProducedRegisterInfo(op_name, blob_vars, &register_info,
    EnvelopeFlag::kOnEnvelope, is_envelope_);

  //if (is_sender) {
  //  AddBlobsToProducedRegisterInfo(op_name, blob_vars, &register_info,
  //    EnvelopeFlag::kInEnvelope, is_not_envelope_);
  //} else {
  //  AddBlobsToProducedRegisterInfo(op_name, blob_vars, &register_info,
  //    EnvelopeFlag::kOnEnvelope, is_envelope_);
  //}

  register_info_manager_.AddProducedRegisterInfoForNonBoxingTask(register_info);
}

template <typename Dtype>
void NetTaskDag<Dtype>::AddConsumedRegisterInfosInPath() {
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

  // 3, Collect all the blobs that should be obtained in the producer's dst
  // register
  CHECK_EQ(op_name_to_node_.size(), 1) << "Only a single net layer";
  auto begin = op_name_to_node_.begin();
  auto op_name = begin->first;
  auto op_node = begin->second;
  auto op_meta = op_node->op();
  auto layer = op_meta->layer();

  bool is_sender = (is_forward_ && forward_is_sender_)    // forward, out_net
    || (!is_forward_ && !forward_is_sender_);             // backward, in_net
  std::vector<std::string> blob_vars
    = is_forward_ ? layer->GetInputVars() : layer->GetOutputVars();
  if (is_sender) {
    AddBlobsToConsumedRegisterInfo(op_name, blob_vars, producer_task_dag,
      group_id, is_not_envelope_);
  } else {
    AddBlobsToConsumedRegisterInfo(op_name, blob_vars, producer_task_dag,
      group_id, is_envelope_);
  }
}

template <typename Dtype>
void NetTaskDag<Dtype>::ForwardSetup() {
  LOG(INFO) << "Setup: " << name_;
  const std::string forward_in_net_prefix = "forward_in_net";
  if (strings::StartsWith(name_, forward_in_net_prefix)) {
    ForwardSetupInNetTask();
  } else {
    TaskDag<Dtype>::ForwardSetup();
  }
}

template <typename Dtype>
void NetTaskDag<Dtype>::ForwardSetupInNetTask() {
  //// 1, Get the corresponding out_net TaskDag
  //auto actor_dag = dag_builder_.actor_dag();
  //auto predecessors = actor_dag->GetPrecedingOpNodeNames(name_);
  //CHECK_EQ(predecessors.size(), 1);
  //auto out_net_name = predecessors[0];
  //int32_t out_net_task_id = actor_dag->GetTaskID(out_net_name);
  //auto out_net_task_dag = dag_builder_.GetTaskDag(out_net_task_id);

  //// 2, Get the output blobs' shape
  //auto blob_names = GetPreceedingDataNodeNamesOfEndNode();
  //CHECK_GE(blob_names.size(), 1);
  //for (auto& blob_name : blob_names) {
  //  Shape shape
  //    = out_net_task_dag->GetBlobShapeFromNonBoxingActor(blob_name);
  //  auto blob_node = GetDataNode(blob_name);
  //  blob_node->data()->mutable_shape() = shape;
  //}

  //// 3, Update the blobs'shape in net_layer's DataParam
  //auto op_names = GetFirstOpNames();
  //CHECK(op_names.size() == 1);
  //auto layer_name = op_names[0];
  //auto op_node = GetOpNode(layer_name);
  //auto layer_meta = op_node->op();
  //auto layer = layer_meta->mutable_layer();

  //std::shared_ptr<DataParam<Dtype>> data_param(layer->CreateDataParam());
  //data_param->AllocateEmptyBlobs();
  //// 3.1 Set the output shape of |data_param|
  //auto output_vars = layer->GetOutputVars();
  //for (auto& output_var : output_vars) {
  //  auto name_in_layer
  //    = strings::full_blob_name_in_layer(layer_name, output_var);
  //  auto name_in_dag
  //    = blob_name_in_layer_to_name_in_dag_.GetValueWithKey(name_in_layer);
  //  auto dnode = GetDataNode(name_in_dag);
  //  auto& blob_meta = dnode->data();
  //  data_param->SetShape(name_in_layer, blob_meta->shape());
  //}
  //// 3.2 Update the internal blobs' shape of layer's DataParam
  //layer->InitFromInputShape(data_param.get());

  //// 3.3 Set the input shape of |data_param|
  //auto input_vars = layer->GetInputVars();
  //for (auto& input_var : input_vars) {
  //  auto name_in_layer
  //    = strings::full_blob_name_in_layer(layer_name, input_var);
  //  auto name_in_dag
  //    = blob_name_in_layer_to_name_in_dag_.GetValueWithKey(name_in_layer);
  //  auto dnode = GetDataNode(name_in_dag);
  //  auto& blob_meta = dnode->data();
  //  Shape shape = data_param->GetShape(name_in_layer);
  //  blob_meta->mutable_shape() = shape;
  //}
}

//INSTANTIATE_CLASS(NetTaskDag);
}  // namespace oneflow
