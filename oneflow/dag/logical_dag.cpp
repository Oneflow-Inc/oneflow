#include "dag/logical_dag.h"
#include <ctype.h>
#include <set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <algorithm>
#include "common/common.h"
#include "common/blob_name_converter.h"
#include "context/one.h"
#include "context/config_parser.h"
#include "context/net_descriptor.h"
#include "context/strategy_descriptor.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "layers/base_layer.h"
#include "layers/loader_layer.h"
#include "layers/layer_factory.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
LogicalDag<Dtype>::LogicalDag(std::shared_ptr<NetDescriptor> net_descriptor,
  PathType path_type, const std::string& name) : Dag(path_type, name),
  net_descriptor_(net_descriptor) {
  Build();
  // Setup();
  PrintDag(name_);
}

template <typename Dtype>
LogicalDag<Dtype>::~LogicalDag() {}

template <typename Dtype>
bool LogicalDag<Dtype>::DagBlobNeedsBP(const std::string& dag_blob) const {
  auto need_bp_it = dag_blob_to_need_bp_.find(dag_blob);
  CHECK(need_bp_it != dag_blob_to_need_bp_.end());
  return need_bp_it->second;
}

template <typename Dtype>
void LogicalDag<Dtype>::Build() {
  int32_t layer_num = net_descriptor_->layer_num();
  for (auto layer_id = 0; layer_id < layer_num; ++layer_id) {
    ProcessLayer(layer_id);
  }
  AddOtherLayerBlobAndDagBlobMap();
  AddStartAndEndNodes();
  PostProcessing();

  SetDagBlobNeedsBP();
}

template <typename Dtype>
void LogicalDag<Dtype>::ProcessLayer(int32_t layer_id) {
  LayerProto layer_proto = net_descriptor_->layer_proto(layer_id);
  auto& message = LayerParameterIntegrityCheck(layer_proto);

  std::string op_name = layer_proto.name();
  std::string op_type = layer_proto.type();
  auto op_node = AddOpNode(op_name, op_type, layer_proto);

  auto op_meta = op_node->op();
  auto layer = op_meta->layer();

  auto input_vars = layer->GetInputVars();
  std::vector<DNode*> input_nodes;
  for (auto& input_var : input_vars) {
    std::string dag_blob;
    // |input_var| has a prefix "data/" which not occurs in text protobuf
    auto no_prefix = strings::remove_prefix_of_blob_variable_name(input_var);
    GetStringValueByKeyOrDie(message, no_prefix, &dag_blob);
    DNode* bottom_dnode;
    if (available_tops_.count(dag_blob)) {
      // |dag_blob| is already created.
      bottom_dnode = GetDataNode(dag_blob);
    } else {
      // |dag_blob| is not declared before, need to create a new DataNode.
      bottom_dnode = AddDataNode(dag_blob);
    }
    input_nodes.push_back(bottom_dnode);
    auto layer_blob = strings::full_blob_name_in_layer(op_name, input_var);
    layer_blob_to_dag_blob_.AddPair(layer_blob, dag_blob);
    dag_blob_to_layer_blobs_.AddTriple(
      dag_blob, op_name, input_var, LayerBlobRole::kInput);
  }

  auto output_vars = layer->GetOutputVars();
  std::vector<DNode*> output_nodes;
  for (auto& output_var : output_vars) {
    std::string alias_name;
    auto no_prefix = strings::remove_prefix_of_blob_variable_name(output_var);
    GetStringValueByKeyOrDie(message, no_prefix, &alias_name);
    // TODO(jiyuan): whether the following requirement is a MUST?
    CHECK(no_prefix == alias_name)
      << "Ensure the key and value are equal while specifying a layer's output";

    auto layer_blob = strings::full_blob_name_in_layer(op_name, output_var);
    auto dag_blob = strings::full_blob_name_in_dag(op_name, alias_name);
    output_nodes.push_back(AddDataNode(dag_blob));
    CHECK(available_tops_.count(dag_blob) == 0)
      << "Ensure each blob is generated exactly once";
    available_tops_.insert(dag_blob);
    layer_blob_to_dag_blob_.AddPair(layer_blob, dag_blob);
    dag_blob_to_layer_blobs_.AddTriple(
      dag_blob, op_name, output_var, LayerBlobRole::kOutput);
  }
  AddEdges(op_node, input_nodes, output_nodes);
}

template <typename Dtype>
void LogicalDag<Dtype>::AddOtherLayerBlobAndDagBlobMap() {
  for (auto& op_name_node_pair : op_name_to_node_) {
    auto op_name = op_name_node_pair.first;
    auto op_node = op_name_node_pair.second;
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    AddLayerBlobs(op_name, layer->GetOtherVars());
    AddLayerBlobs(op_name, layer->GetModelVars());
    AddLayerBlobs(op_name, layer->GetTempVars());
  }
}

template <typename Dtype>
void LogicalDag<Dtype>::AddLayerBlobs(const std::string& layer_name,
  const std::vector<std::string>& vars) {
  for (auto& var : vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, var);
    auto dag_blob = layer_blob;  // Let |dag_blob| be same as |layer_blob|
    layer_blob_to_dag_blob_.AddPair(layer_blob, dag_blob);
  }
}

template <typename Dtype>
void LogicalDag<Dtype>::SetDagBlobNeedsBP() {
  DagIterator<LogicalDag<Dtype>> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kDataNode) continue;
    auto dag_blob = current_node->node_name();

    auto triple = dag_blob_to_layer_blobs_.GetTriples(dag_blob);
    bool need_bp = GetDagBlobNeedsBP(triple);
    dag_blob_to_need_bp_.insert({ dag_blob, need_bp });
  }
}

template <typename Dtype>
bool LogicalDag<Dtype>::GetDagBlobNeedsBP(const LayerBlobTriples& triples) const {
  LayerBlobTriples producer_triples;
  LayerBlobTriples consumer_triples;
  for (auto& triple : triples) {
    if (triple.role == LayerBlobRole::kOutput) {
      producer_triples.push_back(triple);
    } else {
      consumer_triples.push_back(triple);
    }
  }
  // A dag_blob can only have at most one producer, but can have more than one
  // consumers.
  CHECK_LE(producer_triples.size(), 1);
  bool producer_needs_diff = false;
  bool consumer_has_diff = false;
  for (auto& triple : producer_triples) {
    auto layer_name = triple.layer_name;
    auto layer_node = GetOpNode(layer_name);
    auto layer_meta = layer_node->op();
    auto layer = layer_meta->layer();
    auto output_diffs = layer->GetOutputDiffs();
    // Check whether there is an output_diff corresponding to the output blob
    if (strings::has_diff_correspondence({ triple.var_name }, output_diffs)) {
      producer_needs_diff = true;
    }
  }
  for (auto& triple : consumer_triples) {
    auto layer_name = triple.layer_name;
    auto layer_node = GetOpNode(layer_name);
    auto layer_meta = layer_node->op();
    auto layer = layer_meta->layer();
    auto input_diffs = layer->GetInputDiffs();
    if (strings::has_diff_correspondence({ triple.var_name }, input_diffs)) {
      consumer_has_diff = true;
    }
  }
  return producer_needs_diff && consumer_has_diff;
}

// NOTE(jiyuan): It seems that we do not need this member.
//template <typename Dtype>
//void LogicalDag<Dtype>::Setup() {
//  DagIterator<LogicalDag<Dtype>> dag_iterator(*this);
//  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
//    auto current_node = dag_iterator.CurrentNode();
//    if (current_node->Type() == NodeType::kDataNode) {
//      auto current_dnode = dynamic_cast<DNode*>(current_node);
//      CHECK_NOTNULL(current_dnode);
//      SetupDataNode(current_dnode);
//    } else if (current_node->Type() == NodeType::kOpNode) {
//      auto current_onode = dynamic_cast<ONode*>(current_node);
//      CHECK_NOTNULL(current_onode);
//      SetupOpNode(current_onode);
//    } else {
//      // The start and end nodes have the kUnknown node type,
//      // do nothing for them.
//    }
//  }
//}
//
//template <typename Dtype>
//void LogicalDag<Dtype>::SetupDataNode(DNode* dnode) {
//  auto &blob_meta = dnode->data();
//  const Shape& shape = blob_meta->shape();
//  DLOG(INFO) << "Node type: kDataNode";
//  DLOG(INFO) << "Node name: " << blob_meta->name();
//  DLOG(INFO) << "Node shape: " << shape.shape_string();
//}
//
//template <typename Dtype>
//void LogicalDag<Dtype>::SetupOpNode(ONode* onode) {
//  auto &layer_meta = onode->op();
//  DLOG(INFO) << "Node type: kOpNode";
//  DLOG(INFO) << "Layer type: " << layer_meta->type();
//  auto& layer = layer_meta->mutable_layer();
//  auto layer_name = onode->node_name();
//  // If this layer is data LoaderLayer, firstly set the batch
//  // size before calling InitFromInputShape()
//  if (this->IsDataProviderNode(onode)) {
//    auto loader_layer = dynamic_pointer_cast<LoaderLayer<Dtype>>(layer);
//    CHECK_NOTNULL(loader_layer.get());
//    auto& strategy_descriptor
//      = caffe::TheOne<Dtype>::config_parser()->strategy_descriptor();
//    auto max_data_parallel_num = strategy_descriptor->max_data_parallel_num();
//    loader_layer->SetBatchSize(max_data_parallel_num);
//  }
//
//  std::shared_ptr<DataParam<Dtype>> data_param(layer->CreateDataParam());
//  data_param->AllocateEmptyBlobs();
//
//  // Get input shape
//  auto input_vars = layer->GetInputVars();
//  for (auto& input_var : input_vars) {
//    auto layer_input_name
//      = strings::full_blob_name_in_layer(layer_name, input_var);
//    if (IsPSDag() && !dag_blob_dict_.HasLayerBlobName(layer_input_name)) {
//      continue;
//    }
//    std::string dag_blob_name =
//      dag_blob_dict_.GetBlobNameInDag(layer_input_name);
//    auto dnode = GetDataNode(dag_blob_name);
//    auto& blob_meta = dnode->data();
//    data_param->SetShape(layer_input_name, blob_meta->shape());
//  }
//
//  layer->InitFromInputShape(data_param.get());
//
//  // Set output shape
//  auto output_vars = layer->GetOutputVars();
//  for (auto& output_var : output_vars) {
//    auto layer_output_name
//      = strings::full_blob_name_in_layer(layer_name, output_var);
//    std::string dag_blob_name =
//      dag_blob_dict_.GetBlobNameInDag(layer_output_name);
//    auto dnode = GetDataNode(dag_blob_name);
//    auto& blob_meta = dnode->data();
//    blob_meta->mutable_shape() = data_param->GetShape(layer_output_name);
//  }
//  // PrintModelShape()
//}

template <typename Dtype>
OpNode<LayerMeta<Dtype>>* LogicalDag<Dtype>::AddOpNode(
  const std::string& op_name,
  const std::string& op_type,
  const LayerProto& layer_param) {
  auto op_node = NewOpNode(op_name);
  auto& layer_meta = op_node->mutable_op();
  layer_meta = std::make_shared<LayerMeta<Dtype>>(op_type);
  std::string& param_str = layer_meta->mutable_param_str();

  // NOTE(jiyuan): the variable name in caffe.proto should be the lower
  // correspondence of its type name, plus "_proto"
  std::string field_name;
  field_name.resize(op_type.length());
  std::transform(op_type.begin(),
    op_type.end(), field_name.begin(), ::tolower);
  field_name = field_name + "_proto";

  GetMessageContentByKeyOrDie(layer_param, field_name, &param_str);
  auto& layer = layer_meta->mutable_layer();
  layer = LayerRegistry<Dtype>::CreateLayer(op_type, op_name, param_str);
  layer->InitParam();
  layer_meta->mutable_has_BP() = layer->has_BP();

  // TODO(jiyuan): set has_BP_ of |layer_meta|
  auto it = op_name_to_node_.find(op_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << op_name;
  op_name_to_node_.insert({ op_name, op_node });

  return op_node;
}

template <typename Dtype>
DataNode<BlobMeta>* LogicalDag<Dtype>::AddDataNode(
  const std::string& data_name) {
  auto data_node = NewDataNode(data_name);
  auto node_id = data_node->node_id();
  auto& blob_meta = data_node->mutable_data();
  blob_meta = std::make_shared<BlobMeta>(data_name);
  auto it = data_name_to_node_.find(data_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << data_name;
  data_name_to_node_.insert({ data_name, data_node });
  return data_node;
}

template <typename Dtype>
void LogicalDag<Dtype>::GetMessageContentByKeyOrDie(
  const google::protobuf::Message& proto,
  const std::string& key, std::string *value) {
  const google::protobuf::Descriptor *d = proto.GetDescriptor();
  const google::protobuf::Reflection *r = proto.GetReflection();

  const google::protobuf::FieldDescriptor *fd = d->FindFieldByName(key);
  CHECK_NOTNULL(fd);
  if (fd) {
    auto& m = r->GetMessage(proto, fd);
    PrintProtoToString(m, value);
  }
}
template <typename Dtype>
void LogicalDag<Dtype>::GetStringValueByKeyOrDie(
  const google::protobuf::Message& proto,
  const std::string& key, std::string *value) {
  const google::protobuf::Descriptor *d = proto.GetDescriptor();
  const google::protobuf::Reflection *r = proto.GetReflection();

  if (strings::is_array_blob(key)) {
    // For ConcatLayer, its input may be an array of input blobs.
    int32_t idx;
    std::string sub_key;
    strings::parse_blob_variable_name_in_array(key, &idx, &sub_key);
    const google::protobuf::FieldDescriptor *fd = d->FindFieldByName(sub_key);
    CHECK_NOTNULL(fd);
    if (fd) {
      *value = r->GetRepeatedString(proto, fd, idx);
    }
  } else {
    const google::protobuf::FieldDescriptor *fd = d->FindFieldByName(key);
    CHECK_NOTNULL(fd);
    if (fd) {
      *value = r->GetString(proto, fd);
    }
  }
}

template <typename Dtype>
const google::protobuf::Message&
LogicalDag<Dtype>::LayerParameterIntegrityCheck(
  const LayerProto& layer_param) {
  const google::protobuf::Descriptor *d = layer_param.GetDescriptor();
  const google::protobuf::Reflection *r = layer_param.GetReflection();

  // Check the name field
  const google::protobuf::FieldDescriptor *fd = d->FindFieldByName("name");
  CHECK(r->HasField(layer_param, fd))
    << "The name field must be set in the layer parameter";
  std::string op_name = r->GetString(layer_param, fd);

  // Check the type field
  fd = d->FindFieldByName("type");
  CHECK(r->HasField(layer_param, fd))
    << "The type field must be set in the layer parameter";
  std::string op_type = r->GetString(layer_param, fd);
  std::string param_name;
  param_name.resize(op_type.length());
  std::transform(op_type.begin(),
    op_type.end(), param_name.begin(), ::tolower);
  param_name = param_name + "_proto";

  std::vector<const google::protobuf::FieldDescriptor*> fds;
  r->ListFields(layer_param, &fds);
  for (int32_t id = 0; id < fds.size(); ++id) {
    if (fds[id]->name() == "name") continue;
    if (fds[id]->name() == "type") continue;
    CHECK_EQ(fds[id]->name(), param_name)
      << "Please adhere to the naming rule of layer parameter. "
      << op_type << "\t" << param_name << "\t" << fds[id]->name();
    return r->GetMessage(layer_param, fds[id]);
  }
}

template <typename Dtype>
std::string LogicalDag<Dtype>::DagBlobFromLayerBlob(
  const std::string& layer_blob) const {
  return layer_blob_to_dag_blob_.GetValueWithKey(layer_blob);
}

template <typename Dtype>
PlacementInfo LogicalDag<Dtype>::GetPlacementInfo(
  const std::string& layer_name) const {
  auto layer_node = GetOpNode(layer_name);
  auto layer_meta = layer_node->op();
  auto& placement_info = layer_meta->placement_info();
  return placement_info;
}

template <typename Dtype>
bool LogicalDag<Dtype>::BlobDict::HasKey(const std::string& key) const {
  auto value_it = dict_.find(key);
  return value_it != dict_.end();
}

template <typename Dtype>
void LogicalDag<Dtype>::BlobDict::AddPair(
  const std::string& key, const std::string& value) {
  CHECK(dict_.count(key) == 0);
  dict_.insert({ key, value });
}

template <typename Dtype>
const std::string& LogicalDag<Dtype>::BlobDict::GetValueWithKey(
  const std::string& key) const {
  auto value_it = dict_.find(key);
  CHECK(value_it != dict_.end());
  return value_it->second;
}

template <typename Dtype>
void LogicalDag<Dtype>::DagBlobToLayerBlobs::AddTriple(
  const std::string& dag_blob, const std::string& layer_name,
  const std::string& var_name, LayerBlobRole role) {
  auto triple_it = dag_blob_to_triples_.find(dag_blob);
  if (triple_it == dag_blob_to_triples_.end()) {
    LayerBlobTriples triples;
    triples.push_back({ layer_name, var_name, role });
    dag_blob_to_triples_.insert({ dag_blob, triples });
  } else {
    triple_it->second.push_back({ layer_name, var_name, role });
  }
}

template <typename Dtype>
typename LogicalDag<Dtype>::LayerBlobTriples
LogicalDag<Dtype>::DagBlobToLayerBlobs::GetTriples(const std::string& dag_blob) const {
  auto triple_it = dag_blob_to_triples_.find(dag_blob);
  CHECK(triple_it != dag_blob_to_triples_.end());
  return triple_it->second;
}

INSTANTIATE_CLASS(LogicalDag);
}  // namespace caffe