#include <unordered_map>
#include <string>
#include <vector>
#include "proto/oneflow.pb.h"
#include "common/common.h"
#include "context/config_parser.h"
#include "context/net_descriptor.h"
#include "context/strategy_descriptor.h"
#include "context/one.h"
#include "dag/actor_dag.h"
#include "dag/dag_builder.h"
#include "dag/dag_iterator.h"
#include "dag/node_meta.h"
#include "dag/logical_dag.h"
#include "dag/pipe_dag.h"
#include "dag/placement_group_dag.h"
#include "dag/segment_dag.h"
#include "dag/stage_dag.h"
#include "layers/base_layer.h"
#include "path/data_path.h"
#include "path/model_store_path.h"
#include "path/path_share_policy.h"
#include "path/path_manager.h"

namespace oneflow {
template <typename Dtype>
ModelStorePath<Dtype>::ModelStorePath(std::shared_ptr<DataPath<Dtype>> data_path,
   PathManager<Dtype>* path_manager)
  : BasePath<Dtype>::BasePath(PathType::kModelStorePath, path_manager), data_path_(data_path),
    is_train_(data_path_->is_train()), index_(0), seekpos_(0) {
}

template <typename Dtype>
bool ModelStorePath<Dtype>::is_train() const {
  return is_train_;
}

template <typename Dtype>
std::shared_ptr<DagBuilder<Dtype>>
ModelStorePath<Dtype>::dag_builder_of_data_path() const {
  return data_path_->GetDagBuilder(data_path_->net_name());
}

template <typename Dtype>
void ModelStorePath<Dtype>::Build() {
  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*segment_dag_of_data_path);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto segment_name = current_node->node_name();
    if (segment_dag_of_data_path->NeedModelUpdate(segment_name)) {
      BuildModelStoreDagsForSegment(segment_name);
    }
  }
}

template <typename Dtype>
void ModelStorePath<Dtype>::Setup() {

}

template <typename Dtype>
void ModelStorePath<Dtype>::BuildModelStoreDagsForSegment(
  const std::string& segment_name_in_data_path) {

  std::string net_name_in_model_store_path = segment_name_in_data_path;
  std::shared_ptr<DagBuilder<Dtype>> dag_builder;

  NetParameter net_param;
  Strategy strategy;
  NetParameterForModelStorePath(segment_name_in_data_path,
    net_name_in_model_store_path, &net_param);
  StrategyForModelStorePath(segment_name_in_data_path, &strategy);

  auto resource = oneflow::TheOne<Dtype>::config_parser()->resource_descriptor();
  std::shared_ptr<NetDescriptor> net_descriptor(new NetDescriptor(net_param));
  std::shared_ptr<StrategyDescriptor> strategy_descriptor(
    new StrategyDescriptor(strategy, resource));

  dag_builder.reset(new DagBuilder<Dtype>(net_name_in_model_store_path, this,
    net_descriptor, strategy_descriptor));
  dag_builder->Build();
  dag_builder_dict_.insert({ dag_builder->net_name(), dag_builder });
}

template <typename Dtype>
void ModelStorePath<Dtype>::NetParameterForModelStorePath(
  const std::string& segment_name_in_data_path,
  const std::string& net_name_in_model_store_path,
  NetParameter* net_parameter) {
  const std::string model_name = "model";
  const std::string net_name_in_model_update_path = net_name_in_model_store_path;
  const std::string segment_name_in_model_update_path
    = path_manager_->GetUpdateSegmentNameInModelUpdatePath();
  const std::string placeholder_segment_name = placeholder_layer_name_;

  net_parameter->set_name(net_name_in_model_store_path);

  auto layer = net_parameter->add_layer();
  layer->set_name(placeholder_layer_name_);
  layer->set_type(placeholder_type_name_);
  auto placeholder_proto = layer->mutable_placeholder_proto();
  placeholder_proto->set_out(model_name);

  auto store_layer = net_parameter->add_layer();
  store_layer->set_name(store_layer_name_);
  store_layer->set_type(store_type_name_);
  auto store_proto = store_layer->mutable_store_proto();
  store_proto->set_in(
    strings::Join({ placeholder_layer_name_, model_name }, "/"));
  SetStoreProto(segment_name_in_data_path, store_proto);

  PathSharingDescriptor model_sharing_desc{
    {
      PathSharingRole::kProducer,
      PathType::kModelUpdatePath,
      net_name_in_model_update_path,
      segment_name_in_model_update_path,
      RegisterType::kDataType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kYes
    },
    {
      PathSharingRole::kConsumer,
      PathType::kModelStorePath,
      net_name_in_model_store_path,
      placeholder_segment_name,
      RegisterType::kDataType,
      TaskDirection::kForward,
      TaskPlaceholder::kYes,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_sharing_desc);
}

template <typename Dtype>
void ModelStorePath<Dtype>::SetStoreProto(
  const std::string& segment_name_in_data_path, oneflow::StoreProto* store_proto) {

  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();
  auto segment_node
    = segment_dag_of_data_path->GetOpNode(segment_name_in_data_path);
  auto segment_meta = segment_node->op();
  auto layer_names = segment_meta->layer_names();
  auto logical_dag = dag_builder_of_data_path()->logical_dag();
  //auto logical_dag = dag_builder_of_data_path()->logical_dag();
  //// NOTE(jiyuan): why add the filed of 'stop'?
  //if (logical_dag->IsLastOpNode(segment_node)) store_proto->set_stop(true);

  for (auto& layer_name : layer_names) {
    auto layer_node = logical_dag->GetOpNode(layer_name);
    auto layer_node_ptr =
      dynamic_cast<const OpNode<LayerMeta<Dtype>>*>(layer_node);
    CHECK_NOTNULL(layer_node_ptr);
    auto layer_meta = layer_node_ptr->op();
    auto layer = layer_meta->layer();
    auto model_param = layer->GetModelParam();
    auto model_vars = layer->GetModelVars();
    if (model_vars.size() > 0) {
      store_proto->add_store_layer_names(layer_name);
      int64_t size = 0;
      for (auto& model_var : model_vars) {
        auto blob_name = strings::full_blob_name_in_layer(layer_name, model_var);
        auto shape = model_param->GetShape(blob_name);
        size += shape.count();
      }
      store_proto->add_store_layer_shapes(size);
      store_proto->add_layer_seek_pos(this->seekpos_);

      this->seekpos_
        += 2 * sizeof(int32_t) + layer_name.size() * sizeof(char) + size;
    }
  }
}

template <typename Dtype>
void ModelStorePath<Dtype>::StrategyForModelStorePath(
  const std::string& segment_name_in_data_path,
  Strategy* strategy) {
  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();
  auto policy
    = segment_dag_of_data_path->ParallelPolicyOfSegment(segment_name_in_data_path);
  auto device_set
    = segment_dag_of_data_path->DeviceSetOfSegment(segment_name_in_data_path);
  int32_t device_num = device_set.size();
  CHECK_GT(device_num, 0);
  int32_t device_set_begin = device_set.front();
  int32_t device_set_end = device_set.back();
  auto placeholder_device_group = new oneflow::DeviceGroup();
  placeholder_device_group->set_begin(device_set_begin);
  placeholder_device_group->set_end(device_set_end);

  auto placeholder_placement_group = strategy->add_placement_group();
  placeholder_placement_group->set_name(placeholder_layer_name_);
  placeholder_placement_group->set_parallel_policy(policy);
  placeholder_placement_group->mutable_layer_set()->add_name(placeholder_layer_name_);
  placeholder_placement_group->set_allocated_device_group(placeholder_device_group);

  // No need to set the device range
  auto store_device_group = new oneflow::DeviceGroup();

  auto store_placement_group = strategy->add_placement_group();
  store_placement_group->set_name(store_layer_name_);
  store_placement_group->set_parallel_policy(kNaiveParallelOnSingleMachine);
  store_placement_group->mutable_layer_set()->add_name(store_layer_name_);
  store_placement_group->set_allocated_device_group(store_device_group);
}
//INSTANTIATE_CLASS(ModelStorePath);
}  // namespace oneflow
