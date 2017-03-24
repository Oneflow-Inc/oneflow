#include "path/model_update_path.h"
#include "common/common.h"
#include "dag/node_meta.h"
#include "dag/logical_dag.h"
#include "dag/placement_group_dag.h"
#include "dag/segment_dag.h"
#include "dag/stage_dag.h"
#include "dag/pipe_dag.h"
#include "dag/actor_dag.h"
#include "dag/dag_iterator.h"
#include "dag/dag_builder.h"
#include "path/data_path.h"
#include "layers/base_layer.h"
#include "oneflow.pb.h"
#include "context/one.h"
#include "context/config_parser.h"
#include "context/net_descriptor.h"
#include "context/strategy_descriptor.h"
#include "path/path_manager.h"

namespace oneflow {
template <typename Dtype>
ModelUpdatePath<Dtype>::ModelUpdatePath(std::shared_ptr<DataPath<Dtype>> data_path,
  PathManager<Dtype>* path_manager)
  : BasePath(PathType::kModelUpdatePath, path_manager), data_path_(data_path),
    is_train_(data_path_->is_train()) {
}

template <typename Dtype>
bool ModelUpdatePath<Dtype>::is_train() const {
  return is_train_;
}

template <typename Dtype>
std::shared_ptr<DagBuilder<Dtype>>
ModelUpdatePath<Dtype>::dag_builder_of_data_path() const {
  return data_path_->GetDagBuilder(data_path_->net_name());
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::Build() {
  auto data_path_segment_dag = dag_builder_of_data_path()->segment_dag();
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*data_path_segment_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto segment_name = current_node->node_name();
    if (is_train_) {
      if (data_path_segment_dag->NeedModelUpdate(segment_name)) {
        BuildModelUpdateDagsForSegment(segment_name);
      } else if (data_path_segment_dag->NeedNullUpdate(segment_name)) {
        // For example, a segment only has kTemp vars, but without kModel vars.
        BuildNullUpdateDagsForSegment(segment_name);
      }
    } else {
      BuildNullUpdateDagsForSegment(segment_name);
    }
  }
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::Setup() {

}

template <typename Dtype>
void ModelUpdatePath<Dtype>::BuildNullUpdateDagsForSegment(
  const std::string& segment_name) {
  // LOG(FATAL) << "Not implemented";
  // TODO(jiyuan): Two scenarios use null update:
  // (1) in testing procedure, the case could be much simplified. We
  // don't need to accept gradients and add them to the old-version model to
  // get a new one. Instead, we only need to hold the model there unchanged.
  // Especially for the kDataParallelOnMultipleDevices case, we don't need to
  // build the PSDag anymore.
  // (2) the segment just contains kTemp variables, no kModel variables.
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::BuildModelUpdateDagsForSegment(
  const std::string& segment_name) {
  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();

  auto policy = segment_dag_of_data_path->ParallelPolicyOfSegment(segment_name);
  auto device_set = segment_dag_of_data_path->DeviceSetOfSegment(segment_name);
  int32_t device_num = device_set.size();
  CHECK_GT(device_num, 0);

  if (device_num > 1) {
    if (policy == kModelParallelOnMultipleDevices) {
      CreateModelUpdateDags(
        segment_name, ModelUpdateType::kModelParallelismOnMultipleDevices);
    } else if (policy == kDataParallelOnMultipleDevices) {
      CreateModelUpdateDags(
        segment_name, ModelUpdateType::kDataParallelismOnMultipleDevices);
    }
  } else {
    CHECK(policy == kNaiveParallelOnSingleDevice)
      << "A segment is bound to a single device, "
      << "we require its parallel policy is kNaiveParallelOnSingleDevice";
    CreateModelUpdateDags(
      segment_name, ModelUpdateType::kOnSingleDevice);
  }
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::CreateModelUpdateDags(
  const std::string& segment_name_in_data_path,
  ModelUpdateType model_update_type) {
  // Use the |segment_name_in_data_path| as the network's name in model update
  // path.
  std::string net_name_in_model_update_path = segment_name_in_data_path;
  std::shared_ptr<DagBuilder<Dtype>> dag_builder;

  NetParameter net_param;
  Strategy strategy;

  switch (model_update_type) {
  case ModelUpdateType::kOnSingleDevice:
    NetParameterForSingleDevice(segment_name_in_data_path,
      net_name_in_model_update_path, &net_param);
    StrategyForSingleDevice(segment_name_in_data_path, &strategy);
    break;
  case ModelUpdateType::kDataParallelismOnMultipleDevices:
    NetParameterForDataParallelOnMultipleDevice(segment_name_in_data_path,
      net_name_in_model_update_path, &net_param);
    StrategyForDataParallelOnMultipleDevice(
      segment_name_in_data_path, &strategy);
    break;
  case ModelUpdateType::kModelParallelismOnMultipleDevices:
    NetParameterForModelParallelOnMultipleDevices(segment_name_in_data_path,
      net_name_in_model_update_path, &net_param);
    StrategyForModelParallelOnMultipleDevices(
      segment_name_in_data_path, &strategy);
    break;
  }
  auto resource = oneflow::TheOne<Dtype>::config_parser()->resource_descriptor();
  std::shared_ptr<NetDescriptor> net_descriptor(new NetDescriptor(net_param));
  std::shared_ptr<StrategyDescriptor>
    strategy_descriptor(new StrategyDescriptor(strategy, resource));

  dag_builder.reset(new DagBuilder<Dtype>(net_name_in_model_update_path,
    this,
    net_descriptor,
    strategy_descriptor));
  dag_builder->Build();
  dag_builder_dict_.insert({dag_builder->net_name(), dag_builder});
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::NetParameterForSingleDevice(
  const std::string& segment_name_in_data_path,
  const std::string& net_name_in_model_update_path,
  NetParameter* net_parameter) {
  const std::string net_name_in_data_path = data_path_->net_name();
  const std::string segment_name_in_model_update_path = model_update_layer_name_;
  const std::string gradient_name = "out";
  const std::string old_weight_name = "old_weight";
  const std::string weight_name = "weight";

  // Create the proto for an UpdateProto layer and setup its properties.
  // Should we gather all the blobs needing updates as a single vector or treat
  // them separately?
  net_parameter->set_name(net_name_in_model_update_path);
  auto update_layer = net_parameter->add_layer();
  update_layer->set_name(model_update_layer_name_);
  update_layer->set_type(model_update_layer_type_);

  auto update_proto = update_layer->mutable_modelupdate_proto();
  update_proto->set_gradient(gradient_name);
  update_proto->set_old_weight(old_weight_name);
  update_proto->set_weight(weight_name);

  PathSharingDescriptor model_diff_sharing_desc {
    {
      PathSharingRole::kProducer,
      PathType::kDataPath,
      net_name_in_data_path,
      segment_name_in_data_path,
      RegisterType::kModelDiffType,
      TaskDirection::kBackward,
      TaskPlaceholder::kNo,
      RegisterOwner::kYes
    },
    {
      PathSharingRole::kConsumer,
      PathType::kModelUpdatePath,
      net_name_in_model_update_path,
      segment_name_in_model_update_path,
      RegisterType::kDataType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_diff_sharing_desc);

  PathSharingDescriptor model_sharing_desc {
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
      PathType::kDataPath,
      net_name_in_data_path,
      segment_name_in_data_path,
      RegisterType::kModelType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_sharing_desc);
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::StrategyForSingleDevice(
  const std::string& segment_name_in_data_path, Strategy* strategy) {
  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();
  auto device_set
    = segment_dag_of_data_path->DeviceSetOfSegment(segment_name_in_data_path);
  int32_t device_set_begin = device_set.front();
  int32_t device_set_end = device_set.back();

  auto placement_group = strategy->add_placement_group();
  placement_group->set_name(model_update_layer_name_);
  placement_group->mutable_layer_set()->add_name(model_update_layer_name_);
  placement_group->set_parallel_policy(kNaiveParallelOnSingleDevice);

  auto device_group = new oneflow::DeviceGroup();
  device_group->set_begin(device_set_begin);
  device_group->set_end(device_set_end);
  placement_group->set_allocated_device_group(device_group);
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::NetParameterForModelParallelOnMultipleDevices(
  const std::string& segment_name_in_data_path,
  const std::string& net_name_in_model_update_path,
  NetParameter* net_parameter) {
  // Create the proto for an UpdateProto layer and setup its properties.
  // Should we gather all the blobs needing updates as a single vector or treat
  // them separately?
  const std::string net_name_in_data_path = data_path_->net_name();
  const std::string segment_name_in_model_update_path = model_update_layer_name_;
  const std::string gradient_name = "out";
  const std::string old_weight_name = "old_weight";
  const std::string weight_name = "weight";

  net_parameter->set_name(net_name_in_model_update_path);

  auto layer = net_parameter->add_layer();
  layer->set_name(model_update_layer_name_);
  layer->set_type(model_update_layer_type_);
  auto update_proto = layer->mutable_modelupdate_proto();
  update_proto->set_gradient(gradient_name);
  update_proto->set_old_weight(old_weight_name);
  update_proto->set_weight(weight_name);

  PathSharingDescriptor model_diff_sharing_desc {
    {
      PathSharingRole::kProducer,
      PathType::kDataPath,
      net_name_in_data_path,
      segment_name_in_data_path,
      RegisterType::kModelDiffType,
      TaskDirection::kBackward,
      TaskPlaceholder::kNo,
      RegisterOwner::kYes
    },
    {
      PathSharingRole::kConsumer,
      PathType::kModelUpdatePath,
      net_name_in_model_update_path,
      segment_name_in_model_update_path,
      RegisterType::kDataType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_diff_sharing_desc);

  PathSharingDescriptor model_sharing_desc {
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
      PathType::kDataPath,
      net_name_in_data_path,
      segment_name_in_data_path,
      RegisterType::kModelType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_sharing_desc);
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::StrategyForModelParallelOnMultipleDevices(
  const std::string& segment_name_in_data_path, Strategy* strategy) {
  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();
  auto device_set
    = segment_dag_of_data_path->DeviceSetOfSegment(segment_name_in_data_path);
  int32_t device_set_begin = device_set.front();
  int32_t device_set_end = device_set.back();

  auto placement_group = strategy->add_placement_group();
  placement_group->set_name(model_update_layer_name_);
  placement_group->mutable_layer_set()->add_name(model_update_layer_name_);
  placement_group->set_parallel_policy(kModelParallelOnMultipleDevices);
  auto device_group = new oneflow::DeviceGroup();
  device_group->set_begin(device_set_begin);
  device_group->set_end(device_set_end);
  placement_group->set_allocated_device_group(device_group);
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::NetParameterForDataParallelOnMultipleDevice(
  const std::string& segment_name_in_data_path,
  const std::string& net_name_in_model_update_path,
  NetParameter* net_parameter) {
  const std::string net_name_in_data_path = data_path_->net_name();
  const std::string placeholder_segment_name = placeholder_layer_name_;
  const std::string model_update_segment_name = model_update_layer_name_;
  const std::string empty_in = "in";
  const std::string gradient_name = "out";
  const std::string old_weight_name = "old_weight";
  const std::string weight_name = "weight";

  net_parameter->set_name(net_name_in_model_update_path);

  auto placeholder_layer = net_parameter->add_layer();
  placeholder_layer->set_name(placeholder_layer_name_);
  placeholder_layer->set_type(placeholder_layer_type_);
  auto placeholder_proto = placeholder_layer->mutable_placeholder_proto();
  placeholder_proto->set_in(empty_in);
  placeholder_proto->set_out(gradient_name);

  // Add model_update_layer
  auto model_update_layer = net_parameter->add_layer();
  model_update_layer->set_name(model_update_layer_name_);
  model_update_layer->set_type(model_update_layer_type_);
  auto model_update_proto = model_update_layer->mutable_modelupdate_proto();
  model_update_proto->set_gradient(
    strings::Join({ placeholder_layer_name_, gradient_name }, "/"));
  model_update_proto->set_old_weight(old_weight_name);
  model_update_proto->set_weight(weight_name);

  PathSharingDescriptor model_diff_sharing_desc {
    {
      PathSharingRole::kProducer,
      PathType::kDataPath,
      net_name_in_data_path,
      segment_name_in_data_path,
      RegisterType::kModelDiffType,
      TaskDirection::kBackward,
      TaskPlaceholder::kNo,
      RegisterOwner::kYes
    },
    {
      PathSharingRole::kConsumer,
      PathType::kModelUpdatePath,
      net_name_in_model_update_path,
      placeholder_segment_name,
      RegisterType::kDataType,
      TaskDirection::kForward,
      TaskPlaceholder::kYes,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_diff_sharing_desc);

  PathSharingDescriptor model_sharing_desc {
    {
      PathSharingRole::kProducer,
      PathType::kModelUpdatePath,
      net_name_in_model_update_path,
      model_update_segment_name,
      RegisterType::kDataType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kYes
    },
    {
      PathSharingRole::kConsumer,
      PathType::kDataPath,
      net_name_in_data_path,
      segment_name_in_data_path,
      RegisterType::kModelType,
      TaskDirection::kForward,
      TaskPlaceholder::kNo,
      RegisterOwner::kNo
    },
  };
  path_manager_->AddPathSharing(model_sharing_desc);
}

template <typename Dtype>
void ModelUpdatePath<Dtype>::StrategyForDataParallelOnMultipleDevice(
  const std::string& segment_name_in_data_path, Strategy* strategy) {
  auto segment_dag_of_data_path = dag_builder_of_data_path()->segment_dag();
  auto device_set
    = segment_dag_of_data_path->DeviceSetOfSegment(segment_name_in_data_path);
  int32_t device_set_begin = device_set.front();
  int32_t device_set_end = device_set.back();
  auto placeholder_device_group = new oneflow::DeviceGroup();
  placeholder_device_group->set_begin(device_set_begin);
  placeholder_device_group->set_end(device_set_end);

  auto placeholder_placement_group = strategy->add_placement_group();
  placeholder_placement_group->set_name(placeholder_layer_name_);
  placeholder_placement_group->mutable_layer_set()->add_name(placeholder_layer_name_);
  placeholder_placement_group->set_parallel_policy(kDataParallelOnMultipleDevices);
  placeholder_placement_group->set_allocated_device_group(placeholder_device_group);

  auto update_device_group = new oneflow::DeviceGroup();
  update_device_group->set_begin(device_set_begin);
  update_device_group->set_end(device_set_end);

  auto update_placement_group = strategy->add_placement_group();
  update_placement_group->set_name(model_update_layer_name_);
  update_placement_group->set_parallel_policy(kModelParallelOnMultipleDevices);
  update_placement_group->mutable_layer_set()->add_name(model_update_layer_name_);
  update_placement_group->set_allocated_device_group(update_device_group);
}

INSTANTIATE_CLASS(ModelUpdatePath);
}  // namespace oneflow
