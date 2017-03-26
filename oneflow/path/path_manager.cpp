#include "path/path_manager.h"
#include <memory>
#include "common/common.h"
#include "common/stl_util.h"
#include "context/one.h"
#include "context/config_parser.h"
#include "dag/task_dag.h"
#include "dag/compute_task_dag.h"
#include "path/data_path.h"
#include "path/model_update_path.h"
#include "path/model_load_path.h"
#include "path/model_store_path.h"

namespace oneflow {
template <typename Dtype>
void PathManager<Dtype>::Initialize(const SolverProto& param) {
  Build(param);
  Connect();
  Setup();
}

template <typename Dtype>
void PathManager<Dtype>::Build(const SolverProto& param) {
  auto config_parser = oneflow::TheOne<Dtype>::config_parser();
  oneflow::NetParameter net_param;
  oneflow::ReadProtoFromTextFileOrDie(param.train_net(), &net_param);

  // There exist dependencies among various path objects. The paths are created
  // in a particular order, such that if path A is depended on by path B, path A
  // should be created before that of path B. For example, here, the 
  // |model_update_path_| depends on |data_path_|.

  std::shared_ptr<DataPath<Dtype>> data_path;
  data_path.reset(
    new DataPath<Dtype>(net_param, config_parser->strategy_descriptor(), this));
  data_path->Build();
  path_dict_.insert({ PathType::kDataPath, data_path });

  std::shared_ptr<ModelUpdatePath<Dtype>> model_update_path;
  model_update_path.reset(
    new ModelUpdatePath<Dtype>(data_path, this));
  model_update_path->Build();
  path_dict_.insert({ PathType::kModelUpdatePath, model_update_path });

  std::shared_ptr<ModelLoadPath<Dtype>> model_load_path;
  model_load_path.reset(
    new ModelLoadPath<Dtype>(data_path, this));
  model_load_path->Build();
  path_dict_.insert({ PathType::kModelLoadPath, model_load_path });

  std::shared_ptr<ModelStorePath<Dtype>> model_store_path;
  model_store_path.reset(
    new ModelStorePath<Dtype>(data_path, this));
  model_store_path->Build();
  path_dict_.insert({ PathType::kModelStorePath, model_store_path });

  // TODO(jiyuan): add a diagnosing path to monitor the status of all the nodes
}

template <typename Dtype>
std::string PathManager<Dtype>::GetUpdateSegmentNameInModelUpdatePath() const {
  auto base_path = GetPath(PathType::kModelUpdatePath);
  auto model_update_path
    = std::dynamic_pointer_cast<ModelUpdatePath<Dtype>>(base_path);
  CHECK_NOTNULL(model_update_path.get());
  // NOTE(jiyuan): temporal solution, based on the following assumptions:
  // (1) the model update networks corresponding to different segments of data 
  // path share the same update_segment_name.
  // (2) the update_segment_name is the same as the update_layer_name.
  return model_update_path->model_update_layer_name();
}

template <typename Dtype>
void PathManager<Dtype>::AddPathSharing(
  const PathSharingDescriptor& path_sharing_desc) {
  sharing_descriptors_.push_back(path_sharing_desc);
}

template <typename Dtype>
void PathManager<Dtype>::CompleteOneRegisterInfo(
  const PathSharingDescriptor& sharing_desc) {
  auto& consumer_detail = sharing_desc.consumer_detail;
  auto& producer_detail = sharing_desc.producer_detail;

  auto consumer_path = GetPath(consumer_detail.path_type);
  auto producer_path = GetPath(producer_detail.path_type);

  auto consumer_device_ids = consumer_path->GetDeviceIDs(
    consumer_detail.net_name, consumer_detail.segment_name);
  auto producer_device_ids = producer_path->GetDeviceIDs(
    producer_detail.net_name, producer_detail.segment_name);
  CHECK(stl::VectorEqual(consumer_device_ids, producer_device_ids));

  CHECK(consumer_detail.register_owner != producer_detail.register_owner);

  RegisterType consumer_register_type = consumer_detail.register_type;
  RegisterType producer_register_type = producer_detail.register_type;

  for (auto device_id : consumer_device_ids) {
    auto consumer
      = consumer_path->GetCrossPathTaskDag(consumer_detail, device_id);
    auto producer
      = producer_path->GetCrossPathTaskDag(producer_detail, device_id);
    int32_t consumer_task_id = consumer->task_id();
    int32_t producer_task_id = producer->task_id();

    if (producer_detail.register_owner == RegisterOwner::kYes) {
      int64_t group_id
        = producer->GetProducedGroupIdByRegisterType(producer_register_type);
      RegisterInfo consumer_register_info
        = consumer->CompleteConsumedRegisterInfoCrossPath(consumer_register_type,
        group_id);
      RegisterInfo producer_register_info
        = producer->CompleteProducedRegisterInfoCrossPath(producer_register_type,
        consumer_register_info);

      producer->RegisterConsumer(consumer_task_id, group_id);
      consumer->AddConsumedGroupId(group_id);
    } else {
      // Get the produced register info from consumer;
      int64_t group_id
        = consumer->GetProducedGroupIdByRegisterType(consumer_register_type);
      // Replace the RegisterInfo in producer;
      RegisterInfo producer_register_info
        = producer->ReplaceProducedRegisterInfoCrossPath(producer_register_type,
        group_id);
      // Complete the shared RegisterInfo in the view of producer and consumer;
      RegisterInfo consumer_register_info
        = consumer->CompleteProducedRegisterInfoCrossPath(consumer_register_type,
        producer_register_info);
      // The producer treats the shared Register as a consumed one. After it
      // fills the Register, it will notify the owner (here the consumer) that
      // the status of the Register has changed.
      producer->AddConsumedGroupId(group_id);

      // The owner of shared Register (here the consumer) does not need to know
      // the 'active' side: producer. Therefore, we don't call |RegisterConsumer|.
      // If necessary, we could add a new method like |RegisterSharedProducer|.
      // consumer->RegisterConsumer(producer_task_id, group_id);
    }
  }
}

template <typename Dtype>
void PathManager<Dtype>::Connect() {
  for (auto& sharing_desc : sharing_descriptors_) {
    CompleteOneRegisterInfo(sharing_desc);
  }
}

template <typename Dtype>
void PathManager<Dtype>::Setup() {
  auto data_path = GetPath(PathType::kDataPath);
  data_path->Setup();

  auto model_update_path = GetPath(PathType::kModelUpdatePath);
  model_update_path->Setup();

  auto model_store_path = GetPath(PathType::kModelStorePath);
  model_store_path->Setup();

  auto model_load_path = GetPath(PathType::kModelLoadPath);
  model_load_path->Setup();
}

template <typename Dtype>
std::shared_ptr<BasePath<Dtype>> PathManager<Dtype>::GetPath(PathType type) const {
  auto path_it = path_dict_.find(type);
  CHECK(path_it != path_dict_.end());
  return path_it->second;
}

INSTANTIATE_CLASS(PathManager);
}  // namespace oneflow
