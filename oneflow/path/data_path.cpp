#include "path/data_path.h"
#include "common/common.h"
#include "dag/node_meta.h"
#include "dag/dag_builder.h"
#include "dag/copy_task_dag.h"
#include "dag/compute_task_dag.h"
#include "dag/boxing_task_dag.h"
#include "dag/net_task_dag.h"
#include "context/net_descriptor.h"
#include "dag/dag_builder.h"
#include "path/path_manager.h"

namespace oneflow {
template <typename Dtype>
DataPath<Dtype>::DataPath(const oneflow::NetParameter& net_param,
    std::shared_ptr<StrategyDescriptor> strategy_descriptor,
    PathManager<Dtype>* path_manager)
    : net_param_(net_param), strategy_descriptor_(strategy_descriptor),
    net_name_(net_param.name()), BasePath<Dtype>::BasePath(PathType::kDataPath, path_manager) {
}

template <typename Dtype>
void DataPath<Dtype>::Build() {
  std::shared_ptr<NetDescriptor> net_descriptor(new NetDescriptor(net_param_));
  std::shared_ptr<DagBuilder<Dtype>> dag_builder;
  dag_builder.reset(new DagBuilder<Dtype>(net_name_, this,
    net_descriptor, strategy_descriptor_));
  dag_builder->Build();
  dag_builder_dict_.insert({dag_builder->net_name(), dag_builder});
}

template <typename Dtype>
void DataPath<Dtype>::Setup() {
  auto dag_builder = GetDagBuilder(net_name_);
  dag_builder->Setup();
}

template <typename Dtype>
std::string DataPath<Dtype>::net_name() const {
  return net_name_;
}

template <typename Dtype>
bool DataPath<Dtype>::is_train() const {
  return is_train_;
}

INSTANTIATE_CLASS(DataPath);
}  // namespace oneflow
