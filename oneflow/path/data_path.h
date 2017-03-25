#ifndef _PATH_DATA_PATH_H_
#define _PATH_DATA_PATH_H_
#include <memory>
#include "proto/oneflow.pb.h"
#include "proto/proto_io.h"
#include "path/base_path.h"
#include "common/task_type.h"

/*
User needs to specify the logical operations in a network (in the form of 
NetParameter) and the parallelism strategy (in the form of StrategyDescriptor).
*/
namespace oneflow {
class NetDescriptor;
class StrategyDescriptor;

template <typename Dtype>
class DataPath : public BasePath<Dtype> {
public:
  using BasePath<Dtype>::dag_builder_dict_;
  using BasePath<Dtype>::GetDagBuilder;


  DataPath(const oneflow::NetParameter& net_param,
    std::shared_ptr<StrategyDescriptor> strategy_descriptor,
    PathManager<Dtype>* path_manager);
  virtual ~DataPath() {}

  DataPath(const DataPath& other) = delete;
  DataPath& operator=(const DataPath& other) = delete;

  void Build() override;
  void Setup() override;

  // Currently, DataPath only supports one NetDescriptor, hence only one 
  // DagBuilder is supported. The name of the NetDescriptor acts the name of 
  // the DagBuilder object. To get the DagBuilder object, you could call
  // auto dag_builder_of_data_path = data_path.dag_builder(data_path.net_name());
  // where |data_path| is a DataPath object
  std::string net_name() const;
  bool is_train() const;

private:
  const std::string net_name_;
  const oneflow::NetParameter& net_param_;
  std::shared_ptr<StrategyDescriptor> strategy_descriptor_;

  bool is_train_{ true };
};
}  // namespace oneflow
#endif  // _PATH_DATA_PATH_H_
