#ifndef _PATH_MODEL_STORE_PATH_H_
#define _PATH_MODEL_STORE_PATH_H_
#include <memory>
#include <string>
#include <vector>
#include "oneflow.pb.h"
#include "proto_io.h"
#include "path/base_path.h"
#include "dag/node_meta.h"
#include "dag/dag_node.h"

namespace oneflow {
template <typename Dtype>
class DataPath;
class NetDescriptor;
class StrategyDescriptor;
template <typename Dtype>
class LogicalDag;

template <typename Dtype>
class ModelStorePath : public BasePath<Dtype> {
public:
  ModelStorePath(std::shared_ptr<DataPath<Dtype>> data_path,
    PathManager<Dtype>* path_manager);
  virtual ~ModelStorePath() {}

  ModelStorePath(const ModelStorePath& other) = delete;
  ModelStorePath& operator=(const ModelStorePath& other) = delete;

  void Build() override;
  void Setup() override;

  bool is_train() const;

private:
  int32_t index_;
  int64_t seekpos_;
  std::shared_ptr<DataPath<Dtype>> data_path_;
  
  bool is_train_;

  std::shared_ptr<DagBuilder<Dtype>> dag_builder_of_data_path() const;

  void BuildModelStoreDagsForSegment(
    const std::string& segment_name_in_data_path);
  void NetParameterForModelStorePath(
    const std::string& segment_name_in_data_path,
    const std::string& net_name_in_model_store_path,
    NetParameter* net_parameter);
  void StrategyForModelStorePath(const std::string& segment_name_in_data_path,
    Strategy* strategy);

  void SetStoreProto(const std::string& segment_name_in_data_path,
    oneflow::StoreProto* store_proto);

  const std::string placeholder_layer_name_ = "placeholder";
  const std::string placeholder_type_name_ = "Placeholder";
  const std::string store_layer_name_ = "store";
  const std::string store_type_name_ = "Store";
};
}  // namespace oneflow
#endif  // _PATH_MODEL_STORE_PATH_H_
