#ifndef _PATH_MODEL_UPDATE_PATH_H_
#define _PATH_MODEL_UPDATE_PATH_H_
#include <memory>
#include <unordered_map>
#include "path/base_path.h"

// We generate a DagBuiler for each data path segment who needs a model update
// module. There are 3 cases: 
//
// (1) a segment on single device: a single dag_builder will generate a single
// model update task serving to the data path segment;
// (2) a segment on multiple devices in model-parallelism way: a single 
// dag_builder will generate a set of model update task serving each data
// path task corresponding to the data path segment;
// (3) a segment on multiple devices in data-parallelism way: a single
// dag_builder will generate a set of model update task serving all the
// data path tasks corresponding to the data path segment (essentially a PS)
namespace oneflow {
template <typename Dtype>
class DataPath;

template <typename Dtype>
class OpNode;

class SegmentMeta;

class NetParameter;

class Strategy;

template <typename Dtype>
class ModelUpdatePath : public BasePath<Dtype> {
public:
  using BasePath<Dtype>::dag_builder_dict_;
  using BasePath<Dtype>::path_manager_;



  ModelUpdatePath(std::shared_ptr<DataPath<Dtype>> data_path,
    PathManager<Dtype>* path_manager);
  virtual ~ModelUpdatePath() {}

  ModelUpdatePath(const ModelUpdatePath& other) = delete;
  ModelUpdatePath& operator=(const ModelUpdatePath& other) = delete;

  void Build() override;
  void Setup() override;

  bool is_train() const;
  std::string model_update_layer_name() const {
    return is_train_ ? model_update_layer_name_ : null_update_layer_name_;
  }

private:
  enum class ModelUpdateType{
    kOnSingleDevice = 0,
    kModelParallelismOnMultipleDevices,
    kDataParallelismOnMultipleDevices
  };

private:
  std::shared_ptr<DataPath<Dtype>> data_path_;

  bool is_train_;

  std::shared_ptr<DagBuilder<Dtype>> dag_builder_of_data_path() const;

  void BuildNullUpdateDagsForSegment(const std::string& segment_name);
  void BuildModelUpdateDagsForSegment(const std::string& segment_name);

  void CreateModelUpdateDags(const std::string& segment_name_in_data_path,
    ModelUpdateType model_update_type);

  // Used when the segment needs model update and is bound to a single device
  void NetParameterForSingleDevice(
    const std::string& segment_name_in_data_path,
    const std::string& net_name_in_model_update_path,
    NetParameter* net_parameter);
  void StrategyForSingleDevice(
    const std::string& segment_name_in_data_path,
    Strategy* strategy);

  // Used when the segment is with model-parallelism on multiple devices
  void NetParameterForModelParallelOnMultipleDevices(
    const std::string& segment_name_in_data_path,
    const std::string& net_name_in_model_update_path,
    NetParameter* net_parameter);
  void StrategyForModelParallelOnMultipleDevices(
    const std::string& segment_name_in_data_path,
    Strategy* strategy);

  // Used when the segment is with data-parallelism on multiple devices
  void NetParameterForDataParallelOnMultipleDevice(
    const std::string& segment_name_in_data_path,
    const std::string& net_name_in_model_update_path,
    NetParameter* net_paramemter);
  void StrategyForDataParallelOnMultipleDevice(
    const std::string& segment_name_in_data_path,
    Strategy* strategy);

  const std::string model_update_layer_type_ = "ModelUpdate";
  const std::string null_update_layer_type_ = "NullUpdate";
  const std::string model_update_layer_name_ = "model_update";
  const std::string null_update_layer_name_ = "null_update";
  const std::string placeholder_layer_type_ = "Placeholder";
  const std::string placeholder_layer_name_ = "placeholder";
};
}  // namespace oneflow
#endif  // _PATH_MODEL_UPDATE_PATH_H_
