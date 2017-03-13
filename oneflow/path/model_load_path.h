#ifndef _PATH_MODEL_LOAD_PATH_H_
#define _PATH_MODEL_LOAD_PATH_H_
#include <memory>
#include "path/base_path.h"

namespace caffe {
template <typename Dtype>
class DataPath;

template <typename Dtype>
class OpNode;

class SegmentMeta;

class NetParameter;

class Strategy;

template <typename Dtype>
class ModelLoadPath : public BasePath<Dtype> {
public:
  ModelLoadPath(std::shared_ptr<DataPath<Dtype>> data_path,
    PathManager<Dtype>* path_manager);
  virtual ~ModelLoadPath() {}

  ModelLoadPath(const ModelLoadPath& other) = delete;
  ModelLoadPath& operator=(const ModelLoadPath& other) = delete;

  void Build() override;
  void Setup() override;

private:
  int32_t index_;
  std::shared_ptr<DataPath<Dtype>> data_path_;

  std::shared_ptr<DagBuilder<Dtype>> dag_builder_of_data_path() const;

  void BuildModelLoadDagsForSegment(
    const std::string& segment_name_in_data_path);
  void NetParameterForModelLoadPath(
    const std::string& segment_name_in_data_path,
    const std::string& net_name_in_model_load_path,
    NetParameter* net_parameter);
  void StrategyForModelLoadPath(const std::string& segment_name_in_data_path,
    Strategy* strategy);

  void SetLoadProto(const std::string& segment_name_in_data_path,
    caffe::LoadPartialModelProto* loadpartialmodel_proto);

  const std::string placeholder_layer_name_ = "placeholder";
  const std::string placeholder_type_name_ = "Placeholder";
  const std::string loadpartialmodel_layer_name_ = "loadpartialmodel";
  const std::string loadpartialmodel_type_name_ = "LoadPartialModel";
};
}  // namespace caffe
#endif  // _PATH_MODEL_LOAD_PATH_H_
