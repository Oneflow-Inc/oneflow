#ifndef ONEFLOW_LAYER_BASE_LAYER_H_
#define ONEFLOW_LAYER_BASE_LAYER_H_

#include "common/shape.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(BlobDescSet);
  BlobDescSet() = default;
  virtual ~BlobDescSet() = default;

  void init() {
    name_to_pptr_.clear();
  }

 private:
  std::unordered_map<std:string, BlobDescriptor**> name_to_pptr_;

};

class DataBlobDescSet final : public BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(DataBlobDescSet);
  DataParam() = default;
  ~DataBlobDescSet() = default;

  void Init() {
    BlobDescSet::Init();
    input_blob_names_.clear();
    input_diff_blob_names_.clear();
    output_blob_names_.clear();
    output_diff_blob_names_.clear();
    data_tmp_blob_names_.clear();
  }

  const std::vector<std::string>& input_blob_names() const {
    return input_blob_names_;
  }
  const std::vector<std::string>& input_diff_blob_names() const {
    return input_diff_blob_names_;
  }
  const std::vector<std::string>& output_blob_names() const {
    return output_blob_names_;
  }
  const std::vector<std::string>& output_diff_blob_names() const {
    return output_diff_blob_names_;
  }
  const std::vector<std::string>& data_tmp_blob_names() const {
    return data_tmp_blob_names_;
  }

 private:
  std::vector<std::string> input_blob_names_;
  std::vector<std::string> input_diff_blob_names_;
  std::vector<std::string> output_blob_names_;
  std::vector<std::string> output_diff_blob_names_;
  std::vector<std::string> data_tmp_blob_names_;
};

class ModelBlobDescSet final : public BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ModelBlobDescSet);
  ModelBlobDescSet() = default;
  ~ModelBlobDescSet() = default;

  void Init() {
    BlobDescSet::init();
    model_blob_names_.clear();
    model_diff_blob_names_.clear();
    model_tmp_blob_names_.clear();
  }

  const std::vector<std::string>& model_blob_names() const {
    return model_blob_names_;
  }
  const std::vector<std::string>& model_diff_blob_names() const {
    return model_diff_blob_names_;
  }
  const std::vector<std::string>& model_tmp_blob_names() const {
    return model_tmp_blob_names_;
  }

 private:
  std::vector<std::string> model_blob_names_;
  std::vector<std::string> model_diff_blob_names_;
  std::vector<std::string> model_tmp_blob_names_;
};

class BaseLayerDesc {
 public:
  DISALLOW_COPY_AND_MOVE(BaseLayerDesc);
  BaseLayerDesc() = default;
  virtual ~BaseLayer() = default;
  
  virtual void Init(const LayerConf& layer_conf) = 0;
  virtual const DataBlobDescSet* data_blob_desc_set() = 0;
  virtual const ModelBlobDescSet* model_blob_desc_set() = 0;

  const std::string& layer_name() const { return layer_name_; }

 private:
  std::string layer_name_;

};

} // namespace oneflow

#endif // ONEFLOW_LAYER_BASE_LAYER_H_
