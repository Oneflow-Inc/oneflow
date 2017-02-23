#ifndef ONEFLOW_LAYER_BASE_LAYER_DESC_H_
#define ONEFLOW_LAYER_BASE_LAYER_DESC_H_

#include <unordered_map>
#include <string>
#include "blob/blob_descriptor.h"
#include "layer/layer_conf.pb.h"

namespace oneflow {

class BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(BlobDescSet);
  BlobDescSet() = default;
  virtual ~BlobDescSet() = default;

  void Init() {
    name_to_pptr_.clear();
  }

 protected:
  void RegisterBlobNamePptrMap(const std::string& blob_name,
                               BlobDescriptor** pptr);

 private:
  std::unordered_map<std::string, BlobDescriptor**> name_to_pptr_;

};

class DataBlobDescSet : public BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(DataBlobDescSet);
  DataBlobDescSet() = default;
  virtual ~DataBlobDescSet() = default;

  void Init();

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

 protected:
  void RegisterInputBlobPptr(const std::string& blob_name,
                             BlobDescriptor** pptr) {
    input_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }
  void RegisterInputDiffBlobPptr(const std::string& blob_name,
                                 BlobDescriptor** pptr) {
    input_diff_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }
  void RegisterOutputBlobPptr(const std::string& blob_name,
                              BlobDescriptor** pptr) {
    output_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }
  void RegisterOutputDiffBlobPptr(const std::string& blob_name,
                                  BlobDescriptor** pptr) {
    output_diff_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }
  void RegisterDataTmpBlobPptr(const std::string& blob_name,
                               BlobDescriptor** pptr) {
    data_tmp_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }

 private:
  std::vector<std::string> input_blob_names_;
  std::vector<std::string> input_diff_blob_names_;
  std::vector<std::string> output_blob_names_;
  std::vector<std::string> output_diff_blob_names_;
  std::vector<std::string> data_tmp_blob_names_;
};

class ModelBlobDescSet : public BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ModelBlobDescSet);
  ModelBlobDescSet() = default;
  virtual ~ModelBlobDescSet() = default;

  void Init();

  const std::vector<std::string>& model_blob_names() const {
    return model_blob_names_;
  }
  const std::vector<std::string>& model_diff_blob_names() const {
    return model_diff_blob_names_;
  }
  const std::vector<std::string>& model_tmp_blob_names() const {
    return model_tmp_blob_names_;
  }

 protected:
  void RegisterModelBlobPptr(const std::string& blob_name,
                             BlobDescriptor** pptr) {
    model_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }
  void RegisterModelDiffBlobPptr(const std::string& blob_name,
                                 BlobDescriptor** pptr) {
    model_diff_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
  }
  void RegisterModelTmpBlobPptr(const std::string& blob_name,
                                BlobDescriptor** pptr) {
    model_tmp_blob_names_.push_back(blob_name);
    RegisterBlobNamePptrMap(blob_name, pptr);
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
  virtual ~BaseLayerDesc() = default;
  
  const std::string& layer_name() const {
    return layer_name_;
  }
  
  virtual void Init(const LayerConf& layer_conf) = 0;
  virtual const DataBlobDescSet* data_blob_desc_set() = 0;
  virtual const ModelBlobDescSet* model_blob_desc_set() = 0;
 
 protected:
  std::string& mutable_layer_name() {
    return layer_name_;
  }
 
 private:
  std::string layer_name_;

};

} // namespace oneflow

#endif // ONEFLOW_LAYER_BASE_LAYER_DESC_H_
