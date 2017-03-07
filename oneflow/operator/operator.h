#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <unordered_map>
#include <string>
#include "blob/blob_descriptor.h"
#include "operator/op_conf.pb.h"
#include "common/proto_io.h"

namespace oneflow {

class BlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(BlobDescSet);
  BlobDescSet() = default;
  virtual ~BlobDescSet() = default;

  virtual void Init() {
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

  virtual void Init();

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

  virtual void Init();

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

class Operator {
 public:
  DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;
  
  const std::string& op_name() const {
    return op_name_;
  }
  const PbMessage& pb_op_conf() const {
    return *(pb_op_conf_.get());
  }
  const DataBlobDescSet& data_blob_desc_set() const {
    return *(data_blob_desc_set_.get());
  }
  const ModelBlobDescSet& model_blob_desc_set() const {
    return *(model_blob_desc_set_.get());
  }
  
  virtual void Init(const OperatorConf& op_conf) = 0;
  virtual bool IsElemWise() const = 0;
 
 protected:
  std::string& mutable_op_name() {
    return op_name_;
  }
  std::unique_ptr<PbMessage>& mutable_pb_op_conf() {
    return pb_op_conf_;
  }
  std::unique_ptr<DataBlobDescSet>& mutable_data_blob_desc_set() {
    return data_blob_desc_set_;
  }
  std::unique_ptr<ModelBlobDescSet>& mutable_model_blob_desc_set() {
    return model_blob_desc_set_;
  }
 
 private:
  std::string op_name_;
  std::unique_ptr<PbMessage> pb_op_conf_;
  std::unique_ptr<DataBlobDescSet> data_blob_desc_set_;
  std::unique_ptr<ModelBlobDescSet> model_blob_desc_set_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
