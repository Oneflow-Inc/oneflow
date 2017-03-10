#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <unordered_map>
#include <string>
#include "operator/op_conf.pb.h"
#include "common/proto_io.h"
#include "common/util.h"

namespace oneflow {

struct DataBlobNameSet {
  std::vector<std::string> input_blob_names;
  std::vector<std::string> input_diff_blob_names;
  std::vector<std::string> output_blob_names;
  std::vector<std::string> output_diff_blob_names;
  std::vector<std::string> data_tmp_blob_names;
};

struct ModelBlobNameSet {
  std::vector<std::string> model_blob_names;
  std::vector<std::string> model_diff_blob_names;
  std::vector<std::string> model_tmp_blob_names;
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
    return *pb_op_conf_;
  }
  const DataBlobNameSet& data_blob_name_set() const {
    return data_blob_name_set_;
  }
  const ModelBlobNameSet& model_blob_name_set() const {
    return model_blob_name_set_;
  }

  virtual std::string ibn2lbn(const std::string& input_blob_name) const {
    return GetStringValueFromPbMessage(*pb_op_conf_, input_blob_name);
  }
  virtual std::string obn2lbn(const std::string& output_blob_name) const {
    return op_name_ + "/" + output_blob_name;
  }
  virtual std::string idbn2lbn(const std::string input_diff_blob_name) const {
    LOG(FATAL) << "TODO";
    return "";
  }
  virtual std::string odbn2lbn(const std::string output_diff_blob_name) const {
    LOG(FATAL) << "TODO";
    return "";
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
  DataBlobNameSet& mutable_data_blob_name_set() {
    return data_blob_name_set_;
  }
  ModelBlobNameSet& mutable_model_blob_name_set() {
    return model_blob_name_set_;
  }
 
 private:
  std::string op_name_;
  std::unique_ptr<PbMessage> pb_op_conf_;
  DataBlobNameSet data_blob_name_set_;
  ModelBlobNameSet model_blob_name_set_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
