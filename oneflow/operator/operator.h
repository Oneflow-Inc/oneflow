#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <unordered_map>
#include <string>
#include "operator/op_conf.pb.h"
#include "common/proto_io.h"
#include "common/util.h"

namespace oneflow {

inline std::string GenDiffBlobName(const std::string& blob_name) {
  return blob_name + "_diff";
}
inline std::string GenUnDiffBlobName(const std::string& diff_blob_name) {
  CHECK_STREQ(diff_blob_name.substr(diff_blob_name.size() - 5).c_str(), "_diff");
  return diff_blob_name.substr(0, diff_blob_name.size() - 5);
}

class Operator {
 public:
  DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;

  // logical_blob_name
  virtual std::string ibn2lbn(const std::string& input_blob_name) const {
    return GetStringValueFromPbMessage(*pb_op_conf_, input_blob_name);
  }
  virtual std::string obn2lbn(const std::string& output_blob_name) const {
    return op_name_ + "/" + output_blob_name;
  }
  virtual std::string idbn2lbn(const std::string& input_diff_blob_name) const {
    return ibn2lbn(GenUnDiffBlobName(input_diff_blob_name));
  }
  virtual std::string odbn2lbn(const std::string& output_diff_blob_name) const {
    return obn2lbn(GenUnDiffBlobName(output_diff_blob_name));
  }

  std::string dtbn2lbn(const std::string& data_tmp_blob_name) const {
    return op_name_ + "/" + data_tmp_blob_name;
  }
  std::string mtbn2lbn(const std::string& model_tmp_blob_name) const {
    return op_name_ + "/" + model_tmp_blob_name;
  }
  std::string mbn2lbn(const std::string& model_blob_name) const {
    return op_name_ + "/" + model_blob_name;
  }
  std::string mdbn2lbn(const std::string& model_diff_blob_name) const {
    return op_name_ + "/" + model_diff_blob_name;
  }
  
  virtual void Init(const OperatorConf& op_conf) = 0;
  virtual bool IsElemWise() const = 0;
  
  // Getters
  const std::string& op_name() const {
    return op_name_;
  }

  #define DEFINE_BLOB_NAMES_GETTER(getter_name) \
  const std::vector<std::string>& getter_name() const { \
    return getter_name##_; \
  } \

  DEFINE_BLOB_NAMES_GETTER(input_blob_names);
  DEFINE_BLOB_NAMES_GETTER(input_diff_blob_names);
  DEFINE_BLOB_NAMES_GETTER(output_blob_names);
  DEFINE_BLOB_NAMES_GETTER(output_diff_blob_names);
  DEFINE_BLOB_NAMES_GETTER(data_tmp_blob_names);
  DEFINE_BLOB_NAMES_GETTER(model_blob_names);
  DEFINE_BLOB_NAMES_GETTER(model_diff_blob_names);
  DEFINE_BLOB_NAMES_GETTER(model_tmp_blob_names);

  #undef DEFINE_BLOB_NAMES_GETTER
 
 protected:
  std::string& mutable_op_name() {
    return op_name_;
  }
  std::unique_ptr<PbMessage>& mutable_pb_op_conf() {
    return pb_op_conf_;
  }

  // register data blobs
  void RegisterInputBlobName(const std::string& ibn) {
    input_blob_names_.push_back(ibn);
  }
  void RegisterInputDiffBlobName(const std::string& idbn) {
    input_diff_blob_names_.push_back(idbn);
  }
  void RegisterOutputBlobName(const std::string& obn) {
    output_blob_names_.push_back(obn);
  }
  void RegisterOutputDiffBlobName(const std::string& odbn) {
    output_diff_blob_names_.push_back(odbn);
  }
  void RegisterDataTmpBlobName(const std::string& dtbn) {
    data_tmp_blob_names_.push_back(dtbn);
  }

  // register model blobs
  void RegisterModelBlobName(const std::string& mbn) {
    model_blob_names_.push_back(mbn);
  }
  void RegisterModelDiffBlobName(const std::string& mdbn) {
    model_diff_blob_names_.push_back(mdbn);
  }
  void RegisterModelTmpBlobName(const std::string& mtbn) {
    model_tmp_blob_names_.push_back(mtbn);
  }
 
 private:
  std::string op_name_;
  std::unique_ptr<PbMessage> pb_op_conf_;
  
  std::vector<std::string> input_blob_names_;
  std::vector<std::string> input_diff_blob_names_;
  std::vector<std::string> output_blob_names_;
  std::vector<std::string> output_diff_blob_names_;
  std::vector<std::string> data_tmp_blob_names_;

  std::vector<std::string> model_blob_names_;
  std::vector<std::string> model_diff_blob_names_;
  std::vector<std::string> model_tmp_blob_names_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
