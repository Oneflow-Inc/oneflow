#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

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
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;

  std::string dtbn2lbn(const std::string& data_tmp_blob_name) const {
    return op_name_ + "/" + data_tmp_blob_name;
  }
  virtual std::string ibn2lbn(const std::string& input_blob_name) const;
  virtual std::string obn2lbn(const std::string& output_blob_name) const;
  virtual std::string idbn2lbn(const std::string& input_diff_blob_name) const;
  virtual std::string odbn2lbn(const std::string& output_diff_blob_name) const;
  virtual std::string mtbn2lbn(const std::string& model_tmp_blob_name) const;
  virtual std::string mbn2lbn(const std::string& model_blob_name) const;
  virtual std::string mdbn2lbn(const std::string& model_diff_blob_name) const;
  
  virtual void Init(const OperatorConf& op_conf) = 0;
  virtual bool IsElemWise() const = 0;
  
  // Getters
  const std::string& op_name() const {
    return op_name_;
  }
  std::string GetValueFromPbOpConf(const std::string& k) const {
    return GetValueFromPbMessage(*pb_op_conf_, k);
  }

  #define DEFINE_BLOB_NAMES_GETTER(getter_name) \
  const std::vector<std::string>& getter_name() const { \
    return getter_name##_; \
  }

  DEFINE_BLOB_NAMES_GETTER(data_tmp_blob_names);
  DEFINE_BLOB_NAMES_GETTER(input_blob_names);
  DEFINE_BLOB_NAMES_GETTER(input_diff_blob_names);
  DEFINE_BLOB_NAMES_GETTER(output_blob_names);
  DEFINE_BLOB_NAMES_GETTER(output_diff_blob_names);
  DEFINE_BLOB_NAMES_GETTER(model_blob_names);
  DEFINE_BLOB_NAMES_GETTER(model_diff_blob_names);
  DEFINE_BLOB_NAMES_GETTER(model_tmp_blob_names);
  
  #undef DEFINE_BLOB_NAMES_GETTER

 protected:
  std::string& mut_op_name() {
    return op_name_;
  }
  std::unique_ptr<PbMessage>& mut_pb_op_conf() {
    return pb_op_conf_;
  }
  // register data blobs
  void RegisterDataTmpBlobName(const std::string& dtbn) {
    data_tmp_blob_names_.push_back(dtbn);
  }
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
  std::vector<std::string> data_tmp_blob_names_;
  
  std::vector<std::string> input_blob_names_;
  std::vector<std::string> input_diff_blob_names_;
  std::vector<std::string> output_blob_names_;
  std::vector<std::string> output_diff_blob_names_;
  
  std::vector<std::string> model_blob_names_;
  std::vector<std::string> model_diff_blob_names_;
  std::vector<std::string> model_tmp_blob_names_;

};

class UserOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOperator);
  UserOperator() = default;
  virtual ~UserOperator() = default;

  std::string ibn2lbn(const std::string& input_blob_name) const override {
    return GetValueFromPbOpConf(input_blob_name);
  }
  std::string obn2lbn(const std::string& output_blob_name) const override {
    return op_name() + "/" + GetValueFromPbOpConf(output_blob_name);
  }
  std::string idbn2lbn(const std::string& input_diff_blob_name) const override {
    return ibn2lbn(GenUnDiffBlobName(input_diff_blob_name));
  }
  std::string odbn2lbn(const std::string& output_diff_blob_name) const override {
    return obn2lbn(GenUnDiffBlobName(output_diff_blob_name));
  }
  std::string mtbn2lbn(const std::string& model_tmp_blob_name) const override {
    return op_name() + "/" + model_tmp_blob_name;
  }
  std::string mbn2lbn(const std::string& model_blob_name) const override {
    return op_name() + "/" + model_blob_name;
  }
  std::string mdbn2lbn(const std::string& model_diff_blob_name) const override {
    return mbn2lbn(GenUnDiffBlobName(model_diff_blob_name));
  }

};

class SysOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SysOperator);
  SysOperator() = default;
  virtual ~SysOperator() = default;
  
  #define SET_UNEXPECTED(func_name) \
  std::string func_name(const std::string&) const override { \
    UNEXPECTED_RUN(); \
  }
  
  SET_UNEXPECTED(ibn2lbn);
  SET_UNEXPECTED(obn2lbn);
  SET_UNEXPECTED(idbn2lbn);
  SET_UNEXPECTED(odbn2lbn);
  SET_UNEXPECTED(mtbn2lbn);
  SET_UNEXPECTED(mbn2lbn);
  SET_UNEXPECTED(mdbn2lbn);

  #undef SET_UNEXPECTED
  
  bool IsElemWise() const override { return false; }

 private:
};


} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
