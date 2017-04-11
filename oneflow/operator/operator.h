#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <string>
#include "operator/op_conf.pb.h"
#include "common/proto_io.h"
#include "common/util.h"
#include "blob/blob_desc.h"

namespace oneflow {

// bn : blob name

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);

class Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;
  
  virtual void Init(const OperatorConf& op_conf) = 0;
  virtual bool IsElemWise() const = 0;

  // lbn : LogicalBlobName 
  std::string dtbn2lbn(const std::string& data_tmp_bn) const;
  virtual std::string ibn2lbn(const std::string& input_bn) const;
  virtual std::string obn2lbn(const std::string& output_bn) const;
  virtual std::string idbn2lbn(const std::string& input_diff_bn) const;
  virtual std::string odbn2lbn(const std::string& output_diff_bn) const;
  virtual std::string mtbn2lbn(const std::string& model_tmp_bn) const;
  virtual std::string mbn2lbn(const std::string& model_bn) const;
  virtual std::string mdbn2lbn(const std::string& model_diff_bn) const;
  
  // Getters
  const std::string& op_name() const { return op_name_; }
  std::string GetValueFromPbOpConf(const std::string& k) const {
    return GetValueFromPbMessage(*pb_op_conf_, k);
  }

  #define DEFINE_BLOB_NAMES_GETTER(getter_name) \
  const std::vector<std::string>& getter_name() const { \
    return getter_name##_; \
  }

  DEFINE_BLOB_NAMES_GETTER(data_tmp_bns);
  DEFINE_BLOB_NAMES_GETTER(input_bns);
  DEFINE_BLOB_NAMES_GETTER(input_diff_bns);
  DEFINE_BLOB_NAMES_GETTER(output_bns);
  DEFINE_BLOB_NAMES_GETTER(output_diff_bns);
  DEFINE_BLOB_NAMES_GETTER(model_bns);
  DEFINE_BLOB_NAMES_GETTER(model_diff_bns);
  DEFINE_BLOB_NAMES_GETTER(model_tmp_bns);
  
  #undef DEFINE_BLOB_NAMES_GETTER

  // Functions used to inference BlobDesc
  BlobDesc* GetBlobDescPtr(const std::string& bn_in_op) const;
  void SetBlobDescPtr(const std::string& bn_in_op, BlobDesc* ptr);
  void SetNull4AllBlobDescPtr();
  virtual void InferBlobDesc4ObAndDtbFromIb() const;
  virtual void InferBlobDesc4MbAndMtb() const;

 protected:
  std::string& mut_op_name() { return op_name_; }
  std::unique_ptr<PbMessage>& mut_pb_op_conf() { return pb_op_conf_; }
  
  // enroll data blobs
  void EnrollDataTmpBn(const std::string& dtbn);
  void EnrollInputBn(const std::string& ibn);
  void EnrollInputDiffBn(const std::string& idbn);
  void EnrollOutputBn(const std::string& obn);
  void EnrollOutputDiffBn(const std::string& odbn);
  
  // enroll model blobs
  void EnrollModelBn(const std::string& mbn);
  void EnrollModelDiffBn(const std::string& mdbn);
  void EnrollModelTmpBn(const std::string& mtbn);

 private:
  std::string op_name_;
  std::unique_ptr<PbMessage> pb_op_conf_;

  std::vector<std::string> data_tmp_bns_;
  
  std::vector<std::string> input_bns_;
  std::vector<std::string> input_diff_bns_;
  std::vector<std::string> output_bns_;
  std::vector<std::string> output_diff_bns_;
  
  std::vector<std::string> model_bns_;
  std::vector<std::string> model_diff_bns_;
  std::vector<std::string> model_tmp_bns_;

  HashMap<std::string, BlobDesc*> bn_in_op2blob_desc_ptr_;

};

class UserOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOperator);
  UserOperator() = default;
  virtual ~UserOperator() = default;

  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
  std::string idbn2lbn(const std::string& input_diff_bn) const override;
  std::string odbn2lbn(const std::string& output_diff_bn) const override;
  std::string mtbn2lbn(const std::string& model_tmp_bn) const override;
  std::string mbn2lbn(const std::string& model_bn) const override;
  std::string mdbn2lbn(const std::string& model_diff_bn) const override;

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
