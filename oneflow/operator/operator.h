#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <string>
#include "operator/op_conf.pb.h"
#include "common/proto_io.h"
#include "common/util.h"
#include "blob/blob_desc.h"

namespace oneflow {

// bn  : blob name
// lbn : logical blob name

class Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;

  // 
  virtual void Init(const OperatorConf& op_conf) = 0;
  virtual bool IsElemWise() const { return false; }
  virtual bool IsLossOp() const { return false; }

  // bn_in_op2lbn
  std::string dtbn2lbn(const std::string& data_tmp_bn) const;
  std::string idbn2lbn(const std::string& input_diff_bn) const;
  std::string odbn2lbn(const std::string& output_diff_bn) const;
  std::string mdbn2lbn(const std::string& model_diff_bn) const;
  std::string ibn2lbn(const std::string& input_bn) const;

  virtual std::string obn2lbn(const std::string& output_bn) const = 0;
  virtual std::string mtbn2lbn(const std::string& model_tmp_bn) const = 0;
  virtual std::string mbn2lbn(const std::string& model_bn) const = 0;

  void AddSpecialIbn2Lbn(const std::string& ibn, const std::string& lbn) {
    CHECK(special_ibn2lbn_.emplace(ibn, lbn).second);
  }
  
  // Getters
  const std::string& op_name() const { return op_name_; }
  std::string GetValueFromPbOpConf(const std::string& k) const;

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
  DEFINE_BLOB_NAMES_GETTER(model_tmp_bns);
  
  #undef DEFINE_BLOB_NAMES_GETTER

  // Functions used to inference Shape
  Shape* GetShapePtr(const std::string& bn_in_op) const;
  void SetShapePtr(const std::string& bn_in_op, Shape* ptr) const;
  void SetNull4AllShapePtr() const;
  virtual void InferShape4ObAndDtbFromIb() const = 0;
  virtual void InferShape4Mtb() const = 0;

 protected:
  std::string& mut_op_name() { return op_name_; }
  std::unique_ptr<PbMessage>& mut_pb_op_conf() { return pb_op_conf_; }
  virtual std::string normal_ibn2lbn(const std::string& input_bn) const = 0;
  
  // enroll data blobs
  void EnrollDataTmpBn(const std::string& dtbn);
  void EnrollInputBn(const std::string& ibn, bool has_diff);
  void EnrollOutputBn(const std::string& obn, bool has_diff);

  void EnrollInputBn(const std::string& ibn) { EnrollInputBn(ibn, true); }
  void EnrollOutputBn(const std::string& obn) { EnrollOutputBn(obn, true); }
  
  // enroll model blobs
  void EnrollModelBn(const std::string& mbn);
  void EnrollModelTmpBn(const std::string& mtbn);

 private:
  void EnrollBn(std::vector<std::string>* bn_vec, const std::string& bn);

  std::string op_name_;
  std::unique_ptr<PbMessage> pb_op_conf_;

  std::unordered_map<std::string, std::string> special_ibn2lbn_;

  // blob name in op
  std::vector<std::string> data_tmp_bns_;
  std::vector<std::string> input_bns_;
  std::vector<std::string> input_diff_bns_;
  std::vector<std::string> output_bns_;
  std::vector<std::string> output_diff_bns_;
  std::vector<std::string> model_bns_;
  std::vector<std::string> model_diff_bns_;
  std::vector<std::string> model_tmp_bns_;

  mutable HashMap<std::string, Shape*> bn_in_op2shape_ptr_;

};

class UserOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOperator);
  UserOperator() = default;
  virtual ~UserOperator() = default;

  std::string normal_ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
  std::string mtbn2lbn(const std::string& model_tmp_bn) const override;
  std::string mbn2lbn(const std::string& model_bn) const override;

};

class SysOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SysOperator);
  SysOperator() = default;
  virtual ~SysOperator() = default;
  
  #define SET_UNEXPECTED(func_name) \
  virtual std::string func_name(const std::string&) const override { \
    UNEXPECTED_RUN(); \
  }
  
  SET_UNEXPECTED(normal_ibn2lbn);
  SET_UNEXPECTED(obn2lbn);
  SET_UNEXPECTED(mtbn2lbn);
  SET_UNEXPECTED(mbn2lbn);

  #undef SET_UNEXPECTED
  
  void InferShape4Mtb() const override { UNEXPECTED_RUN(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
