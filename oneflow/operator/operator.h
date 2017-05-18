#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <string>
#include "operator/op_conf.pb.h"
#include "conf/strategy.pb.h"
#include "common/shape.h"
#include "common/protobuf.h"
#include "common/util.h"
#include "operator/operator.pb.h"

namespace oneflow {

// bn  : blob name
// lbn : logical blob name

class Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;

  //
  virtual void InitFromOpConf(const OperatorConf& op_conf) = 0;
  virtual bool IsElemWise() const { return false; }
  virtual bool IsLossOp() const { return false; }
  
  // this <-> OpProto
  void InitFromProto(const OperatorProto& operatorproto);
  void ToProto(OperatorProto* ret) const;
  
  // bn_in_op <-> lbn
  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const;
  void ModifyLbn4BnInOp(const std::string& bn_in_op, const std::string& lbn);
  
  // Getters
  const std::string& op_name() const { return op_conf_.name(); }
  const OperatorConf& op_conf() const { return op_conf_; }
  virtual const PbMessage& GetSpecialConf() const = 0;

  #define DEFINE_GET_VAL_FROM_SPECIAL_CONF(ret_type, func_name) \
  ret_type Get##func_name##FromSpecialConf( \
      const std::string& field_name) const { \
    const PbMessage& special_conf = GetSpecialConf(); \
    return Get##func_name##FromPbMessage(special_conf, field_name); \
  }

  DEFINE_GET_VAL_FROM_SPECIAL_CONF(std::string, String);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(int32_t, Int32);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(uint32_t, UInt32);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(int64_t, Int64);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(uint64_t, UInt64);

  #undef DEFINE_GET_VAL_FROM_SPECIAL_CONF
  
  const std::string& SoleIbn() const;
  const std::string& SoleIdbn() const;
  const std::string& SoleObn() const;
  const std::string& SoleOdbn() const;
  const std::string& SoleDtbn() const;

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

  // Read: shape of input_blobs
  // Write: shape of output_blobs, model_blobs, data_tmp_blobs, model_tmp_blobs
  virtual void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const = 0;

 protected:
  virtual std::string ibn2lbn(const std::string& input_bn) const = 0;
  virtual std::string obn2lbn(const std::string& output_bn) const = 0;
  virtual std::string mtbn2lbn(const std::string& model_tmp_bn) const = 0;
  virtual std::string mbn2lbn(const std::string& model_bn) const = 0;

  OperatorConf& mut_op_conf() { return op_conf_; }
  
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
  std::string dtbn2lbn(const std::string& data_tmp_bn) const;
  
  OperatorConf op_conf_;
  HashMap<std::string, std::string> bn_in_op2lbn_;

  // blob name in op
  std::vector<std::string> data_tmp_bns_;
  std::vector<std::string> input_bns_;
  std::vector<std::string> input_diff_bns_;
  std::vector<std::string> output_bns_;
  std::vector<std::string> output_diff_bns_;

  std::vector<std::string> model_bns_;
  std::vector<std::string> model_diff_bns_;
  std::vector<std::string> model_tmp_bns_;

};

class UserOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOperator);
  UserOperator() = default;
  virtual ~UserOperator() = default;
 
 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
  std::string mtbn2lbn(const std::string& model_tmp_bn) const override;
  std::string mbn2lbn(const std::string& model_bn) const override;

};

class SysOperator : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SysOperator);
  SysOperator() = default;
  virtual ~SysOperator() = default;
  
  virtual void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const override {
    UNEXPECTED_RUN();
  }

 private:
  #define SET_INSIGNIFICANT(func_name) \
  virtual std::string func_name(const std::string&) const override { \
    LOG(FATAL) << #func_name << " is insignificant for " \
               << typeid(*this).name(); \
  }
  
  SET_INSIGNIFICANT(ibn2lbn);
  SET_INSIGNIFICANT(obn2lbn);
  SET_INSIGNIFICANT(mtbn2lbn);
  SET_INSIGNIFICANT(mbn2lbn);

  #undef SET_INSIGNIFICANT
  
};

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
