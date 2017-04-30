#ifndef ONEFLOW_OPERATOR_OP_H_
#define ONEFLOW_OPERATOR_OP_H_

#include <string>
#include "operator/op_conf.pb.h"
#include "job/strategy.pb.h"
#include "common/shape.h"
#include "common/proto_io.h"
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
  //
  virtual void InitFromOperatorProto(const OperatorProto& operatorproto);
  virtual OperatorProto ToOperatorProto();
  
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
  virtual const std::string& op_name() const { return op_conf_.OperatorConf::name(); }
  virtual const OperatorConf& op_conf() const { return op_conf_; }
  virtual std::string GetValueFromPbOpConf(const std::string& k) const = 0;
  
  const std::string& SoleIbn() const {
    CHECK_EQ(input_bns_.size(), 1);
    return *(input_bns_.begin());
  }
  const std::string& SoleObn() const {
    CHECK_EQ(output_bns_.size(), 1);
    return *(output_bns_.begin());
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

  // Read: shape of input_blobs
  // Write: shape of output_blobs, model_blobs, data_tmp_blobs, model_tmp_blobs
  virtual void InferShape4FwBlobs(
      const HashMap<std::string, Shape*>& bn_in_op2shape_ptr,
      ParallelPolicy policy,
      uint64_t parallel_id) const = 0;

 protected:
  OperatorConf& mut_op_conf() {
    return op_conf_;
  }
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

  OperatorConf op_conf_;

  HashMap<std::string, std::string> special_ibn2lbn_;

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
  
  #define SET_INSIGNIFICANT(func_name) \
  virtual std::string func_name(const std::string&) const override { \
    LOG(FATAL) << #func_name << " is insignificant for " \
               << typeid(*this).name(); \
  }
  
  SET_INSIGNIFICANT(normal_ibn2lbn);
  SET_INSIGNIFICANT(obn2lbn);
  SET_INSIGNIFICANT(mtbn2lbn);
  SET_INSIGNIFICANT(mbn2lbn);

  #undef SET_UNEXPECTED
  
  virtual void InferShape4FwBlobs(
      const HashMap<std::string, Shape*>& bn_in_op2shape_ptr,
      ParallelPolicy policy,
      uint64_t parallel_id) const override {
    UNEXPECTED_RUN();
  }
  
 private:
};

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_OP_H_
