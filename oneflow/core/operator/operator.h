#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/keyword.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

// bn  : blob name
// lbn : logical blob name

class Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;

  //
  void InitFromOpConf(const OperatorConf& op_conf);
  virtual void InitFromOpConf() = 0;
  virtual bool IsElemWiseOp() const { return false; }
  virtual bool IsLossOp() const { return false; }
  virtual bool IsRecordOp() const { return false; }
  virtual bool IsDataLoaderOp() const { return false; }

  // bn_in_op <-> lbn
  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const;
  void ModifyLbn4BnInOp(const std::string& bn_in_op, const std::string& lbn);
  int8_t TryModifyLbn4BnInOp(const std::string& bn_in_op,
                             const std::string& lbn);

  // Getters
  const std::string& op_name() const { return op_conf_.name(); }
  const OperatorConf& op_conf() const { return op_conf_; }
  virtual const PbMessage& GetSpecialConf() const = 0;

#define DEFINE_GET_VAL_FROM_SPECIAL_CONF(ret_type, func_name)             \
  ret_type Get##func_name##FromSpecialConf(const std::string& field_name) \
      const {                                                             \
    const PbMessage& special_conf = GetSpecialConf();                     \
    return Get##func_name##FromPbMessage(special_conf, field_name);       \
  }

  DEFINE_GET_VAL_FROM_SPECIAL_CONF(std::string, String);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(int32_t, Int32);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(int64_t, Int64);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(bool, Bool);
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(const PbMessage&, Message);

  template<typename T>
  const T& GetMsgFromSpecialConf(const std::string& field_name) const {
    return static_cast<const T&>(GetMessageFromSpecialConf(field_name));
  }

#undef DEFINE_GET_VAL_FROM_SPECIAL_CONF

  const std::string& SoleIbn() const;
  const std::string& SoleIdbn() const;
  const std::string& SoleObn() const;
  const std::string& SoleOdbn() const;
  const std::string& SoleDtbn() const;

#define DEFINE_BLOB_NAMES_GETTER(getter_name) \
  const std::vector<std::string>& getter_name() const { return getter_name##_; }

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
  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {
    TODO();
  }

  void FixParallelDesc(ParallelDesc* pr_desc) const;
  virtual int32_t ModelSplitAxis() const { return -1; }
  virtual int32_t MaxModelSplitNum() const { return -1; }
  void GenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      bool is_forward, const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const;

 protected:
  virtual void VirtualFixParallelDesc(ParallelDesc* pr_desc) const {}
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      bool is_forward, const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const {}

  virtual std::string ibn2lbn(const std::string& input_bn) const;
  virtual std::string obn2lbn(const std::string& output_bn) const;
  virtual std::string mtbn2lbn(const std::string& model_tmp_bn) const;
  virtual std::string mbn2lbn(const std::string& model_bn) const;

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

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);
std::string GetOpNameFromLbn(const std::string& lbn);
std::string GetBnInOpFromLbn(const std::string& lbn);
std::pair<std::string, std::string> ParseLbn(const std::string& lbn);

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*()> creator);

std::shared_ptr<Operator> ConstructOp(const OperatorConf&);

template<OperatorConf::OpTypeCase op_type_case, typename OpType>
struct OpRegister {
  OpRegister() {
    AddOpCreator(op_type_case, []() { return new OpType; });
  }
};

#define REGISTER_OP(OpTypeCase, OpType) \
  static OpRegister<OpTypeCase, OpType> g_##OpType##_register_var;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_H_
