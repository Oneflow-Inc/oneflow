#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_H_

#include "oneflow/core/common/preprocessor.h"
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

  virtual bool NeedExtraInDiffMemWhenBackward() const { return true; }
  virtual bool NeedOutWhenBackward() const { return true; }
  virtual bool IsLossOp() const { return false; }
  virtual bool IsPrintOp() const { return false; }
  virtual bool IsDecodeOp() const { return false; }
  virtual bool IsRecurrentOp() const { return false; }
  virtual bool IsCloneOp() const { return false; }
  virtual bool IsNormalizationOp() const { return false; }

  bool HasModelOrModelTmpBlob() const {
    return !model_bns_.empty() || !model_tmp_bns_.empty();
  }

  // bn_in_op <-> lbn
  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const;
  void ModifyLbn4BnInOp(const std::string& bn_in_op, const std::string& lbn);
  int8_t TryModifyLbn4BnInOp(const std::string& bn_in_op,
                             const std::string& lbn);

  // Getters
  const std::string& op_name() const { return op_conf_.name(); }
  bool UseCudnn() const { return op_conf_.use_cudnn_on_gpu(); }
  const OperatorConf& op_conf() const { return op_conf_; }
  virtual const PbMessage& GetCustomizedConf() const { UNIMPLEMENTED(); }
  virtual PbMessage* MutableCustomizedKernelConf(
      KernelConf* kernel_conf) const {
    UNIMPLEMENTED();
  }

#define DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF(ret_type, func_name)             \
  ret_type Get##func_name##FromCustomizedConf(const std::string& field_name) \
      const {                                                                \
    const PbMessage& customized_conf = GetCustomizedConf();                  \
    return Get##func_name##FromPbMessage(customized_conf, field_name);       \
  }

  OF_PP_FOR_EACH_TUPLE(DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF,
                       PROTOBUF_BASIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(
                           const PbMessage&, Message));

  template<typename T>
  const T& GetMsgFromCustomizedConf(const std::string& field_name) const {
    return static_cast<const T&>(GetMessageFromCustomizedConf(field_name));
  }

  template<typename T>
  const PbRf<T>& GetPbRfFromCustomizedConf(
      const std::string& field_name) const {
    return GetPbRfFromPbMessage<T>(GetCustomizedConf(), field_name);
  }
  template<typename T>
  const PbRpf<T>& GetPbRpfFromCustomizedConf(
      const std::string& field_name) const {
    return GetPbRpfFromPbMessage<T>(GetCustomizedConf(), field_name);
  }

#undef DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF

#define DEFINE_SET_VAL_IN_CUSTOMIZED_CONF(val_type, func_name)                 \
  void Set##func_name##InCustomizedConf(const std::string& field_name,         \
                                        val_type val) const {                  \
    const PbMessage& customized_conf = GetCustomizedConf();                    \
    PbMessage* customized_conf_ptr = &const_cast<PbMessage&>(customized_conf); \
    Set##func_name##InPbMessage(customized_conf_ptr, field_name, val);         \
  }

  OF_PP_FOR_EACH_TUPLE(DEFINE_SET_VAL_IN_CUSTOMIZED_CONF,
                       PROTOBUF_BASIC_DATA_TYPE_SEQ);

#undef DEFINE_SET_VAL_IN_CUSTOMIZED_CONF

#define DEFINE_SET_VAL_IN_CUSTOMIZED_KERNEL_CONF(val_type, func_name)         \
  void Set##func_name##InCustomizedKernelConf(                                \
      KernelConf* kernel_conf, const std::string& field_name, val_type val)   \
      const {                                                                 \
    PbMessage* customized_kernel_conf_ptr =                                   \
        MutableCustomizedKernelConf(kernel_conf);                             \
    Set##func_name##InPbMessage(customized_kernel_conf_ptr, field_name, val); \
  }

  OF_PP_FOR_EACH_TUPLE(DEFINE_SET_VAL_IN_CUSTOMIZED_KERNEL_CONF,
                       PROTOBUF_BASIC_DATA_TYPE_SEQ);

#undef DEFINE_SET_VAL_IN_CUSTOMIZED_KERNEL_CONF

#define DEFINE_ADD_VAL_TO_PBRF_IN_CUSTOMIZED_KERNEL_CONF(val_type, func_name) \
  void Add##func_name##ToPbRfInCustomizedKernelConf(                          \
      KernelConf* kernel_conf, const std::string& field_name, val_type val)   \
      const {                                                                 \
    PbMessage* customized_kernel_conf_ptr =                                   \
        MutableCustomizedKernelConf(kernel_conf);                             \
    Add##func_name##InPbRf(customized_kernel_conf_ptr, field_name, val);      \
  }

  OF_PP_FOR_EACH_TUPLE(DEFINE_ADD_VAL_TO_PBRF_IN_CUSTOMIZED_KERNEL_CONF,
                       PROTOBUF_BASIC_DATA_TYPE_SEQ);

#undef DEFINE_SET_VAL_IN_CUSTOMIZED_KERNEL_CONF

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
  DEFINE_BLOB_NAMES_GETTER(other_bns);

#undef DEFINE_BLOB_NAMES_GETTER

  // Read: shape of input_blobs
  // Write: shape of output_blobs, model_blobs, data_tmp_blobs, model_tmp_blobs
  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, DeviceType device_type) const;
  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;

  void FixParallelDesc(ParallelDesc* pr_desc) const;
  void FixLbnWhenShareModel(const std::string& shared_op_name);
  virtual int32_t ModelSplitAxis() const { return -1; }
  virtual int32_t MaxModelSplitNum() const { return -1; }
  void GenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      bool is_forward, DeviceType, const ParallelContext*, KernelConf*) const;

 protected:
  virtual void VirtualFixParallelDesc(ParallelDesc* pr_desc) const {}
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*) const {}

  virtual std::string ibn2lbn(const std::string& input_bn) const;
  virtual std::string obn2lbn(const std::string& output_bn) const;
  virtual std::string mtbn2lbn(const std::string& model_tmp_bn) const;
  virtual std::string mbn2lbn(const std::string& model_bn) const;
  virtual std::string otbn2lbn(const std::string& other_bn) const;

  OperatorConf& mut_op_conf() { return op_conf_; }

  // enroll data blobs
  void EnrollDataTmpBn(const std::string& dtbn);
  void EnrollInputBn(const std::string& ibn, bool has_diff);
  void EnrollInputBn(const std::string& ibn) { EnrollInputBn(ibn, true); }
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num,
                             bool has_diff);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, bool has_diff);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix);
  void EnrollOutputBn(const std::string& obn, bool has_diff);
  void EnrollOutputBn(const std::string& obn) { EnrollOutputBn(obn, true); }

  // enroll model blobs
  void EnrollModelBn(const std::string& mbn);
  void EnrollModelTmpBn(const std::string& mtbn);

  void EnrollOtherBn(const std::string& otbn);

  void StrFieldTolower(const std::string& field_name);

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

  std::vector<std::string> other_bns_;
};

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);
std::string GetOpNameFromLbn(const std::string& lbn);
std::string GetBnInOpFromLbn(const std::string& lbn);
std::pair<std::string, std::string> ParseLbn(const std::string& lbn);

void AddOpCreator(OperatorConf::OpTypeCase op_type_case,
                  std::function<Operator*(const OperatorConf&)> creator);
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

struct OpCreatorRegister {
  OpCreatorRegister(OperatorConf::OpTypeCase op_type_case,
                    std::function<Operator*(const OperatorConf&)> creator) {
    AddOpCreator(op_type_case, creator);
  }
};

#define REGISTER_OP_CREATOR(op_type_case, creator) \
  static OpCreatorRegister g_op_creator_register_var(op_type_case, creator);

void EraseEmptyBnInVec(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    PbRpf<std::string>* bns);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_H_
