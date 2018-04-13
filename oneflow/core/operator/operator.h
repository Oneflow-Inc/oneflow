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

struct OpContext {
  virtual ~OpContext() {}
};

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
  bool UseCudnn(DeviceType device_type) const;
  const OperatorConf& op_conf() const { return op_conf_; }
  virtual const PbMessage& GetCustomizedConf() const { UNIMPLEMENTED(); }

  bool HasFieldInCustomizedConf(const std::string& field_name) const {
    return HasFieldInPbMessage(GetCustomizedConf(), field_name);
  }

  template<typename T>
  T GetValFromCustomizedConf(const std::string& field_name) const {
    return GetValFromPbMessage<T>(GetCustomizedConf(), field_name);
  }

  int32_t GetEnumFromCustomizedConf(const std::string& field_name) const {
    return GetEnumFromPbMessage(GetCustomizedConf(), field_name);
  }

  template<typename T>
  const T& GetMsgFromCustomizedConf(const std::string& field_name) const {
    return static_cast<const T&>(
        GetValFromCustomizedConf<const PbMessage&>(field_name));
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
  DEFINE_BLOB_NAMES_GETTER(forward_model_bns);

#undef DEFINE_BLOB_NAMES_GETTER

  // Read: shape of input_blobs
  // Write: shape of output_blobs, model_blobs, data_tmp_blobs, model_tmp_blobs
  void InferBlobDescsIf(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*, DeviceType,
      std::function<void(OpContext*)> EnrollOpCtx) const;
  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*, DeviceType,
      std::function<void(OpContext*)> EnrollOpCtx) const;
  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*, DeviceType) const;
  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*) const;

  void FixParallelDesc(ParallelDesc* pr_desc) const;
  void FixLbnWhenShareModel(const std::string& shared_op_name);
  virtual int32_t ModelSplitAxis() const { return -1; }
  virtual int32_t MaxModelSplitNum() const { return -1; }
  void GenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      bool is_forward, DeviceType, const ParallelContext*, KernelConf*,
      const OpContext*) const;

 protected:
  virtual PbMessage* MutableCustomizedKernelConf(KernelConf*) const {
    UNIMPLEMENTED();
  }
  template<typename T>
  void SetValInCustomizedConf(const std::string& field_name,
                              const T& val) const {
    SetValInPbMessage<T>(&const_cast<PbMessage&>(GetCustomizedConf()),
                         field_name, val);
  }

  template<typename T>
  void SetValInCustomizedKernelConf(KernelConf* kernel_conf,
                                    const std::string& field_name,
                                    const T& val) const {
    PbMessage* customized_conf = MutableCustomizedKernelConf(kernel_conf);
    SetValInPbMessage<T>(customized_conf, field_name, val);
  }

  template<typename T>
  T* MutableMsgInCustomizedKernelConf(KernelConf* kernel_conf,
                                      const std::string& field_name) const {
    PbMessage* customized_conf = MutableCustomizedKernelConf(kernel_conf);
    return static_cast<T*>(
        MutableMessageInPbMessage(customized_conf, field_name));
  }

  template<typename T>
  void AddValToPbRfInCustomizedKernelConf(KernelConf* kernel_conf,
                                          const std::string& field_name,
                                          const T& val) const {
    PbMessage* customized_conf = MutableCustomizedKernelConf(kernel_conf);
    AddValInPbRf<T>(customized_conf, field_name, val);
  }

  virtual void VirtualFixParallelDesc(ParallelDesc* pr_desc) const {}
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*, const OpContext*) const;
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*) const {}

  virtual std::string ibn2lbn(const std::string& input_bn) const;
  virtual std::string obn2lbn(const std::string& output_bn) const;
  virtual std::string mtbn2lbn(const std::string& model_tmp_bn) const;
  virtual std::string mbn2lbn(const std::string& model_bn) const;
  virtual std::string fwmbn2lbn(const std::string& forward_model_bn) const;

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

  void EnrollForwardModelBn(const std::string& fwmbn);

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

  std::vector<std::string> forward_model_bns_;
};

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);
std::string GenUnCloneLbn(const std::string& clone_lbn);
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
