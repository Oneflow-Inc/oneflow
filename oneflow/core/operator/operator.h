#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/job/keyword.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

struct OpContext {
  virtual ~OpContext() {}
};

class LogicalNode;

class Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator() = default;
  virtual ~Operator() = default;

  //
  void InitFromOpConf(const OperatorConf& op_conf);
  virtual void InitFromOpConf() = 0;
  virtual bool IsElemWiseOp() const { return false; }

  ActivationType GetActivationType() const;

  virtual LogicalNode* NewProperLogicalNode();

  virtual bool IsLossOp() const { return false; }
  virtual bool IsRecurrentOp() const { return false; }
  virtual bool IsEmbeddingLookupOp() const { return false; }
  virtual bool HasSameShapeBetweenInOut() const { return false; }

  bool NeedOutBlobWhenBackwardIf() const {
    return NeedOutBlobWhenBackward() || (GetActivationType() != ActivationType::kNone);
  }
  virtual bool NeedOutBlobWhenBackward() const { return true; }
  bool NeedInBlobWhenBackwardIf() const { return NeedInBlobWhenBackward(); }
  virtual bool NeedInBlobWhenBackward() const { return true; }

  // bn_in_op <-> lbi
  const LogicalBlobId& BnInOp2Lbi(const std::string& bn_in_op) const;
  LogicalBlobId* MutBnInOp2Lbi(const std::string& bn_in_op);

  // Getters
  const std::string& op_name() const { return op_conf().name(); }
  DeviceType device_type() const { return op_attribute_.op_conf().device_type(); }
  bool EnableCudnn() const { return op_conf().enable_cudnn(); }
  bool DevIsGpuAndEnableCudnn() const { return device_type() == DeviceType::kGPU && EnableCudnn(); }
  const OperatorConf& op_conf() const { return op_attribute_.op_conf(); }
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
    return static_cast<const T&>(GetValFromCustomizedConf<const PbMessage&>(field_name));
  }

  template<typename T>
  const PbRf<T>& GetPbRfFromCustomizedConf(const std::string& field_name) const {
    return GetPbRfFromPbMessage<T>(GetCustomizedConf(), field_name);
  }
  template<typename T>
  const PbRpf<T>& GetPbRpfFromCustomizedConf(const std::string& field_name) const {
    return GetPbRpfFromPbMessage<T>(GetCustomizedConf(), field_name);
  }

  const std::string& SoleIbn() const;
  const std::string& SoleIdbn() const;
  const std::string& SoleObn() const;
  const std::string& SoleOdbn() const;
  const std::string& SoleDtbn() const;
  const std::string& SoleFbbn() const;
  const std::string& SoleBbbn() const;

#define DEFINE_BLOB_NAMES_GETTER(getter_name)                                           \
  const PbRpf<std::string>& getter_name() const { return op_attribute_.getter_name(); } \
  PbRpf<std::string>* mut_##getter_name() { return op_attribute_.mutable_##getter_name(); }

  DEFINE_BLOB_NAMES_GETTER(data_tmp_bns);
  DEFINE_BLOB_NAMES_GETTER(fw_buf_bns);
  DEFINE_BLOB_NAMES_GETTER(bw_buf_bns);
  DEFINE_BLOB_NAMES_GETTER(input_bns);
  DEFINE_BLOB_NAMES_GETTER(input_diff_bns);
  DEFINE_BLOB_NAMES_GETTER(output_bns);
  DEFINE_BLOB_NAMES_GETTER(output_diff_bns);
  DEFINE_BLOB_NAMES_GETTER(model_bns);
  DEFINE_BLOB_NAMES_GETTER(model_diff_bns);
  DEFINE_BLOB_NAMES_GETTER(const_model_bns);
  DEFINE_BLOB_NAMES_GETTER(const_buf_bns);
  DEFINE_BLOB_NAMES_GETTER(forward_model_bns);

#undef DEFINE_BLOB_NAMES_GETTER

  // Read: shape of input_blobs
  // Write: shape of output_blobs, model_blobs, data_tmp_blobs, const_model_blobs, const_buf_blobs
  void InferBlobDescsIf(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                        const ParallelContext*, std::function<void(OpContext*)> EnrollOpCtx) const;
  virtual void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext*,
                              std::function<void(OpContext*)> EnrollOpCtx) const;
  virtual void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext*) const;
  void InferBwBufBlobDescsIf(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*, const OpContext*) const;
  virtual void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext*, const OpContext*) const;
  virtual void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext*) const {}
  virtual void InferDiffBlobDescsWithoutFwBlob(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const {
    UNIMPLEMENTED();
  }
  virtual void FixInDiffBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext*) const {}

  void FixParallelDesc(ParallelDesc* pr_desc) const;
  void FixLbiWhenShareModel(const std::string& shared_op_name);
  virtual int32_t ModelSplitAxis() const { return -1; }
  virtual int32_t MaxModelSplitNum() const { return -1; }
  void GenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                     bool is_forward, const ParallelContext*, KernelConf*, const OpContext*) const;

 protected:
  int64_t cudnn_buf_limit_byte() const;

  virtual PbMessage* MutableCustomizedKernelConf(KernelConf*) const {
    UNIMPLEMENTED();
    return nullptr;
  }
  template<typename T>
  void SetValInCustomizedConf(const std::string& field_name, const T& val) const {
    SetValInPbMessage<T>(&const_cast<PbMessage&>(GetCustomizedConf()), field_name, val);
  }

  template<typename T>
  void SetValInCustomizedKernelConf(KernelConf* kernel_conf, const std::string& field_name,
                                    const T& val) const {
    PbMessage* customized_conf = MutableCustomizedKernelConf(kernel_conf);
    SetValInPbMessage<T>(customized_conf, field_name, val);
  }

  template<typename T>
  T* MutableMsgInCustomizedKernelConf(KernelConf* kernel_conf,
                                      const std::string& field_name) const {
    PbMessage* customized_conf = MutableCustomizedKernelConf(kernel_conf);
    return static_cast<T*>(MutableMessageInPbMessage(customized_conf, field_name));
  }

  template<typename T>
  void AddValToPbRfInCustomizedKernelConf(KernelConf* kernel_conf, const std::string& field_name,
                                          const T& val) const {
    PbMessage* customized_conf = MutableCustomizedKernelConf(kernel_conf);
    AddValInPbRf<T>(customized_conf, field_name, val);
  }

  virtual void VirtualFixParallelDesc(ParallelDesc* pr_desc) const {}
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const;
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*) const {}

  virtual LogicalBlobId ibn2lbi(const std::string& input_bn) const;
  virtual LogicalBlobId obn2lbi(const std::string& output_bn) const;
  virtual LogicalBlobId cmbn2lbi(const std::string& const_model_bn) const;
  virtual LogicalBlobId cbbn2lbi(const std::string& const_buf_bn) const;
  virtual LogicalBlobId mbn2lbi(const std::string& model_bn) const;
  virtual LogicalBlobId fwmbn2lbi(const std::string& forward_model_bn) const;

  OperatorConf* mut_op_conf() { return op_attribute_.mutable_op_conf(); }

  // enroll data blobs
  void EnrollDataTmpBn(const std::string& dtbn);
  void EnrollFwBufBn(const std::string& fbbn);
  void EnrollBwBufBn(const std::string& bbbn);
  void EnrollInputBn(const std::string& ibn, bool has_diff);
  void EnrollInputBn(const std::string& ibn) { EnrollInputBn(ibn, true); }
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num, bool has_diff);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, bool has_diff);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix);
  void EnrollOutputBn(const std::string& obn, bool has_diff);
  void EnrollOutputBn(const std::string& obn) { EnrollOutputBn(obn, true); }
  void EnrollRepeatedOutputBn(const std::string& obn_prefix, int32_t num, bool has_diff);
  void EnrollRepeatedOutputBn(const std::string& obn_prefix, bool has_diff);
  void EnrollRepeatedOutputBn(const std::string& obn_prefix, int32_t num);
  void EnrollRepeatedOutputBn(const std::string& obn_prefix);

  // enroll model blobs
  void EnrollModelBn(const std::string& mbn);
  void EnrollModelDiffBn(const std::string& mdbn);
  void EnrollConstModelBn(const std::string& cmbn);

  void EnrollConstBufBn(const std::string& cbbn);

  void EnrollForwardModelBn(const std::string& fwmbn);

  void StrFieldTolower(const std::string& field_name);

 private:
  LogicalBlobId dtbn2lbi(const std::string& data_tmp_bn) const;
  LogicalBlobId fbbn2lbi(const std::string& fw_buf_bn) const { return dtbn2lbi(fw_buf_bn); }
  LogicalBlobId bbbn2lbi(const std::string& bw_buf_bn) const { return dtbn2lbi(bw_buf_bn); }
  void InferTotalInstanceNumDesc(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*,
                                 std::function<void(OpContext*)> EnrollOpCtx) const;
  std::string Bn2ConfName(const std::string& bn) const;
  PbMap<std::string, LogicalBlobId>* mut_bn_in_op2lbi() {
    return op_attribute_.mutable_bn_in_op2lbi();
  }

  OpAttribute op_attribute_;
};

std::string GenDiffBn(const std::string& bn);
std::string GenUnDiffBn(const std::string& diff_bn);
std::string GenRepeatedBn(const std::string& bn_prefix, int32_t idx);
std::pair<std::string, int32_t> GenUnRepeatedBn(const std::string& bn);

struct OnlyCpuSupportPredicator {
  OnlyCpuSupportPredicator(bool only_cpu) : only_cpu_(only_cpu) {}
  operator bool() { return only_cpu_; }

 private:
  bool only_cpu_;
};

#define REGISTER_OP(op_type_case, OpType)                                       \
  REGISTER_CLASS_CREATOR(op_type_case, OnlyCpuSupportPredicator,                \
                         ([] { return new OnlyCpuSupportPredicator(false); })); \
  REGISTER_CLASS_WITH_ARGS(op_type_case, Operator, OpType, const OperatorConf&)

#define REGISTER_CPU_OP(op_type_case, OpType)                                  \
  REGISTER_CLASS_CREATOR(op_type_case, OnlyCpuSupportPredicator,               \
                         ([] { return new OnlyCpuSupportPredicator(true); })); \
  extern "C" void Oneflow_OnlyCpuSupported_##OpType() {}                       \
  REGISTER_CLASS_WITH_ARGS(op_type_case, Operator, OpType, const OperatorConf&)

#define REGISTER_OP_CREATOR(op_type_case, creator)                              \
  REGISTER_CLASS_CREATOR(op_type_case, OnlyCpuSupportPredicator,                \
                         ([] { return new OnlyCpuSupportPredicator(false); })); \
  REGISTER_CLASS_CREATOR(op_type_case, Operator, creator, const OperatorConf&)

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf);

void EraseEmptyBnInVec(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                       PbRpf<std::string>* bns);

inline LogicalBlobId GenPackedLbi() {
  LogicalBlobId lbi;
  lbi.set_is_packed_id(true);
  return lbi;
}

inline LogicalBlobId GenLogicalBlobId(const std::string& lbn) {
  LogicalBlobId lbi;
  size_t pos = lbn.find('/');
  CHECK_NE(pos, std::string::npos);
  lbi.set_op_name(lbn.substr(0, pos));
  lbi.set_blob_name(lbn.substr(pos + 1));
  return lbi;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_H_
