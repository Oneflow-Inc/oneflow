/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_H_

#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/mirrored_parallel.pb.h"
#include "oneflow/core/operator/op_conf_util.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/job/parallel_distribution_infer_hint.h"

namespace oneflow {

class MirroredSigInferHint;
class OpNodeSignature;
class Scope;

class Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Operator);
  Operator();
  virtual ~Operator() = default;

  //
  void Init(const OperatorConf& op_conf);
  void Init(std::shared_ptr<const OperatorConf> op_conf);
  virtual void InitFromOpConf() = 0;

  // bn_in_op <-> lbi
  const LogicalBlobId& BnInOp2Lbi(const std::string& bn_in_op) const;

  // Getters
  const std::string& op_name() const { return op_conf().name(); }
  DeviceType device_type() const;
  const OperatorConf& op_conf() const;
  std::shared_ptr<const OperatorConf> shared_op_conf() const;
  const PbMessage& GetCustomizedConf() const {
    return GetMessageInPbMessage(op_conf(), op_conf().op_type_case());
  }

  template<typename T>
  T GetValFromCustomizedConf(const std::string& field_name) const {
    return GetValFromPbMessage<T>(GetCustomizedConf(), field_name);
  }

  template<typename T>
  const PbRpf<T>& GetPbRpfFromCustomizedConf(const std::string& field_name) const {
    return GetPbRpfFromPbMessage<T>(GetCustomizedConf(), field_name);
  }

  const std::string& SoleIbn() const;
  const std::string& SoleObn() const;
  const std::string& SoleTbn() const;
  Maybe<const std::string*> obn4lbi(const LogicalBlobId& lbi) const;

  const PbRpf<std::string>& input_bns() const;
  const PbRpf<std::string>& output_bns() const;
  const PbRpf<std::string>& tmp_bns() const;
  const PbRpf<std::string>& input_output_bns() const;

  Maybe<void> FillOpParallelDesc(const ParallelDesc& parallel_desc);
  Maybe<void> FillOpParallelDesc(std::shared_ptr<const ParallelDesc> parallel_desc);
  Maybe<const ParallelDesc> GetOpParallelDesc() const;

  Maybe<void> InferParallelSignatureIf();
  Maybe<const ParallelDesc> GetParallelDesc4BnInOp(const std::string& bn) const;

  Maybe<void> FillLogicalInBlobDesc(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp);
  Maybe<void> FillLogicalInBlobDesc(
      const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp);
  Maybe<void> FillLogicalInBlobDesc(
      const std::function<Maybe<const BlobDesc>(int32_t)>& BlobDesc4InputIndex);
  Maybe<const BlobDesc> GetLogicalBlobDesc4Ibn(const std::string& ibn) const;
  Maybe<const BlobDesc> GetLogicalBlobDesc4InputIndex(int32_t index) const;
  Maybe<void> FillLogicalOutBlobDesc(
      const std::function<const BlobDesc&(const std::string&)>& BlobDesc4BnInOp);
  Maybe<void> FillLogicalOutBlobDesc(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp);
  Maybe<const BlobDesc> GetLogicalBlobDesc4Obn(const std::string& obn) const;
  Maybe<const BlobDesc> GetLogicalBlobDesc4OutputIndex(int32_t index) const;
  Maybe<const BlobDesc*> GetLogicalBlobDescPtr4OutputIndex(int32_t index) const;
  Maybe<const BlobDesc> GetLogicalBlobDesc4BnInOp(const std::string& bn) const;
  Maybe<void> InferLogicalOutBlobDescsIf();
  virtual Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const = 0;

  // Read: shape of input_blobs
  // Write: shape of output_blobs
  Maybe<void> InferBlobDescsIf(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext*, const JobDesc* job_desc) const;

  Maybe<void> InferOutBlobDescsIf(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext*) const;

  Maybe<void> InferInternalBlobDescsIf(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const JobDesc* job_desc) const;

  Maybe<void> InferInplaceObn2IbnIf(
      HashMap<std::string, std::string>* mut_inplace_obn2ibn,
      HashMap<std::string, std::string>* con_inplace_obn2ibn,
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;

  Maybe<void> FillInputBlobTimeShape(
      const std::function<Maybe<const Shape>(int32_t)>& GetTimeShape4InputIndex);
  Maybe<void> InferOpTimeShapeIf();
  virtual Maybe<void> InferOpTimeShape(
      const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
      std::shared_ptr<const Shape>* time_shape) const;
  Maybe<const Shape> GetOpTimeShape() const;
  Maybe<const Shape> GetInputBlobFastestTimeShape() const;
  Maybe<const Shape> GetInputOutputFastestTimeShape() const;

  Maybe<void> InferSbpSignature(SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
                                const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint) const;
  Maybe<void> FillSbpSignature(const SbpSignature& sbp_signature);
  Maybe<void> FillParallelDistributionSignature(const ParallelDistributionSignature& signature);
  Maybe<void> InferSbpSignatureIf(
      const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc);
  Maybe<void> InferParallelDistributionSignatureIf(
      const ParallelDistributionSignature& parallel_distribution_constraints,
      const ParallelDesc& parallel_desc,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn);
  // Infer blob's MirroredSignature
  Maybe<void> InferMirroredSignatureIf(
      std::function<Maybe<const MirroredSigInferHint*>(const std::string&)>
          MirroredSigInferHint4Ibn,
      bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc);
  void GenKernelConf(const std::function<const BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                     const ParallelContext*, KernelConf*) const;
  const InputBlobModifier& InputBlobModifier4Ibn(const std::string& ibn) const;
  const OutputBlobModifier& OutputBlobModifier4Obn(const std::string& obn) const;
  Maybe<const SbpParallel*> SbpParallel4BnInOp(const std::string& bn_in_op) const;
  Maybe<const ParallelDistribution*> ParallelDistribution4BnInOp(const std::string& bn_in_op) const;
  Maybe<const OptMirroredParallel*> OptMirroredParallel4BnInOp(const std::string& bn_in_op) const;

  Maybe<void> GetSbpSignaturesIf(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const;

  void ForEachBnInOp(std::function<void(const std::string&)>) const;

  virtual Symbol<OperatorConf> GetOpConfWithoutOpNameAndLbn() const;
  std::shared_ptr<OpAttribute> GetOpAttributeWithoutOpNameAndLbn() const;

  Maybe<const SbpSignature*> sbp_signature() const;
  Maybe<const ParallelDistributionSignature*> parallel_distribution_signature() const;
  BlobLastUsedSignature* mut_blob_last_used_signature();
  BlobBackwardUsedSignature* mut_blob_backward_used_signature();

  Maybe<int32_t> GetInputIndex(const std::string& ibn) const;
  Maybe<int32_t> GetOutputIndex(const std::string& obn) const;
  Maybe<int32_t> GetOutputIndex(const LogicalBlobId& lbi) const;

  Maybe<void> ToOpAttribute(OpAttribute* op_attribute) const;

 protected:
  Maybe<void> FillBlobParallelDesc(
      const std::function<Maybe<const ParallelDesc>(const std::string&)>& ParallelDesc4Bn);
  virtual Maybe<void> InferBlobParallelDesc();
  virtual Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;
  virtual Maybe<void> InferInternalBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const JobDesc* job_desc) const;
  virtual Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const {
    return GetSbpSignatures(LogicalBlobDesc4Ibn, sbp_sig_list);
  }
  virtual Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const {
    return GetSbpSignatures(sbp_sig_list);
  }
  virtual Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const;
  virtual Maybe<void> InferParallelDistributionSignature(
      ParallelDistributionSignature* parallel_distribution_signature,
      const ParallelDistributionSignature& parallel_distribution_constraints,
      const ParallelDesc& parallel_desc,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn) const;
  virtual Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
    UNIMPLEMENTED() << " GetSbpSignatures unimplemented, op name: " << op_name();
    return Maybe<void>::Ok();
  }
  virtual Maybe<void> InferMirroredSignature(
      std::function<Maybe<const MirroredSigInferHint*>(const std::string&)>
          MirroredSigInferHint4Ibn,
      bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc);

  virtual Maybe<void> InferInplaceObn2Ibn(
      HashMap<std::string, std::string>* mut_inplace_obn2ibn,
      HashMap<std::string, std::string>* con_inplace_obn2ibn,
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;

  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*) const;

  virtual void AddLbi2OutputIndex(const LogicalBlobId& lbi, int32_t output_index);

  virtual LogicalBlobId lbi4ibn(const std::string& input_bn) const;
  virtual LogicalBlobId lbi4obn(const std::string& output_bn) const;

  // enroll data blobs
  void EnrollTmpBn(const std::string& dtbn);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num, bool has_diff);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, bool has_diff);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix, int32_t num);
  void EnrollRepeatedInputBn(const std::string& ibn_prefix);

  void EnrollRepeatedOutputBn(const std::string& obn_prefix, int32_t num, bool has_diff);
  void EnrollRepeatedOutputBn(const std::string& obn_prefix, bool has_diff);
  void EnrollRepeatedOutputBn(const std::string& obn_prefix, int32_t num);
  void EnrollRepeatedOutputBn(const std::string& obn_prefix);

  void EnrollRepeatedOutputBnWithSetter(
      const std::string& obn_prefix, int32_t num, bool has_diff,
      const std::function<void(OutputBlobModifier*)>& ModifierSetter);
  void EnrollRepeatedOutputBnWithSetter(
      const std::string& obn_prefix, bool has_diff,
      const std::function<void(OutputBlobModifier*)>& ModifierSetter);
  void EnrollRepeatedOutputBnWithSetter(
      const std::string& obn_prefix, int32_t num,
      const std::function<void(OutputBlobModifier*)>& ModifierSetter);
  void EnrollRepeatedOutputBnWithSetter(
      const std::string& obn_prefix,
      const std::function<void(OutputBlobModifier*)>& ModifierSetter);

  InputBlobModifier* EnrollInputBn(const std::string& ibn, bool has_diff);
  InputBlobModifier* EnrollInputBn(const std::string& ibn) { return EnrollInputBn(ibn, true); }
  OutputBlobModifier* EnrollOutputBn(const std::string& obn, bool has_diff);
  OutputBlobModifier* EnrollOutputBn(const std::string& obn) { return EnrollOutputBn(obn, true); }

  InputBlobModifier* MutInputBlobModifier4Ibn(const std::string& ibn);
  OutputBlobModifier* MutOutputBlobModifier4Obn(const std::string& obn);
  OptMirroredParallel* MutOptMirroredParallel(const std::string& bn_in_op);

 private:
  enum BlobNameTag {
    kInputBlobName,
    kOutputBlobName,
  };
  Maybe<void> FilterAndCheckValidSbpSignatureListByLogicalShape(
      const SbpSignatureList& total_sbp_sig_list,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* valid_sbp_sig_list) const;

  LogicalBlobId tbn2lbi(const std::string& data_tmp_bn) const;
  std::string Bn2ConfName(const std::string& bn) const;
  PbMap<std::string, LogicalBlobId>* mut_bn_in_op2lbi() {
    return arg_signature_.mutable_bn_in_op2lbi();
  }

  std::shared_ptr<const OperatorConf> op_conf_;
  std::shared_ptr<const ParallelDesc> op_parallel_desc_;
  std::unique_ptr<HashMap<std::string, std::shared_ptr<const ParallelDesc>>> bn2parallel_desc_;
  std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>> input_index2logical_blob_desc_;
  std::unique_ptr<std::vector<std::shared_ptr<const BlobDesc>>> output_index2logical_blob_desc_;
  std::unique_ptr<std::vector<std::shared_ptr<const Shape>>> input_index2time_shape_;
  std::shared_ptr<const Shape> input_blob_fastest_time_shape_;
  std::shared_ptr<const Shape> input_output_fastest_time_shape_;
  std::shared_ptr<const Shape> op_time_shape_;
  std::shared_ptr<const SbpSignature> sbp_signature_;
  std::shared_ptr<const ParallelDistributionSignature> parallel_distribution_signature_;
  PbRpf<std::string> input_bns_;
  PbRpf<std::string> output_bns_;
  PbRpf<std::string> tmp_bns_;
  PbRpf<std::string> input_output_bns_;
  DeviceType device_type_;
  ArgSignature arg_signature_;
  ArgModifierSignature arg_modifier_signature_;
  std::unique_ptr<BlobLastUsedSignature> blob_last_used_signature_;
  std::unique_ptr<BlobBackwardUsedSignature> blob_backward_used_signature_;
  std::unique_ptr<MirroredSignature> mirrored_signature_;

  HashMap<std::string, std::pair<BlobNameTag, int32_t>> bn2index_pair_;
  HashMap<LogicalBlobId, int32_t> lbi2output_index_;
};

std::string GenRepeatedBn(const std::string& bn_prefix, int32_t idx);
std::pair<std::string, int32_t> GenUnRepeatedBn(const std::string& bn);

bool IsCpuOnly(const OperatorConf& op_conf);

struct OnlyCpuSupportPredicator {
  OnlyCpuSupportPredicator(bool only_cpu) : only_cpu_(only_cpu) {}
  operator bool() { return only_cpu_; }

 private:
  bool only_cpu_;
};

struct RuntimeRegstNum4OpSameOutputBlob final {
  RuntimeRegstNum4OpSameOutputBlob(size_t num) : num_(num) {}
  RuntimeRegstNum4OpSameOutputBlob(std::function<size_t()> get_num)
      : get_num_(new std::function<size_t()>(get_num)) {}
  operator size_t() {
    if (!get_num_) { return num_; }
    return (*this->get_num_)();
  }

 private:
  size_t num_;
  std::unique_ptr<std::function<size_t()>> get_num_;
};

#define REGISTER_OP(op_type_case, OpType)                                       \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, OnlyCpuSupportPredicator,       \
                         ([] { return new OnlyCpuSupportPredicator(false); })); \
  REGISTER_CLASS_WITH_ARGS(int32_t, op_type_case, Operator, OpType, const OperatorConf&)

#define REGISTER_CPU_OP(op_type_case, OpType)                                  \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, OnlyCpuSupportPredicator,      \
                         ([] { return new OnlyCpuSupportPredicator(true); })); \
  REGISTER_CLASS_WITH_ARGS(int32_t, op_type_case, Operator, OpType, const OperatorConf&)

#define REGISTER_OP_CREATOR(op_type_case, creator)                              \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, OnlyCpuSupportPredicator,       \
                         ([] { return new OnlyCpuSupportPredicator(false); })); \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, Operator, creator, const OperatorConf&)

#define REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(op_type_case, num)                 \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, RuntimeRegstNum4OpSameOutputBlob, \
                         ([] { return new RuntimeRegstNum4OpSameOutputBlob(num); }))

struct IsInterfaceOpConf4OpTypeCase final {};

#define REGISTER_INTERFACE_OP(op_type_case)                                   \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, IsInterfaceOpConf4OpTypeCase, \
                         ([] { return new IsInterfaceOpConf4OpTypeCase(); }))

struct DisableInputBoxingGroup final {};

#define REGISTER_DISABLE_INPUT_BOXING_GROUP(op_type_case)                \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, DisableInputBoxingGroup, \
                         ([] { return new DisableInputBoxingGroup(); }))

struct IsTickTockOpTypeCase final {};

#define REGISTER_TICK_TOCK_OP(op_type_case)                           \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, IsTickTockOpTypeCase, \
                         ([] { return new IsTickTockOpTypeCase; }))

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf);
std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf, DeviceType device_type);

inline LogicalBlobId GenPackedLbi() {
  LogicalBlobId lbi;
  lbi.set_is_packed_id(true);
  return lbi;
}

inline OpBlobArg GenOpBlobArg(const std::string& op_name, const std::string& bn_in_op) {
  OpBlobArg oba;
  oba.set_op_name(op_name);
  oba.set_bn_in_op(bn_in_op);
  return oba;
}

LogicalBlobId GenLogicalBlobId(const std::string& lbn);

inline std::string GenLogicalBlobName(const std::string& op_name, const std::string& blob_name) {
  return op_name + "/" + blob_name;
}

inline std::string GenLogicalBlobName(const LogicalBlobId& lbi) {
  CHECK_EQ(lbi.has_op_name(), true);
  CHECK_EQ(lbi.has_blob_name(), true);
  CHECK_EQ(lbi.is_packed_id(), false);
  return GenLogicalBlobName(lbi.op_name(), lbi.blob_name());
}

Maybe<bool> GetSbpParallelInLbnOrNothing(const std::string& lbn, SbpParallel* sbp);
Maybe<bool> ParseDisableBoxingFlag(const std::string& lbn_with_hint, bool* disable_boxing);

std::string GetInputLbnInOpCustomizedConf(const OperatorConf& op_conf,
                                          const std::string& fd_name_may_have_idx);

// return old value
std::string ReplaceInputLbnInOpCustomizedConf(OperatorConf* op_conf,
                                              const std::string& fd_name_may_have_idx,
                                              const std::string& new_val);

bool operator==(const OperatorConf& lhs, const OperatorConf& rhs);

Maybe<Operator> ConstructAndInferOp(const OperatorConf& op_conf,
                                    const OpNodeSignature& upstream_signature, const Scope& scope);
Maybe<Shape> GetPhysicalShape(const Shape& logical_shape,
                              const ParallelDistribution& parallel_distribution,
                              const ParallelDesc& parallel_desc,
                              const ParallelContext& parallel_ctx);
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::OperatorConf> final {
  size_t operator()(const oneflow::OperatorConf& op_conf) {
    std::string serialized;
    op_conf.SerializeToString(&serialized);
    return std::hash<std::string>()(serialized);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_H_
