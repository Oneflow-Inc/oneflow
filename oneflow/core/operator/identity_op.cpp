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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  *BlobDesc4BnInOp("out") = *BlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

}  // namespace

template<typename T>
class IdentityOpTpl final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentityOpTpl);
  IdentityOpTpl() = default;
  ~IdentityOpTpl() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(BlobDesc4BnInOp);
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(GetBlobDesc4BnInOp);
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
    SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
    const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in")).shape().NumAxes();
    SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

struct IdentityOp {};
REGISTER_OP(OperatorConf::kIdentityConf, IdentityOpTpl<IdentityOp>);

struct CopyOp {};
REGISTER_OP(OperatorConf::kCopyConf, IdentityOpTpl<CopyOp>);

class MirroredCastOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MirroredCastOp);
  MirroredCastOp() = default;
  virtual ~MirroredCastOp() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(BlobDesc4BnInOp);
  }
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(GetBlobDesc4BnInOp);
  }

 private:
};

namespace {

class CastToMirroredOp : public MirroredCastOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastToMirroredOp);
  CastToMirroredOp() = default;
  virtual ~CastToMirroredOp() override = default;

 private:
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    *out = *BlobDesc4BnInOp("in");
    const SbpParallel& conf_sbp = op_conf().cast_to_mirrored_conf().sbp_parallel();
    if (conf_sbp.has_split_parallel()) {
      const int64_t axis = conf_sbp.split_parallel().axis();
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, out->shape().NumAxes());
      const int64_t dim_value = out->shape().At(axis);
      const int64_t parallel_num = parallel_desc.parallel_num();
      CHECK_EQ_OR_RETURN(dim_value % parallel_num, 0);
      out->mut_shape().Set(axis, dim_value / parallel_num);
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    CHECK_NE_OR_RETURN(op_conf().cast_to_mirrored_conf().sbp_parallel().parallel_type_case(),
                       SbpParallel::PARALLEL_TYPE_NOT_SET)
        << "attribute sbp_parallel not set.";
    const auto& ibn_hint = *JUST(SbpInferHint4Ibn("in"));
    CHECK_EQ_OR_RETURN(ibn_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    auto* map = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const SbpParallel& conf_sbp = op_conf().cast_to_mirrored_conf().sbp_parallel();
    CHECK_OR_RETURN(ibn_hint.sbp_parallel() == conf_sbp);
    (*map)["in"] = ibn_hint.sbp_parallel();
    (*map)["out"] = conf_sbp;
    return Maybe<void>::Ok();
  }
  Maybe<void> InferMirroredSignature(
      std::function<Maybe<const MirroredSigInferHint*>(const std::string&)>
          MirroredSigInferHint4Ibn,
      bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc) override {
    const auto& in_infer_hint = *JUST(MirroredSigInferHint4Ibn("in"));
    CHECK_OR_RETURN(!in_infer_hint.is_mirrored_parallel_view())
        << "error use of CastToMirroredOp. `in' shouldn't be a mirrored blob";
    CHECK_EQ_OR_RETURN(in_infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    MutOptMirroredParallel("in")->clear_mirrored_parallel();
    MutOptMirroredParallel("out")->mutable_mirrored_parallel();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCastToMirroredConf, CastToMirroredOp);

}  // namespace

namespace {

class CastFromMirroredOp : public MirroredCastOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastFromMirroredOp);
  CastFromMirroredOp() = default;
  virtual ~CastFromMirroredOp() override = default;

 private:
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    *out = *BlobDesc4BnInOp("in");
    const SbpParallel& conf_sbp = op_conf().cast_from_mirrored_conf().sbp_parallel();
    if (conf_sbp.has_split_parallel()) {
      const int64_t axis = conf_sbp.split_parallel().axis();
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, out->shape().NumAxes());
      out->mut_shape().Set(axis, out->shape().At(axis) * parallel_desc.parallel_num());
    }
    return Maybe<void>::Ok();
  }
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    CHECK_NE_OR_RETURN(op_conf().cast_from_mirrored_conf().sbp_parallel().parallel_type_case(),
                       SbpParallel::PARALLEL_TYPE_NOT_SET)
        << "attribute sbp_parallel not set.";
    const auto& ibn_hint = *JUST(SbpInferHint4Ibn("in"));
    CHECK_EQ_OR_RETURN(ibn_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    auto* map = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*map)["in"] = ibn_hint.sbp_parallel();
    (*map)["out"] = op_conf().cast_from_mirrored_conf().sbp_parallel();
    return Maybe<void>::Ok();
  }
  Maybe<void> InferMirroredSignature(
      std::function<Maybe<const MirroredSigInferHint*>(const std::string&)>
          MirroredSigInferHint4Ibn,
      bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc) override {
    const auto& in_infer_hint = *JUST(MirroredSigInferHint4Ibn("in"));
    CHECK_OR_RETURN(in_infer_hint.is_mirrored_parallel_view())
        << "error use of CastFromMirroredOp. `in' should be a mirrored blob";
    CHECK_EQ_OR_RETURN(in_infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    MutOptMirroredParallel("in")->mutable_mirrored_parallel();
    MutOptMirroredParallel("out")->clear_mirrored_parallel();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCastFromMirroredConf, CastFromMirroredOp);

}  // namespace

}  // namespace oneflow
