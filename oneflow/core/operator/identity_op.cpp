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
#include "oneflow/core/job/local_sig_infer_hint.h"
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

  Maybe<void> InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
    return Maybe<void>::Ok();
  }
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(BlobDesc4BnInOp);
  }
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
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

class LocalCastOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalCastOp);
  LocalCastOp() = default;
  virtual ~LocalCastOp() override = default;

  Maybe<void> InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
    return Maybe<void>::Ok();
  }
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(BlobDesc4BnInOp);
  }
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    return InferBlobDescs(GetBlobDesc4BnInOp);
  }

 private:
};

namespace {

class CastToLocalOp : public LocalCastOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastToLocalOp);
  CastToLocalOp() = default;
  virtual ~CastToLocalOp() override = default;

 private:
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    *out = *BlobDesc4BnInOp("in");
    const SbpParallel& conf_sbp = SbpParallel(op_conf().cast_to_local_conf().sbp_parallel());
    if (conf_sbp.has_split_parallel()) {
      const int64_t axis = conf_sbp.split_parallel().axis();
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, out->shape().NumAxes());
      const int64_t dim_value = out->shape().At(axis);
      const int64_t parallel_num = parallel_desc.parallel_num();
      CHECK_EQ_OR_RETURN(dim_value % parallel_num, 0);
      Shape output = out->shape();
      output.Set(axis, dim_value / parallel_num);
      out->set_shape(output);
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    CHECK_NE_OR_RETURN(op_conf().cast_to_local_conf().sbp_parallel().parallel_type_case(),
                       SbpParallel::PARALLEL_TYPE_NOT_SET)
        << "attribute sbp_parallel not set.";
    const auto& ibn_hint = *JUST(SbpInferHint4Ibn("in"));
    CHECK_EQ_OR_RETURN(ibn_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    auto* map = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const SbpParallel& conf_sbp = SbpParallel(op_conf().cast_to_local_conf().sbp_parallel());
    CHECK_OR_RETURN(ibn_hint.sbp_parallel() == conf_sbp);
    (*map)["in"] = ibn_hint.sbp_parallel();
    (*map)["out"] = conf_sbp;
    return Maybe<void>::Ok();
  }
  Maybe<void> InferLocalSignature(
      std::function<Maybe<const LocalSigInferHint*>(const std::string&)> LocalSigInferHint4Ibn,
      bool is_local_parallel_view_conf, const ParallelDesc& parallel_desc) override {
    const auto& in_infer_hint = *JUST(LocalSigInferHint4Ibn("in"));
    CHECK_OR_RETURN(!in_infer_hint.is_local_parallel_view())
        << "error use of CastToLocalOp. `in' shouldn't be a local blob";
    CHECK_EQ_OR_RETURN(in_infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    MutOptLocalParallel("in")->clear_local_parallel();
    MutOptLocalParallel("out")->mutable_local_parallel();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCastToLocalConf, CastToLocalOp);

}  // namespace

namespace {

class CastFromLocalOp : public LocalCastOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastFromLocalOp);
  CastFromLocalOp() = default;
  virtual ~CastFromLocalOp() override = default;

 private:
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    BlobDesc* out = BlobDesc4BnInOp("out");
    *out = *BlobDesc4BnInOp("in");
    const SbpParallel& conf_sbp = SbpParallel(op_conf().cast_from_local_conf().sbp_parallel());
    if (conf_sbp.has_split_parallel()) {
      const int64_t axis = conf_sbp.split_parallel().axis();
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, out->shape().NumAxes());
      Shape output = out->shape();
      output.Set(axis, out->shape().At(axis) * parallel_desc.parallel_num());
      out->set_shape(output);
    }
    return Maybe<void>::Ok();
  }
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    CHECK_NE_OR_RETURN(op_conf().cast_from_local_conf().sbp_parallel().parallel_type_case(),
                       SbpParallel::PARALLEL_TYPE_NOT_SET)
        << "attribute sbp_parallel not set.";
    const auto& ibn_hint = *JUST(SbpInferHint4Ibn("in"));
    CHECK_EQ_OR_RETURN(ibn_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    auto* map = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*map)["in"] = ibn_hint.sbp_parallel();
    (*map)["out"] = SbpParallel(op_conf().cast_from_local_conf().sbp_parallel());
    return Maybe<void>::Ok();
  }
  Maybe<void> InferLocalSignature(
      std::function<Maybe<const LocalSigInferHint*>(const std::string&)> LocalSigInferHint4Ibn,
      bool is_local_parallel_view_conf, const ParallelDesc& parallel_desc) override {
    const auto& in_infer_hint = *JUST(LocalSigInferHint4Ibn("in"));
    CHECK_OR_RETURN(in_infer_hint.is_local_parallel_view())
        << "error use of CastFromLocalOp. `in' should be a local blob";
    CHECK_EQ_OR_RETURN(in_infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    MutOptLocalParallel("in")->mutable_local_parallel();
    MutOptLocalParallel("out")->clear_local_parallel();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCastFromLocalConf, CastFromLocalOp);

}  // namespace

}  // namespace oneflow
