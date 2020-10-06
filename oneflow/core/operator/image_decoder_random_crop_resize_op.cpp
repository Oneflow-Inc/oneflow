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
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/eager/eager_symbol_storage.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

class ImageDecoderRandomCropResizeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ImageDecoderRandomCropResizeOp);
  ImageDecoderRandomCropResizeOp() = default;
  ~ImageDecoderRandomCropResizeOp() override = default;

 private:
  void InitFromOpConf() override {
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
    EnrollTmpBn("tmp");
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const ImageDecoderRandomCropResizeOpConf& conf =
        this->op_conf().image_decoder_random_crop_resize_conf();
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    CHECK_EQ_OR_RETURN(in->data_type(), DataType::kTensorBuffer);
    *out = *in;
    out->set_data_type(DataType::kUInt8);
    DimVector out_dim_vec = in->shape().dim_vec();
    out_dim_vec.push_back(conf.target_height());
    out_dim_vec.push_back(conf.target_width());
    out_dim_vec.push_back(3);
    out->mut_shape() = Shape(out_dim_vec);
    BlobDesc* tmp = GetBlobDesc4BnInOp("tmp");
    tmp->set_data_type(DataType::kUInt8);
    tmp->mut_shape() = Shape({conf.max_num_pixels() * 3 * conf.num_workers()});
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("in", 0)
        .Split("out", 0)
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("in")).shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const ImageDecoderRandomCropResizeOpConf& conf =
        this->op_conf().image_decoder_random_crop_resize_conf();
    int64_t seed;
    if (conf.has_seed()) {
      seed = conf.seed();
    } else {
      std::random_device rd;
      seed = rd();
    }
    std::seed_seq seq{seed};
    std::vector<int64_t> seeds(parallel_ctx->parallel_num());
    seq.generate(seeds.begin(), seeds.end());
    kernel_conf->mutable_image_decoder_random_crop_resize_conf()->set_seed(
        seeds.at(parallel_ctx->parallel_id()));
    kernel_conf->mutable_image_decoder_random_crop_resize_conf()->set_batch_size(
        GetBlobDesc4BnInOp("in")->shape().elem_cnt());
  }

  LogicalNode* NewProperLogicalNode() const override {
    if (device_type() == DeviceType::kGPU) {
      return new DecodeH2DLogicalNode();
    } else {
      return new NormalForwardLogicalNode();
    }
  }

  Maybe<void> InferParallelSignature() override {
    if (device_type() == DeviceType::kCPU) {
      return Operator::InferParallelSignature();
    } else if (device_type() == DeviceType::kGPU) {
      const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
      const auto& scope = JUST(scope_storage.MaybeGet(op_conf().scope_symbol_id()));
      const int64_t device_parallel_desc_symbol_id =
          scope.scope_proto().device_parallel_desc_symbol_id();
      const int64_t host_parallel_desc_symbol_id =
          scope.scope_proto().host_parallel_desc_symbol_id();
      mut_parallel_signature()->set_op_parallel_desc_symbol_id(device_parallel_desc_symbol_id);
      auto* map = mut_parallel_signature()->mutable_bn_in_op2parallel_desc_symbol_id();
      for (const auto& ibn : input_bns()) { (*map)[ibn] = host_parallel_desc_symbol_id; }
      for (const auto& obn : output_bns()) { (*map)[obn] = device_parallel_desc_symbol_id; }
      for (const auto& tbn : tmp_bns()) { (*map)[tbn] = device_parallel_desc_symbol_id; }
      return Maybe<void>::Ok();
    } else {
      UNIMPLEMENTED();
      return Maybe<void>::Ok();
    }
  }
};

#if defined(WITH_CUDA) && CUDA_VERSION >= 10020
REGISTER_OP(OperatorConf::kImageDecoderRandomCropResizeConf, ImageDecoderRandomCropResizeOp);
#else
REGISTER_CPU_OP(OperatorConf::kImageDecoderRandomCropResizeConf, ImageDecoderRandomCropResizeOp);
#endif
}  // namespace oneflow
