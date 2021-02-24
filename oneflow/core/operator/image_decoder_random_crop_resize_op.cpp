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
#include "oneflow/core/vm/symbol_storage.h"
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

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
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
    return Maybe<void>::Ok();
  }

  Maybe<void> InferInternalBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const override {
    const ImageDecoderRandomCropResizeOpConf& conf =
        this->op_conf().image_decoder_random_crop_resize_conf();
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

  Maybe<void> InferBlobParallelDesc() override {
    HashMap<std::string, std::shared_ptr<const ParallelDesc>> bn2parallel_desc;
    const std::shared_ptr<const ParallelDesc> op_parallel_desc = JUST(GetOpParallelDesc());
    bn2parallel_desc["out"] = op_parallel_desc;
    if (device_type() == DeviceType::kCPU) {
      bn2parallel_desc["in"] = op_parallel_desc;
    } else if (device_type() == DeviceType::kGPU) {
      std::shared_ptr<ParallelDesc> in_parallel_desc =
          std::make_shared<ParallelDesc>(*op_parallel_desc);
      in_parallel_desc->set_device_type(DeviceType::kCPU);
      bn2parallel_desc["in"] = in_parallel_desc;
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    FillBlobParallelDesc([&](const std::string& bn) -> Maybe<const ParallelDesc> {
      auto it = bn2parallel_desc.find(bn);
      CHECK_OR_RETURN(it != bn2parallel_desc.end());
      return it->second;
    });
    return Maybe<void>::Ok();
  }
};

#if defined(WITH_CUDA) && CUDA_VERSION >= 10020
REGISTER_OP(OperatorConf::kImageDecoderRandomCropResizeConf, ImageDecoderRandomCropResizeOp);
#else
REGISTER_CPU_OP(OperatorConf::kImageDecoderRandomCropResizeConf, ImageDecoderRandomCropResizeOp);
#endif
}  // namespace oneflow
