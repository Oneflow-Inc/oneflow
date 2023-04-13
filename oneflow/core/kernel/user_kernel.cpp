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
#include "oneflow/core/kernel/user_kernel.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/op_kernel_infer_cache.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

bool IsAllBlobEmpty(const PbRpf<std::string>& bns,
                    const std::function<Blob*(const std::string& bn)>& BnInOp2Blob) {
  for (const auto& bn : bns) {
    Blob* blob = BnInOp2Blob(bn);
    if (blob && !blob->IsBodyEmpty()) { return false; }
  }
  return true;
}

}  // namespace

using Arg2Tensor =
    HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::BlobTensorView>>;
using ArgVec = std::vector<std::pair<std::string, int32_t>>;

namespace {

void FillTensorDescWithBlob(const Blob* blob, user_op::NaiveTensorDesc* tensor_desc) {
  BlobDescProto proto;
  blob->blob_desc().shape().ToProto(proto.mutable_shape());
  blob->blob_desc().stride().ToProto(proto.mutable_stride());
  proto.set_data_type(blob->blob_desc().data_type());
  proto.set_is_dynamic(blob->blob_desc().is_dynamic());
  *tensor_desc = proto;
  Shape tensor_desc_shape = tensor_desc->shape();
  tensor_desc_shape.CheckNumAxesIdenticalAndAssign(blob->shape());
  tensor_desc->set_shape(tensor_desc_shape);
  Stride tensor_desc_stride = tensor_desc->stride();
  tensor_desc_stride.CheckNumAxesIdenticalAndAssign(blob->stride());
  tensor_desc->set_stride(tensor_desc_stride);
}

}  // namespace

class UserKernelBaseContext {
 public:
  explicit UserKernelBaseContext(const KernelConf& kernel_conf) {
    CHECK(kernel_conf.has_user_conf());
    CHECK(kernel_conf.op_attribute().op_conf().has_user_conf());

    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                           ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg_vec->emplace_back(std::make_pair(it->first, i));
        }
      }
    };
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().input(), &inputs_);
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().output(), &outputs_);
    device_type_ =
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag()));
    parallel_ctx_ = kernel_conf.parallel_ctx();
    for (const auto& pair : kernel_conf.user_conf().bn_in_op2blob_desc()) {
      arg2bn_and_tensor_desc_.emplace(
          GenUnRepeatedBn(pair.first),
          std::make_pair(pair.first, user_op::NaiveTensorDesc(pair.second)));
    }
  }
  ~UserKernelBaseContext() = default;

  DeviceType device_type() const { return device_type_; }
  const ParallelContext& parallel_ctx() const { return parallel_ctx_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const {
    auto it = arg2bn_and_tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2bn_and_tensor_desc_.end()) { return nullptr; }
    return &(it->second.second);
  }

  const ArgVec& inputs() const { return inputs_; }
  const ArgVec& outputs() const { return outputs_; }

 private:
  friend class UserKernelInitAndCacheContext;
  HashMap<std::pair<std::string, int32_t>, std::pair<std::string, user_op::NaiveTensorDesc>>
      arg2bn_and_tensor_desc_;
  ArgVec inputs_;
  ArgVec outputs_;
  DeviceType device_type_;
  ParallelContext parallel_ctx_;
};

class UserKernelInitAndCacheContext final : public user_op::KernelInitContext,
                                            public user_op::KernelCacheContext {
 public:
  explicit UserKernelInitAndCacheContext(ep::Stream* stream, const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()),
        stream_(stream),
        base_ctx_(UserKernelBaseContext(kernel_conf)),
        parallel_desc_(kernel_conf.op_attribute().parallel_conf_signature().op_parallel_conf()) {
    nd_sbp_signature_ = NdSbpSignature(kernel_conf.op_attribute().nd_sbp_signature());
    if (kernel_conf.op_attribute().has_sbp_signature()) {
      sbp_signature_ = SbpSignature(kernel_conf.op_attribute().sbp_signature());
    }
    bool is_dynamic = false;
    for (const auto& pair : kernel_conf.user_conf().bn_in_op2blob_desc()) {
      if (pair.second.is_dynamic()) {
        is_dynamic = true;
        break;
      }
    }
    if (!is_dynamic || parallel_ctx().parallel_num() == 1) {
      for (const auto& pair :
           kernel_conf.op_attribute().logical_blob_desc_signature().bn_in_op2blob_desc()) {
        arg2logical_tensor_desc_.emplace(GenUnRepeatedBn(pair.first),
                                         user_op::NaiveTensorDesc(pair.second));
      }
    }
  }
  ~UserKernelInitAndCacheContext() override = default;

  ep::Stream* stream() override { return stream_; }

  void UpdateTensorWithCorrBlob(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    for (auto& pair : base_ctx_.arg2bn_and_tensor_desc_) {
      const std::string& bn = pair.second.first;
      auto& tensor_desc = pair.second.second;
      Blob* blob = BnInOp2Blob(bn);
      CHECK(blob != nullptr) << "Blob " << bn << " is not found in cache context.";
      if (blob->blob_desc().is_dynamic()) {
        Shape shape;
        blob->shape().ToShape(&shape);
        tensor_desc.set_shape(shape);
      }
    }
  }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    auto it = arg2logical_tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2logical_tensor_desc_.end()) {
      return nullptr;
    } else {
      return &(it->second);
    }
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    CHECK_EQ(parallel_desc_.hierarchy()->NumAxes(), 1);
    const auto& bn2sbp = sbp_signature_.bn_in_op2sbp_parallel();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2sbp.find(bn);
    CHECK(it != bn2sbp.end());
    return it->second;
  }

  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    const auto& bn2nd_sbp = nd_sbp_signature_.bn_in_op2nd_sbp();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2nd_sbp.find(bn);
    CHECK(it != bn2nd_sbp.end());
    return it->second;
  }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }
  const ParallelDesc& parallel_desc() const override { return parallel_desc_; }

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

  user_op::UserOpConfWrapper user_op_conf_;
  ep::Stream* stream_;
  UserKernelBaseContext base_ctx_;
  SbpSignature sbp_signature_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2logical_tensor_desc_;
  ParallelDesc parallel_desc_;
  NdSbpSignature nd_sbp_signature_;
};

using UserKernelInitContext = UserKernelInitAndCacheContext;
using UserKernelCacheContext = UserKernelInitAndCacheContext;

class UserKernelOpInferContext : public user_op::InferContext {
 public:
  explicit UserKernelOpInferContext(const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()),
        parallel_ctx_(kernel_conf.parallel_ctx()),
        nd_sbp_signature_(kernel_conf.op_attribute().nd_sbp_signature()),
        parallel_desc_(kernel_conf.op_attribute().parallel_conf_signature().op_parallel_conf()) {
    if (kernel_conf.op_attribute().has_sbp_signature()) {
      sbp_signature_ = SbpSignature(kernel_conf.op_attribute().sbp_signature());
    }
    auto InitTensorDesc = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                              ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          std::pair<std::string, int32_t> arg_pair = std::make_pair(arg_name, i);
          arg_vec->emplace_back(arg_pair);
          arg2tensor_desc_.emplace(arg_pair, nullptr);
        }
      }
    };
    InitTensorDesc(kernel_conf.op_attribute().op_conf().user_conf().input(), &inputs_);
    InitTensorDesc(kernel_conf.op_attribute().op_conf().user_conf().output(), &outputs_);
    for (const auto& pair :
         kernel_conf.op_attribute().logical_blob_desc_signature().bn_in_op2blob_desc()) {
      arg2logical_tensor_desc_.emplace(GenUnRepeatedBn(pair.first),
                                       user_op::NaiveTensorDesc(pair.second));
    }
  }
  ~UserKernelOpInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    auto it = arg2logical_tensor_desc_.find(std::make_pair(arg_name, index));
    CHECK(it != arg2logical_tensor_desc_.end())
        << "Arg (" << arg_name << "," << index << ") is not found";
    return &(it->second);
  }

  const user_op::TensorDesc& InputTensorDesc(const std::string& arg_name,
                                             int32_t index) const override {
    return *TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc& OutputTensorDesc(const std::string& arg_name,
                                              int32_t index) const override {
    return *TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  user_op::TensorDesc* MutOutputTensorDesc(const std::string& arg_name, int32_t index) override {
    return MutTensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return it->second.get();
  }
  user_op::TensorDesc* MutTensorDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index) {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return it->second.get();
  }
  const Shape& InputShape(const std::string& arg_name, int32_t index) const override {
    return Shape4ArgNameAndIndex(arg_name, index);
  }
  const Shape& OutputShape(const std::string& arg_name, int32_t index) const override {
    return Shape4ArgNameAndIndex(arg_name, index);
  }
  void SetOutputShape(const std::string& arg_name, int32_t index, const Shape& shape) override {
    SetShape4ArgNameAndIndex(arg_name, index, shape);
  }
  const Shape& Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->shape();
  }
  void SetShape4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                const Shape& shape) override {
    return MutTensorDesc4ArgNameAndIndex(arg_name, index)->set_shape(shape);
  }
  const Stride& InputStride(const std::string& arg_name, int32_t index) const override {
    return Stride4ArgNameAndIndex(arg_name, index);
  }
  const Stride& OutputStride(const std::string& arg_name, int32_t index) const override {
    return Stride4ArgNameAndIndex(arg_name, index);
  }
  void SetOutputStride(const std::string& arg_name, int32_t index, const Stride& stride) override {
    return SetStride4ArgNameAndIndex(arg_name, index, stride);
  }
  const Stride& Stride4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->stride();
  }
  void SetStride4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                 const Stride& stride) override {
    return MutTensorDesc4ArgNameAndIndex(arg_name, index)->set_stride(stride);
  }
  DataType InputDType(const std::string& arg_name, int32_t index) const override {
    return Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType OutputDType(const std::string& arg_name, int32_t index) const override {
    return Dtype4ArgNameAndIndex(arg_name, index);
  }
  void SetOutputDType(const std::string& arg_name, int32_t index, DataType data_type) override {
    return SetDtype4ArgNameAndIndex(arg_name, index, data_type);
  }
  DataType Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->data_type();
  }
  void SetDtype4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                DataType data_type) override {
    return MutTensorDesc4ArgNameAndIndex(arg_name, index)->set_data_type(data_type);
  }
  bool InputIsDynamic(const std::string& arg_name, int32_t index) const override {
    return IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  bool OutputIsDynamic(const std::string& arg_name, int32_t index) const override {
    return IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  void SetOutputIsDynamic(const std::string& arg_name, int32_t index, bool is_dynamic) override {
    return SetIsDynamic4ArgNameAndIndex(arg_name, index, is_dynamic);
  }
  bool IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->is_dynamic();
  }
  void SetIsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                    bool is_dynamic) override {
    return MutTensorDesc4ArgNameAndIndex(arg_name, index)->set_is_dynamic(is_dynamic);
  }

  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }
  const ParallelContext& parallel_ctx() const override { return parallel_ctx_; };
  const ParallelDesc& parallel_desc() const override { return parallel_desc_; }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    CHECK_EQ(parallel_desc_.hierarchy()->NumAxes(), 1);
    const auto& bn2sbp = sbp_signature_.bn_in_op2sbp_parallel();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2sbp.find(bn);
    CHECK(it != bn2sbp.end());
    return it->second;
  }
  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    const auto& bn2nd_sbp = nd_sbp_signature_.bn_in_op2nd_sbp();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2nd_sbp.find(bn);
    CHECK(it != bn2nd_sbp.end());
    return it->second;
  }
  void UpdateArg2TensorDesc(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    for (auto& pair : arg2tensor_desc_) {
      const auto& arg_pair = pair.first;
      std::unique_ptr<user_op::NaiveTensorDesc>* arg_tensor_desc_ptr = &pair.second;
      Blob* blob = BnInOp2Blob(GenRepeatedBn(arg_pair.first, arg_pair.second));
      CHECK_NOTNULL(blob);
      if (*arg_tensor_desc_ptr) {
        Shape tensor_desc_shape = (*arg_tensor_desc_ptr)->shape();
        tensor_desc_shape.CheckNumAxesIdenticalAndAssign(blob->shape());
        (*arg_tensor_desc_ptr)->set_shape(tensor_desc_shape);
        Stride tensor_desc_stride = (*arg_tensor_desc_ptr)->stride();
        tensor_desc_stride.CheckNumAxesIdenticalAndAssign(blob->stride());
        (*arg_tensor_desc_ptr)->set_stride(tensor_desc_stride);
      } else {
        arg_tensor_desc_ptr->reset(new user_op::NaiveTensorDesc());
        FillTensorDescWithBlob(blob, arg_tensor_desc_ptr->get());
      }
    }
  }

  int64_t parallel_num() const override { return parallel_ctx_.parallel_num(); }

  const std::string& input(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const override {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const override {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const override { return user_op_conf().op_name(); }
  const std::string& op_type_name() const override { return user_op_conf().op_type_name(); }
  const std::string& op_loc() const override { return user_op_conf_.op_conf().loc(); }

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const { return user_op_conf_; }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

  user_op::UserOpConfWrapper user_op_conf_;
  ArgVec inputs_;
  ArgVec outputs_;
  ParallelContext parallel_ctx_;
  SbpSignature sbp_signature_;
  NdSbpSignature nd_sbp_signature_;
  ParallelDesc parallel_desc_;
  HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::NaiveTensorDesc>>
      arg2tensor_desc_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2logical_tensor_desc_;
};

class UserKernelInferContext final : public user_op::KernelInferContext {
 public:
  explicit UserKernelInferContext(ep::Stream* stream, const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()),
        stream_(stream),
        base_ctx_(UserKernelBaseContext(kernel_conf)),
        op_infer_ctx_(kernel_conf) {
    auto InitArg2Blob = [this](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg2tensor_.emplace(std::make_pair(arg_name, i), nullptr);
        }
      }
    };
    InitArg2Blob(kernel_conf.op_attribute().op_conf().user_conf().input());
    InitArg2Blob(kernel_conf.op_attribute().op_conf().user_conf().output());

    const auto* op_reg_val = user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(
        kernel_conf.op_attribute().op_conf().user_conf().op_type_name());
    CHECK_NOTNULL(op_reg_val);
    if (op_reg_val->physical_tensor_desc_infer_fn) {
      tensor_desc_infer_fn_ = op_reg_val->physical_tensor_desc_infer_fn;
    } else {
      UNIMPLEMENTED();
    }
  }
  ~UserKernelInferContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  ep::Stream* stream() override { return stream_; }
  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) override {
    auto it = arg2tensor_.find(std::make_pair(arg_name, arg_index));
    CHECK(it != arg2tensor_.end()) << "Arg (" << arg_name << "," << arg_index << ") is not found";
    return it->second.get();
  }
  ShapeView ShapeView4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) override {
    user_op::Tensor* arg_tensor = Tensor4ArgNameAndIndex(arg_name, arg_index);
    CHECK(arg_tensor != nullptr) << "Tensor of arg (" << arg_name << "," << arg_index
                                 << ") is not found";
    return arg_tensor->shape_view();
  }
  MutShapeView MutShapeView4ArgNameAndIndex(const std::string& arg_name,
                                            int32_t arg_index) override {
    user_op::Tensor* arg_tensor = Tensor4ArgNameAndIndex(arg_name, arg_index);
    CHECK(arg_tensor != nullptr) << "Tensor of arg (" << arg_name << "," << arg_index
                                 << ") is not found";
    return arg_tensor->mut_shape_view();
  }

  user_op::InferContext* MutOpInferContext() override { return &op_infer_ctx_; }
  const user_op::TensorDescInferFn& GetOpInferFn() const override { return tensor_desc_infer_fn_; }

  void UpdateArg2Tensor(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    for (auto& pair : arg2tensor_) {
      const auto& arg_pair = pair.first;
      std::unique_ptr<user_op::BlobTensorView>* arg_tensor_ptr = &pair.second;
      Blob* blob = BnInOp2Blob(GenRepeatedBn(arg_pair.first, arg_pair.second));
      if (blob == nullptr) { continue; }
      if (*arg_tensor_ptr) {
        arg_tensor_ptr->get()->Reset(blob);
      } else {
        arg_tensor_ptr->reset(new user_op::BlobTensorView(blob));
      }
    }
  }

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

  user_op::UserOpConfWrapper user_op_conf_;
  ep::Stream* stream_;
  UserKernelBaseContext base_ctx_;
  UserKernelOpInferContext op_infer_ctx_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::BlobTensorView>> arg2tensor_;
};

namespace {

struct BnTensorPair {
  std::string bn;
  std::unique_ptr<user_op::BlobTensorView> tensor;
};

BnTensorPair MakeBnTensorPair(const std::string& bn) {
  BnTensorPair pair;
  pair.bn = bn;
  return pair;
}

BnTensorPair MakeBnTensorPair(const std::string& bn,
                              std::unique_ptr<user_op::BlobTensorView>&& tensor) {
  BnTensorPair pair;
  pair.bn = bn;
  pair.tensor = std::move(tensor);
  return pair;
}

}  // namespace

class UserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit UserKernelComputeContext(ep::Stream* stream, const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()),
        stream_(stream),
        base_ctx_(kernel_conf) {
    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (const auto& it : arg_map) {
        const std::string& arg_name = it.first;
        for (int32_t i = 0; i < it.second.s_size(); ++i) {
          arg2bn_tensor_pair_.emplace(std::make_pair(arg_name, i),
                                      MakeBnTensorPair(GenRepeatedBn(arg_name, i)));
        }
      }
    };
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().input());
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().output());
    arg2bn_tensor_pair_.emplace(std::make_pair("tmp_buffer", 0),
                                MakeBnTensorPair(GenRepeatedBn("tmp_buffer", 0)));
  }
  ~UserKernelComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2bn_tensor_pair_.find(std::make_pair(arg_name, index));
    if (it == arg2bn_tensor_pair_.end()) { return nullptr; }
    return it->second.tensor.get();
  }
  ep::Stream* stream() override { return stream_; }

  bool UpdateTensorWithCorrBlob(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    bool updated = false;
    for (auto& pair : arg2bn_tensor_pair_) {
      std::unique_ptr<user_op::BlobTensorView>* arg_tensor_ptr = &pair.second.tensor;
      Blob* blob = BnInOp2Blob(pair.second.bn);
      if (blob == nullptr) {
        if (*arg_tensor_ptr) {
          arg_tensor_ptr->reset(nullptr);
          updated = true;
        }
      } else {
        if (*arg_tensor_ptr) {
          if (arg_tensor_ptr->get()->blob() != blob) {
            arg_tensor_ptr->get()->Reset(blob);
            updated = true;
          } else {
            if (blob->blob_desc().is_dynamic()) { updated = true; }
          }
        } else {
          arg_tensor_ptr->reset(new user_op::BlobTensorView(blob));
          updated = true;
        }
      }
    }
    return updated;
  }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

  user_op::UserOpConfWrapper user_op_conf_;
  ep::Stream* stream_;
  HashMap<std::pair<std::string, int32_t>, BnTensorPair> arg2bn_tensor_pair_;
  UserKernelBaseContext base_ctx_;
};

// kernel registry context used in kernel creation
class UserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit UserKernelRegContext(const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()),
        base_ctx_(UserKernelBaseContext(kernel_conf)) {}
  ~UserKernelRegContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

 private:
  user_op::UserOpConfWrapper user_op_conf_;
  UserKernelBaseContext base_ctx_;
};

UserKernel::~UserKernel() = default;

void UserKernel::InitUserKernel(ep::Stream* stream) {
  ctx_.reset(new UserKernelComputeContext(stream, kernel_conf()));
  infer_ctx_.reset(new UserKernelInferContext(stream, kernel_conf()));
  cache_ctx_.reset(new UserKernelCacheContext(stream, kernel_conf()));
  infer_cache_.reset(new user_op::OpKernelInferCache(kernel_conf(), this));
  {
    const std::string& op_type_name =
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name();
    const user_op::OpKernelRegistryResult* kernel_reg_val =
        CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
            op_type_name, UserKernelRegContext(kernel_conf())));
    CHECK_NOTNULL(kernel_reg_val);
    kernel_.reset(kernel_reg_val->create_fn());
  }
}

std::shared_ptr<user_op::OpKernelState> UserKernel::CreateOpKernelState(KernelContext* ctx) {
  UserKernelInitContext init_ctx(ctx->stream(), kernel_conf());
  return kernel_->CreateOpKernelState(&init_ctx);
}

const std::shared_ptr<user_op::OpKernelState>& UserKernel::GetOpKernelState() const {
  return opkernel_state_;
}

void UserKernel::ForwardUserKernel(const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                                   user_op::OpKernelState* opkernel_state) const {
  const bool updated = ctx_->UpdateTensorWithCorrBlob(BnInOp2Blob);

  if (updated) {
    cache_ctx_->UpdateTensorWithCorrBlob(BnInOp2Blob);
    kernel_->InitOpKernelCacheWithFlags(cache_ctx_.get(), user_op::OpKernelCache::kAttrNotChanged,
                                        &opkernel_cache_);
  } else {
    // do nothing
  }
#ifdef WITH_CUDA_GRAPHS
  bool current_scope_capturing = false;
  if (cuda_graph_exec_) {
    auto* cuda_stream = dynamic_cast<ep::CudaStream*>(ctx_->stream());
    if (!cuda_stream->IsGraphCapturing()) {
      if (cuda_graph_exec_->IsInstantiated() && (!updated)) {
        cuda_stream->LaunchGraph(cuda_graph_exec_.get());
        return;
      }
      const auto* cuda_graph_support =
          CHECK_NOTNULL(dynamic_cast<const user_op::CudaGraphSupport*>(kernel_.get()));
      if (cuda_graph_support->IsReadyForCapture(ctx_.get(), opkernel_state,
                                                opkernel_cache_.get())) {
        current_scope_capturing = true;
        cuda_stream->BeginGraphCapture();
      }
    }
  }
#endif  // WITH_CUDA_GRAPHS

  kernel_->Compute(ctx_.get(), opkernel_state, opkernel_cache_.get());

#ifdef WITH_CUDA_GRAPHS
  if (cuda_graph_exec_ && current_scope_capturing) {
    auto* cuda_stream = dynamic_cast<ep::CudaStream*>(ctx_->stream());
    cuda_stream->EndGraphCapture(cuda_graph_exec_.get());
    cuda_stream->LaunchGraph(cuda_graph_exec_.get());
  }
#endif  // WITH_CUDA_GRAPHS
}

bool UserKernel::IsCudaGraphSupported() const {
#ifdef WITH_CUDA_GRAPHS
  return cuda_graph_exec_.get() != nullptr;
#else
  return false;
#endif  // WITH_CUDA_GRAPHS
}

bool UserKernel::IsReadyForCudaGraphCapture(KernelContext* ctx) const {
  const auto* cuda_graph_support = dynamic_cast<const user_op::CudaGraphSupport*>(kernel_.get());
  if (cuda_graph_support == nullptr) { return false; }
  return cuda_graph_support->IsReadyForCapture(ctx_.get(), opkernel_state_.get(),
                                               opkernel_cache_.get());
}

void UserKernel::VirtualKernelInit(KernelContext* ctx) {
  InitUserKernel(ctx->stream());
  CHECK(opkernel_state_.get() == nullptr);
  opkernel_state_ = CreateOpKernelState(ctx);
  kernel_->InitOpKernelCacheWithFlags(cache_ctx_.get(), user_op::OpKernelCache::kAllMayChanged,
                                      &opkernel_cache_);
#ifdef WITH_CUDA_GRAPHS
  if (ParseBooleanFromEnv("ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH", false)
      && (!ParseBooleanFromEnv("ONEFLOW_GRAPH_ENABLE_STREAM_ORDERED_MEMORY_ALLOCATION", false))) {
    UserKernelInitContext init_ctx(ctx->stream(), kernel_conf());
    auto* cuda_stream = dynamic_cast<ep::CudaStream*>(ctx->stream());
    const auto* cuda_graph_support = dynamic_cast<const user_op::CudaGraphSupport*>(kernel_.get());
    if (cuda_stream != nullptr) {
      if (cuda_graph_support != nullptr
          && cuda_graph_support->IsCudaGraphSupported(&init_ctx, opkernel_state_.get())) {
        cuda_graph_exec_.reset(new ep::CudaGraphExecutable());
        VLOG(3) << "CUDA Graphs Kernel: " << op_conf().name() << " ("
                << op_conf().user_conf().op_type_name() << ")";
      } else {
        VLOG(3) << "CUDA Graphs not supported: " << op_conf().name() << " ("
                << op_conf().user_conf().op_type_name() << ")";
      }
    }
  }
#endif  // WITH_CUDA_GRAPHS
}

void UserKernel::ForwardDataContent(KernelContext* ctx) const {
  const auto BnInOp2Blob = [ctx](const std::string& bn) { return ctx->BnInOp2Blob(bn); };
  ForwardUserKernel(BnInOp2Blob, opkernel_state_.get());
}

void UserKernel::ForwardShape(KernelContext* ctx) const {
  const auto BnInOp2Blob = [ctx](const std::string& bn) { return ctx->BnInOp2Blob(bn); };
  infer_ctx_->UpdateArg2Tensor(BnInOp2Blob);
  infer_cache_->UpdateCacheKey(infer_ctx_.get());
  if (!infer_cache_->IsCacheHit()) {
    auto* op_infer_ctx = dynamic_cast<UserKernelOpInferContext*>(infer_ctx_->MutOpInferContext());
    CHECK_NOTNULL(op_infer_ctx);
    op_infer_ctx->UpdateArg2TensorDesc(BnInOp2Blob);
    kernel_->InferShape(infer_ctx_.get());
    for (const auto& out_arg_pair : infer_ctx_->outputs()) {
      const Shape& static_shape =
          infer_ctx_->TensorDesc4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second)->shape();
      const ShapeView& shape_view =
          infer_ctx_->ShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      CHECK_LE(shape_view.elem_cnt(), static_shape.elem_cnt())
          << "InferShape of OpKernel (op_type_name: " << op_conf().user_conf().op_type_name()
          << ", op_name: " << op_conf().name()
          << ") raise error, output arg's (name: " << out_arg_pair.first
          << ", index: " << out_arg_pair.second << ") runtime shape " << shape_view.ToString()
          << " surpass the limit of static shape " << static_shape.ToString();
    }
    infer_cache_->UpdateCacheValue(infer_ctx_.get());
  } else {
    std::shared_ptr<const OpInferCacheValue> cache_value_ptr = infer_cache_->GetCacheValue();
    FOR_RANGE(int, i, 0, infer_ctx_->outputs().size()) {
      const auto& out_arg_pair = infer_ctx_->outputs().at(i);
      MutShapeView mut_shape_view =
          infer_ctx_->MutShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      mut_shape_view.set_shape(*cache_value_ptr->obn_idx2shape_sym.at(i));
    }
  }
}

bool UserKernel::IsStateless() const { return !kernel_->AlwaysComputeWhenAllOutputsEmpty(); }
NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel).SetIsMatchedPred([](const KernelConf&) {
  return true;
});

}  // namespace oneflow
