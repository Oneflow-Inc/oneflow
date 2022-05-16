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
#include "oneflow/core/framework/op_arg_util.h"

namespace oneflow {

namespace compatible_py {

namespace {

bool operator==(const OptMirroredParallel& lhs, const OptMirroredParallel& rhs) {
  return lhs.has_mirrored_parallel() == rhs.has_mirrored_parallel();
}

}  // namespace

OpArgBlobAttribute::OpArgBlobAttribute(const std::shared_ptr<BlobDescProto>& blob_desc,
                                       const std::string& logical_blob_name)
    : blob_desc_(blob_desc), logical_blob_name_(logical_blob_name) {
  shape_ = std::make_shared<Shape>(blob_desc_->shape());
  stride_ = std::make_shared<Stride>(blob_desc_->stride());
}

std::shared_ptr<BlobDescProto> OpArgBlobAttribute::blob_desc() const { return blob_desc_; }

std::shared_ptr<Shape> OpArgBlobAttribute::shape() const { return shape_; }

std::shared_ptr<Stride> OpArgBlobAttribute::stride() const { return stride_; }

std::string OpArgBlobAttribute::logical_blob_name() const { return logical_blob_name_; }

DataType OpArgBlobAttribute::get_dtype() const { return blob_desc_->data_type(); }

bool OpArgBlobAttribute::is_dynamic() const { return blob_desc_->is_dynamic(); }

bool OpArgBlobAttribute::operator==(const OpArgBlobAttribute& other) const {
  return (*shape_ == *other.shape()) && (*stride_ == *other.stride())
         && (get_dtype() == other.get_dtype()) && (is_dynamic() == other.is_dynamic())
         && (logical_blob_name_ == other.logical_blob_name());
}

std::shared_ptr<OpArgBlobAttribute> OpArgBlobAttribute::GetPhysicalOpArgBlobAttr(
    int64_t split_axis, int64_t parallel_num, int64_t parallel_id) const {
  std::shared_ptr<BlobDescProto> blob_desc = std::make_shared<BlobDescProto>();
  blob_desc->CopyFrom(*blob_desc_);
  int64_t physical_len =
      BalancedSplitter(shape_->At(split_axis), parallel_num).At(parallel_id).size();
  blob_desc->mutable_shape()->set_dim(split_axis, physical_len);
  blob_desc->mutable_stride()->set_dim(split_axis, physical_len);
  return std::make_shared<OpArgBlobAttribute>(blob_desc, logical_blob_name_);
}

OpArgParallelAttribute::OpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc,
    const std::shared_ptr<SbpParallel>& sbp_parallel,
    const std::shared_ptr<OptMirroredParallel>& opt_mirrored_parallel)
    : parallel_desc_(parallel_desc),
      sbp_parallel_(sbp_parallel),
      opt_mirrored_parallel_(opt_mirrored_parallel) {
  hash_ = Hash();
}

std::shared_ptr<ParallelDesc> OpArgParallelAttribute::parallel_desc_symbol() const {
  return parallel_desc_;
}

std::shared_ptr<SbpParallel> OpArgParallelAttribute::sbp_parallel() const { return sbp_parallel_; }

std::shared_ptr<OptMirroredParallel> OpArgParallelAttribute::opt_mirrored_parallel() const {
  return opt_mirrored_parallel_;
}

bool OpArgParallelAttribute::is_mirrored() const {
  return opt_mirrored_parallel_->has_mirrored_parallel();
}

std::size_t OpArgParallelAttribute::_Hash() const { return hash_; }

bool OpArgParallelAttribute::operator==(const OpArgParallelAttribute& other) const {
  return (*parallel_desc_ == *other.parallel_desc_symbol())
         && (*sbp_parallel_ == *other.sbp_parallel())
         && (*opt_mirrored_parallel_ == *other.opt_mirrored_parallel());
}

void OpArgParallelAttribute::Assign(const std::shared_ptr<OpArgParallelAttribute>& other) {
  parallel_desc_ = other->parallel_desc_symbol();
  sbp_parallel_ = other->sbp_parallel();
  opt_mirrored_parallel_ = other->opt_mirrored_parallel();
  hash_ = other->_Hash();
}

std::string OpArgParallelAttribute::ToString() const {
  return std::string("\nparallel_desc_symbol: ") + parallel_desc_->parallel_conf().DebugString()
         + "\nsbp_parallel: " + sbp_parallel_->DebugString()
         + "\nopt_mirrored_parallel: " + opt_mirrored_parallel_->DebugString() + "\n";
}

Maybe<OpArgBlobAttribute> GetOpArgBlobAttribute(const OpAttribute& op_attribute,
                                                const std::string& bn_in_op) {
  if (!op_attribute.has_logical_blob_desc_signature()) {
    return std::shared_ptr<OpArgBlobAttribute>();
  }
  const auto& blob_desc_signature_map =
      op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc();
  const auto& arg_signature_map = op_attribute.arg_signature().bn_in_op2lbi();
  const auto& lbi = arg_signature_map.at(bn_in_op);
  std::shared_ptr<BlobDescProto> blob_desc = std::make_shared<BlobDescProto>();
  if (blob_desc_signature_map.find(bn_in_op) != blob_desc_signature_map.end()) {
    blob_desc.reset(new BlobDescProto(blob_desc_signature_map.at(bn_in_op)));
  }
  return std::make_shared<OpArgBlobAttribute>(blob_desc, lbi.op_name() + "/" + lbi.blob_name());
}

Maybe<OpArgParallelAttribute> GetOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol, const OpAttribute& op_attribute,
    const std::string& bn_in_op) {
  const auto& sbp_signature_map = op_attribute.sbp_signature().bn_in_op2sbp_parallel();
  const auto& mirrored_signature_map =
      op_attribute.mirrored_signature().bn_in_op2opt_mirrored_parallel();
  std::shared_ptr<SbpParallel> sbp_parallel = std::make_shared<SbpParallel>();
  if (sbp_signature_map.find(bn_in_op) != sbp_signature_map.end()) {
    sbp_parallel.reset(new SbpParallel(sbp_signature_map.at(bn_in_op)));
  }
  std::shared_ptr<OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<OptMirroredParallel>();
  if (mirrored_signature_map.find(bn_in_op) != mirrored_signature_map.end()) {
    opt_mirrored_parallel.reset(new OptMirroredParallel(mirrored_signature_map.at(bn_in_op)));
  }
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgParallelAttribute> MakeMirroredOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<SbpParallel> sbp_parallel = std::make_shared<SbpParallel>();
  std::shared_ptr<OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<OptMirroredParallel>();
  opt_mirrored_parallel->mutable_mirrored_parallel();
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgParallelAttribute> MakeBroadcastOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<SbpParallel> sbp_parallel = std::make_shared<SbpParallel>();
  sbp_parallel->mutable_broadcast_parallel();
  std::shared_ptr<OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<OptMirroredParallel>();
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

}  // namespace compatible_py

}  // namespace oneflow
