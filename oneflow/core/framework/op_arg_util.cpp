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

OpArgBlobAttribute::OpArgBlobAttribute(const std::shared_ptr<cfg::BlobDescProto>& blob_desc,
                                       const std::string& logical_blob_name)
    : blob_desc_(blob_desc), logical_blob_name_(logical_blob_name) {
  ShapeProto shape_proto;
  blob_desc_->body().shape().ToProto(&shape_proto);
  shape_ = std::make_shared<Shape>(shape_proto);
}

std::shared_ptr<cfg::BlobDescProto> OpArgBlobAttribute::blob_desc() const { return blob_desc_; }

std::shared_ptr<Shape> OpArgBlobAttribute::shape() const { return shape_; }

std::string OpArgBlobAttribute::logical_blob_name() const { return logical_blob_name_; }

cfg::DataType OpArgBlobAttribute::get_dtype() const { return blob_desc_->body().data_type(); }

bool OpArgBlobAttribute::is_tensor_list() const { return blob_desc_->is_tensor_list(); }

bool OpArgBlobAttribute::is_dynamic() const { return blob_desc_->is_dynamic(); }

bool OpArgBlobAttribute::operator==(const OpArgBlobAttribute& other) const {
  return (*shape_ == *other.shape()) && (get_dtype() == other.get_dtype())
         && (is_tensor_list() == other.is_tensor_list()) && (is_dynamic() == other.is_dynamic())
         && (logical_blob_name_ == other.logical_blob_name());
}

std::shared_ptr<OpArgBlobAttribute> OpArgBlobAttribute::GetPhysicalOpArgBlobAttr(
    int64_t split_axis, int64_t parallel_num, int64_t parallel_id) const {
  std::shared_ptr<cfg::BlobDescProto> blob_desc = std::make_shared<cfg::BlobDescProto>();
  blob_desc->CopyFrom(*blob_desc_);
  int64_t physical_len =
      BalancedSplitter(shape_->At(split_axis), parallel_num).At(parallel_id).size();
  blob_desc->mutable_body()->mutable_shape()->set_dim(split_axis, physical_len);
  return std::make_shared<OpArgBlobAttribute>(blob_desc, logical_blob_name_);
}

void OpArgBlobAttribute::DumpToInterfaceBlobConf(
    std::shared_ptr<cfg::InterfaceBlobConf> interface_blob_conf) const {
  for (int i = 0; i < shape_->NumAxes(); ++i) {
    interface_blob_conf->mutable_shape()->add_dim(shape_->At(i));
  }
  interface_blob_conf->set_data_type(blob_desc_->body().data_type());
  interface_blob_conf->set_is_dynamic(is_dynamic());
  interface_blob_conf->set_is_tensor_list(is_tensor_list());
}

void OpArgBlobAttribute::DumpToOpNodeSignature(
    std::string bn_in_op, std::shared_ptr<cfg::OpNodeSignature> op_node_signature) const {
  auto& blob_sig =
      *(op_node_signature->mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc());
  CHECK(blob_sig.find(bn_in_op) == blob_sig.end());
  blob_sig[bn_in_op].CopyFrom(*blob_desc_);
}

OpArgParallelAttribute::OpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc,
    const std::shared_ptr<cfg::SbpParallel>& sbp_parallel,
    const std::shared_ptr<cfg::OptMirroredParallel>& opt_mirrored_parallel)
    : parallel_desc_(parallel_desc),
      sbp_parallel_(sbp_parallel),
      opt_mirrored_parallel_(opt_mirrored_parallel) {
  hash_ = Hash();
}

std::shared_ptr<ParallelDesc> OpArgParallelAttribute::parallel_desc_symbol() const {
  return parallel_desc_;
}

std::shared_ptr<cfg::SbpParallel> OpArgParallelAttribute::sbp_parallel() const {
  return sbp_parallel_;
}

std::shared_ptr<cfg::OptMirroredParallel> OpArgParallelAttribute::opt_mirrored_parallel() const {
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

void OpArgParallelAttribute::DumpToInterfaceBlobConf(
    std::shared_ptr<cfg::InterfaceBlobConf> interface_blob_conf) const {
  if (sbp_parallel_->has_split_parallel()) {
    interface_blob_conf->mutable_split_axis()->set_value(sbp_parallel_->split_parallel().axis());
  } else {
    interface_blob_conf->clear_split_axis();
  }
}

void OpArgParallelAttribute::DumpToOpNodeSignature(
    std::string bn_in_op, std::shared_ptr<cfg::OpNodeSignature> op_node_signature) const {
  auto& sbp_sig = *(op_node_signature->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel());
  CHECK(sbp_sig.find(bn_in_op) == sbp_sig.end());
  sbp_sig[bn_in_op].CopyFrom(*sbp_parallel_);

  auto& mirrored_sig =
      *(op_node_signature->mutable_mirrored_signature()->mutable_bn_in_op2opt_mirrored_parallel());
  CHECK(mirrored_sig.find(bn_in_op) == mirrored_sig.end());
  mirrored_sig[bn_in_op].CopyFrom(*opt_mirrored_parallel_);

  auto& parallel_sig = *(
      op_node_signature->mutable_parallel_signature()->mutable_bn_in_op2parallel_desc_symbol_id());
  CHECK(parallel_sig.find(bn_in_op) == parallel_sig.end());
  parallel_sig[bn_in_op] = CHECK_JUST(parallel_desc_->symbol_id());
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
  auto& blob_desc_signature_map = op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc();
  auto& arg_signature_map = op_attribute.arg_signature().bn_in_op2lbi();
  auto& lbi = arg_signature_map.at(bn_in_op);
  std::shared_ptr<cfg::BlobDescProto> blob_desc = std::make_shared<cfg::BlobDescProto>();
  if (blob_desc_signature_map.find(bn_in_op) != blob_desc_signature_map.end()) {
    blob_desc.reset(new cfg::BlobDescProto(blob_desc_signature_map.at(bn_in_op)));
  }
  return std::make_shared<OpArgBlobAttribute>(blob_desc, lbi.op_name() + "/" + lbi.blob_name());
}

Maybe<OpArgParallelAttribute> GetOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol, const OpAttribute& op_attribute,
    const std::string& bn_in_op) {
  auto& sbp_signature_map = op_attribute.sbp_signature().bn_in_op2sbp_parallel();
  auto& mirrored_signature_map = op_attribute.mirrored_signature().bn_in_op2opt_mirrored_parallel();
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = std::make_shared<cfg::SbpParallel>();
  if (sbp_signature_map.find(bn_in_op) != sbp_signature_map.end()) {
    sbp_parallel.reset(new cfg::SbpParallel(sbp_signature_map.at(bn_in_op)));
  }
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<cfg::OptMirroredParallel>();
  if (mirrored_signature_map.find(bn_in_op) != mirrored_signature_map.end()) {
    opt_mirrored_parallel.reset(new cfg::OptMirroredParallel(mirrored_signature_map.at(bn_in_op)));
  }
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgParallelAttribute> MakeMirroredOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = std::make_shared<cfg::SbpParallel>();
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<cfg::OptMirroredParallel>();
  opt_mirrored_parallel->mutable_mirrored_parallel();
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgParallelAttribute> MakeBroadcastOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = std::make_shared<cfg::SbpParallel>();
  sbp_parallel->mutable_broadcast_parallel();
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<cfg::OptMirroredParallel>();
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

}  // namespace compatible_py

}  // namespace oneflow
