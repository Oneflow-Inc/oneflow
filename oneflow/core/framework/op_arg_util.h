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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_ARG_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_ARG_UTIL_H_

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/mirrored_parallel.cfg.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/common/shape.cfg.h"
#include "oneflow/core/register/logical_blob_id.cfg.h"
#include "oneflow/core/operator/inter_face_blob_conf.cfg.h"
#include "oneflow/core/register/pod.cfg.h"
#include "oneflow/core/register/blob_desc.cfg.h"

namespace oneflow {

namespace compatible_py {

class OpArgBlobAttribute {
 public:
  OpArgBlobAttribute(const std::shared_ptr<cfg::OptInt64>& batch_axis,
                     const std::shared_ptr<cfg::BlobDescProto>& blob_desc,
                     const std::string& logical_blob_name)
      : batch_axis_(batch_axis), blob_desc_(blob_desc), logical_blob_name_(logical_blob_name) {
    ShapeProto shape;
    blob_desc_->body().shape().ToProto(&shape);
    shape_ = std::make_shared<Shape>(shape);
  }

  OpArgBlobAttribute(const OpArgBlobAttribute& op_arg_blob_attr) = default;
  virtual ~OpArgBlobAttribute() = default;

  std::shared_ptr<cfg::BlobDescProto> blob_desc() const { return blob_desc_; }

  std::shared_ptr<Shape> shape() const { return shape_; }

  std::string logical_blob_name() const { return logical_blob_name_; }

  cfg::DataType get_dtype() const { return blob_desc_->body().data_type(); }

  std::shared_ptr<cfg::OptInt64> batch_axis() const { return batch_axis_; }

  bool is_tensor_list() const { return blob_desc_->is_tensor_list(); }

  bool is_dynamic() const { return blob_desc_->is_dynamic(); }

  bool operator==(const OpArgBlobAttribute& other) const {
    return (*shape_ == *other.shape()) && (*batch_axis_ == *other.batch_axis())
           && (get_dtype() == other.get_dtype()) && (is_tensor_list() == other.is_tensor_list())
           && (is_dynamic() == other.is_dynamic())
           && (logical_blob_name_ == other.logical_blob_name());
  }

  std::shared_ptr<OpArgBlobAttribute> GetPhysicalOpArgBlobAttr(int64_t split_axis,
                                                               int64_t parallel_num,
                                                               int64_t parallel_id) const {
    std::shared_ptr<cfg::BlobDescProto> blob_desc = std::make_shared<cfg::BlobDescProto>();
    blob_desc->CopyFrom(*blob_desc_);
    int64_t physical_len =
        BalancedSplitter(shape_->At(split_axis), parallel_num).At(parallel_id).size();
    blob_desc->mutable_body()->mutable_shape()->set_dim(split_axis, physical_len);
    return std::make_shared<OpArgBlobAttribute>(batch_axis_, blob_desc, logical_blob_name_);
  }

  void DumpToInterfaceBlobConf(std::shared_ptr<cfg::InterfaceBlobConf> interface_blob_conf) const {
    for (int i = 0; i < shape_->NumAxes(); ++i) {
      interface_blob_conf->mutable_shape()->add_dim(shape_->At(i));
    }
    interface_blob_conf->set_data_type(blob_desc_->body().data_type());
    interface_blob_conf->set_is_dynamic(is_dynamic());
    interface_blob_conf->set_is_tensor_list(is_tensor_list());
    if (batch_axis_->has_value()) {
      interface_blob_conf->mutable_batch_axis()->CopyFrom(*batch_axis_);
    }
  }

 private:
  std::shared_ptr<cfg::OptInt64> batch_axis_;
  std::shared_ptr<cfg::BlobDescProto> blob_desc_;
  std::string logical_blob_name_;
  std::shared_ptr<Shape> shape_;
};

class OpArgParallelAttribute {
 public:
  OpArgParallelAttribute(const std::shared_ptr<ParallelDesc>& parallel_desc,
                         const std::shared_ptr<cfg::SbpParallel>& sbp_parallel,
                         const std::shared_ptr<cfg::OptMirroredParallel>& opt_mirrored_parallel)
      : parallel_desc_(parallel_desc),
        sbp_parallel_(sbp_parallel),
        opt_mirrored_parallel_(opt_mirrored_parallel) {
    hash_ = Hash();
  }

  OpArgParallelAttribute(const OpArgParallelAttribute& op_arg_para_attr) = default;
  virtual ~OpArgParallelAttribute() = default;

  std::shared_ptr<ParallelDesc> parallel_desc_symbol() const { return parallel_desc_; }

  std::shared_ptr<cfg::SbpParallel> sbp_parallel() const { return sbp_parallel_; }

  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel() const {
    return opt_mirrored_parallel_;
  }

  bool is_mirrored() const { return opt_mirrored_parallel_->has_mirrored_parallel(); }

  std::size_t _Hash() const { return hash_; }

  bool operator==(const OpArgParallelAttribute& other) const {
    return (*parallel_desc_ == *other.parallel_desc_symbol())
           && (*sbp_parallel_ == *other.sbp_parallel())
           && (*opt_mirrored_parallel_ == *other.opt_mirrored_parallel());
  }

  void Assign(const std::shared_ptr<OpArgParallelAttribute>& other) {
    parallel_desc_ = other->parallel_desc_symbol();
    sbp_parallel_ = other->sbp_parallel();
    opt_mirrored_parallel_ = other->opt_mirrored_parallel();
    hash_ = other->_Hash();
  }

  void DumpToInterfaceBlobConf(std::shared_ptr<cfg::InterfaceBlobConf> interface_blob_conf) {
    if (sbp_parallel_->has_split_parallel()) {
      interface_blob_conf->mutable_split_axis()->set_value(sbp_parallel_->split_parallel().axis());
    } else {
      interface_blob_conf->clear_split_axis();
    }
  }

  std::string ToString() const {
    return std::string("\nparallel_desc_symbol: ") + parallel_desc_->parallel_conf().DebugString()
           + "\nsbp_parallel: " + sbp_parallel_->DebugString()
           + "\nopt_mirrored_parallel: " + opt_mirrored_parallel_->DebugString() + "\n";
  }

 protected:
  std::size_t Hash() const {
    std::size_t sbp_hash = 0;
    if (!opt_mirrored_parallel_->has_mirrored_parallel()) {
      sbp_hash ^= std::hash<cfg::SbpParallel>()(*sbp_parallel_);
    }
    return sbp_hash ^ (std::hash<ParallelDesc>()(*parallel_desc_))
           ^ (std::hash<cfg::OptMirroredParallel>()(*opt_mirrored_parallel_));
  }

 private:
  std::shared_ptr<ParallelDesc> parallel_desc_;
  std::shared_ptr<cfg::SbpParallel> sbp_parallel_;
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel_;
  std::size_t hash_;
};

}  // namespace compatible_py

}  // namespace oneflow

namespace std {

template<>
struct hash<::oneflow::compatible_py::OpArgParallelAttribute> {
  std::size_t operator()(const ::oneflow::compatible_py::OpArgParallelAttribute& s) const {
    return s._Hash();
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_ARG_UTIL_H_
