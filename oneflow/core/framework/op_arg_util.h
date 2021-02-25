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
#include "oneflow/core/operator/interface_blob_conf.cfg.h"
#include "oneflow/core/register/pod.cfg.h"
#include "oneflow/core/register/blob_desc.cfg.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"
#include "oneflow/core/job/parallel_signature.cfg.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace compatible_py {

class OpArgBlobAttribute {
 public:
  OpArgBlobAttribute(const std::shared_ptr<cfg::BlobDescProto>& blob_desc,
                     const std::string& logical_blob_name);

  OpArgBlobAttribute(const OpArgBlobAttribute& op_arg_blob_attr) = default;
  virtual ~OpArgBlobAttribute() = default;

  std::shared_ptr<cfg::BlobDescProto> blob_desc() const;

  std::shared_ptr<Shape> shape() const;

  std::string logical_blob_name() const;

  cfg::DataType get_dtype() const;

  bool is_tensor_list() const;

  bool is_dynamic() const;

  bool operator==(const OpArgBlobAttribute& other) const;

  std::shared_ptr<OpArgBlobAttribute> GetPhysicalOpArgBlobAttr(int64_t split_axis,
                                                               int64_t parallel_num,
                                                               int64_t parallel_id) const;

  void DumpToInterfaceBlobConf(std::shared_ptr<cfg::InterfaceBlobConf> interface_blob_conf) const;

  void DumpToOpNodeSignature(std::string bn_in_op,
                             std::shared_ptr<cfg::OpNodeSignature> op_node_signature) const;

 private:
  std::shared_ptr<cfg::BlobDescProto> blob_desc_;
  std::string logical_blob_name_;
  std::shared_ptr<Shape> shape_;
};

class OpArgParallelAttribute {
 public:
  OpArgParallelAttribute(const std::shared_ptr<ParallelDesc>& parallel_desc,
                         const std::shared_ptr<cfg::SbpParallel>& sbp_parallel,
                         const std::shared_ptr<cfg::OptMirroredParallel>& opt_mirrored_parallel);

  OpArgParallelAttribute(const OpArgParallelAttribute& op_arg_para_attr) = default;
  virtual ~OpArgParallelAttribute() = default;

  std::shared_ptr<ParallelDesc> parallel_desc_symbol() const;

  std::shared_ptr<cfg::SbpParallel> sbp_parallel() const;

  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel() const;

  bool is_mirrored() const;

  std::size_t _Hash() const;

  bool operator==(const OpArgParallelAttribute& other) const;

  void Assign(const std::shared_ptr<OpArgParallelAttribute>& other);

  void DumpToInterfaceBlobConf(std::shared_ptr<cfg::InterfaceBlobConf> interface_blob_conf) const;

  void DumpToOpNodeSignature(std::string bn_in_op,
                             std::shared_ptr<cfg::OpNodeSignature> op_node_signature) const;

  std::string ToString() const;

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

Maybe<OpArgBlobAttribute> GetOpArgBlobAttribute(const OpAttribute& op_attribute,
                                                const std::string& bn_in_op);

Maybe<OpArgParallelAttribute> GetOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol, const OpAttribute& op_attribute,
    const std::string& bn_in_op);

Maybe<OpArgParallelAttribute> MakeMirroredOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol);

Maybe<OpArgParallelAttribute> MakeBroadcastOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol);

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
