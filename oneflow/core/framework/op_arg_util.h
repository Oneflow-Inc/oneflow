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
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace compatible_py {

class OpArgBlobAttribute {
 public:
  OpArgBlobAttribute(const std::shared_ptr<BlobDescProto>& blob_desc,
                     const std::string& logical_blob_name);

  OpArgBlobAttribute(const OpArgBlobAttribute& op_arg_blob_attr) = default;
  OpArgBlobAttribute(OpArgBlobAttribute&& op_arg_blob_attr) = delete;
  OpArgBlobAttribute& operator=(const OpArgBlobAttribute&) = delete;
  OpArgBlobAttribute& operator=(OpArgBlobAttribute&&) = delete;
  virtual ~OpArgBlobAttribute() = default;

  std::shared_ptr<BlobDescProto> blob_desc() const;

  std::shared_ptr<Shape> shape() const;

  std::string logical_blob_name() const;

  DataType get_dtype() const;

  bool is_dynamic() const;

  bool operator==(const OpArgBlobAttribute& other) const;

  std::shared_ptr<OpArgBlobAttribute> GetPhysicalOpArgBlobAttr(int64_t split_axis,
                                                               int64_t parallel_num,
                                                               int64_t parallel_id) const;

 private:
  std::shared_ptr<BlobDescProto> blob_desc_;
  std::string logical_blob_name_;
  std::shared_ptr<Shape> shape_;
};

}  // namespace compatible_py

}  // namespace oneflow

namespace std {

template<>
struct hash<::oneflow::OptMirroredParallel> {
  std::size_t operator()(const ::oneflow::OptMirroredParallel& x) const {
    return std::hash<bool>()(x.has_mirrored_parallel());
  }
};

}  // namespace std

namespace oneflow {

namespace compatible_py {

class OpArgParallelAttribute {
 public:
  OpArgParallelAttribute(const std::shared_ptr<ParallelDesc>& parallel_desc,
                         const std::shared_ptr<SbpParallel>& sbp_parallel,
                         const std::shared_ptr<OptMirroredParallel>& opt_mirrored_parallel);

  OpArgParallelAttribute(const OpArgParallelAttribute& op_arg_para_attr) = default;
  OpArgParallelAttribute(OpArgParallelAttribute&& op_arg_blob_attr) = delete;
  OpArgParallelAttribute& operator=(const OpArgParallelAttribute&) = delete;
  OpArgParallelAttribute& operator=(OpArgParallelAttribute&&) = delete;
  virtual ~OpArgParallelAttribute() = default;

  std::shared_ptr<ParallelDesc> parallel_desc_symbol() const;

  std::shared_ptr<SbpParallel> sbp_parallel() const;

  std::shared_ptr<OptMirroredParallel> opt_mirrored_parallel() const;

  bool is_mirrored() const;

  std::size_t _Hash() const;

  bool operator==(const OpArgParallelAttribute& other) const;

  void Assign(const std::shared_ptr<OpArgParallelAttribute>& other);

  std::string ToString() const;

 protected:
  std::size_t Hash() const {
    std::size_t sbp_hash = 0;
    if (!opt_mirrored_parallel_->has_mirrored_parallel()) {
      sbp_hash ^= std::hash<SbpParallel>()(*sbp_parallel_);
    }
    return sbp_hash ^ (std::hash<ParallelDesc>()(*parallel_desc_))
           ^ (std::hash<OptMirroredParallel>()(*opt_mirrored_parallel_));
  }

 private:
  std::shared_ptr<ParallelDesc> parallel_desc_;
  std::shared_ptr<SbpParallel> sbp_parallel_;
  std::shared_ptr<OptMirroredParallel> opt_mirrored_parallel_;
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
