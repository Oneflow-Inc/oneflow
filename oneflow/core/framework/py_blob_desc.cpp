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
#include "oneflow/core/framework/py_blob_desc.h"

namespace oneflow {

namespace compatible_py {

BlobDesc::BlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                   const std::shared_ptr<Distribute>& distribute)
    : lbi_(lbi), distribute_(distribute) {
  lbn_ = lbi->op_name() + "/" + lbi->blob_name();
}

std::shared_ptr<cfg::LogicalBlobId> BlobDesc::lbi() const { return lbi_; }
std::string BlobDesc::logical_blob_name() const { return lbn_; }
std::string BlobDesc::op_name() const { return lbi_->op_name(); }
std::string BlobDesc::blob_name() const { return lbi_->blob_name(); }
std::shared_ptr<Shape> BlobDesc::shape() const { UNIMPLEMENTED(); }
DataType BlobDesc::dtype() const { UNIMPLEMENTED(); }
std::shared_ptr<cfg::ParallelConf> BlobDesc::parallel_conf() const { UNIMPLEMENTED(); }

bool BlobDesc::is_dynamic() const { UNIMPLEMENTED(); }
bool BlobDesc::is_tensor_list() const { UNIMPLEMENTED(); }
std::shared_ptr<Distribute> BlobDesc::distribute() const { return distribute_; }
std::string BlobDesc::unique_name() const { return lbn_ + *CHECK_JUST(Distribute2Str()); }

void BlobDesc::set_distribute(const std::shared_ptr<Distribute> distribute) {
  distribute_ = distribute;
}

Maybe<std::string> BlobDesc::Distribute2Str() const {
  if (std::dynamic_pointer_cast<AutoDistribute>(distribute_)) {
    return std::string("");
  } else if (std::dynamic_pointer_cast<BroadcastDistribute>(distribute_)) {
    return std::string(":B");
  } else if (std::dynamic_pointer_cast<SplitDistribute>(distribute_)) {
    return std::string(":S") + std::to_string(distribute_->axis());
  } else {
    OF_UNIMPLEMENTED();
  }
  return std::string("");
}

}  // namespace compatible_py

}  // namespace oneflow
