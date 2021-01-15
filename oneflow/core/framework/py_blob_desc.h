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
#ifndef ONEFLOW_CORE_FRAMEWORK_PY_BLOB_DESC_H_
#define ONEFLOW_CORE_FRAMEWORK_PY_BLOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/register/logical_blob_id.cfg.h"
#include "oneflow/core/framework/py_distribute.h"

namespace oneflow {

namespace compatible_py {

static int64_t INVALID_BATCH_AXIS = -22;
static int64_t INVALID_SPLIT_AXIS = -22;

class BlobDesc : public Tensor {
 public:
  BlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
           const std::shared_ptr<Distribute>& distribute)
      : lbi_(lbi), distribute_(distribute) {
    lbn_ = lbi->op_name() + "/" + lbi->blob_name();
  }

  BlobDesc(const BlobDesc& blob_desc) = default;
  virtual ~BlobDesc() override = default;

  virtual std::shared_ptr<cfg::LogicalBlobId> lbi() const override { return lbi_; }
  virtual std::string logical_blob_name() const override { return lbn_; }
  virtual std::string op_name() const override { return lbi_->op_name(); }
  virtual std::string blob_name() const override { return lbi_->blob_name(); }
  virtual std::shared_ptr<Shape> shape() const override { UNIMPLEMENTED(); }
  virtual DataType dtype() const override { UNIMPLEMENTED(); }
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const override { UNIMPLEMENTED(); }

  virtual int64_t batch_axis() const { UNIMPLEMENTED(); }
  virtual bool has_batch_axis() const { return batch_axis() != INVALID_BATCH_AXIS; }
  virtual bool is_dynamic() const { UNIMPLEMENTED(); }
  virtual bool is_tensor_list() const { UNIMPLEMENTED(); }
  virtual std::shared_ptr<Distribute> distribute() const { return distribute_; }
  virtual std::string unique_name() const { return lbn_ + *CHECK_JUST(Distribute2Str()); }

  void set_distribute(const std::shared_ptr<Distribute> distribute) { distribute_ = distribute; }

 protected:
  Maybe<std::string> Distribute2Str() const {
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

  std::shared_ptr<cfg::LogicalBlobId> lbi_;
  std::shared_ptr<Distribute> distribute_;
  std::string lbn_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PY_BLOB_DESC_H_
