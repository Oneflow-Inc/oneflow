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

static int64_t INVALID_SPLIT_AXIS = -22;

class BlobDesc : public Tensor {
 public:
  BlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
           const std::shared_ptr<Distribute>& distribute);

  BlobDesc(const BlobDesc& blob_desc) = default;
  virtual ~BlobDesc() override = default;

  virtual std::shared_ptr<cfg::LogicalBlobId> lbi() const override;
  virtual std::string logical_blob_name() const override;
  virtual std::string op_name() const override;
  virtual std::string blob_name() const override;
  virtual std::shared_ptr<Shape> shape() const override;
  virtual DataType dtype() const override;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;

  virtual bool is_dynamic() const;
  virtual bool is_tensor_list() const;
  virtual std::shared_ptr<Distribute> distribute() const;
  virtual std::string unique_name() const;

  void set_distribute(const std::shared_ptr<Distribute> distribute);

 protected:
  Maybe<std::string> Distribute2Str() const;

  std::shared_ptr<cfg::LogicalBlobId> lbi_;
  std::shared_ptr<Distribute> distribute_;
  std::string lbn_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PY_BLOB_DESC_H_
