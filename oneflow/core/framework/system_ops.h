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
#ifndef ONEFLOW_CORE_FRAMEWORK_SYSTEM_OPS_H_
#define ONEFLOW_CORE_FRAMEWORK_SYSTEM_OPS_H_

#include "oneflow/core/framework/op_base.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace schema {

class CastToGlobalOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override;
  const HashSet<std::string>& AttrNames() const override;

 public:
  Shape shape;
  DataType dtype;
};

class SelectTopNOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override;
  const HashSet<std::string>& AttrNames() const override;

 public:
  int32_t top_n;
};

class FeedInputOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override;
};

class FetchOutputOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override;
};

class FeedVariableOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override;
  const HashSet<std::string>& AttrNames() const override;

 public:
  double _l2;
};

class ImageDecoderRandomCropResizeOp : public OpBase {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override;
  const HashSet<std::string>& AttrNames() const override;

 public:
  int64_t target_width;
  int64_t target_height;
  int64_t num_workers;
  int64_t max_num_pixels;
  int64_t warmup_size;
  int64_t seed;
  int64_t num_attempts;
  float random_area_min;
  float random_area_max;
  float random_aspect_ratio_min;
  float random_aspect_ratio_max;
};

}  // namespace schema
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYSTEM_OPS_H_
