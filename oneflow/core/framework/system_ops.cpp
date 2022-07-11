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
#include "oneflow/core/framework/system_ops.h"
#include "oneflow/core/framework/attr_value.h"

namespace oneflow {
namespace schema {

Maybe<AttrVal> CastToGlobalOp::GetAttr(const std::string& attr_name) const {
  if (attr_name == "shape") {
    return CastAttrValue(&shape);
  } else if (attr_name == "dtype") {
    return CastAttrValue(&dtype);
  } else {
    return Error::RuntimeError() << "CastToGlobal op has no attribute named " << attr_name;
  }
}

const HashSet<std::string>& CastToGlobalOp::AttrNames() const {
  static HashSet<std::string> attr_names{"shape", "dtype"};
  return attr_names;
}

Maybe<AttrVal> SelectTopNOp::GetAttr(const std::string& attr_name) const {
  if (attr_name == "top_n") {
    return CastAttrValue(&top_n);
  } else {
    return Error::RuntimeError() << "SelectTopN op has no attribute named " << attr_name;
  }
}

const HashSet<std::string>& SelectTopNOp::AttrNames() const {
  static HashSet<std::string> attr_names{"top_n"};
  return attr_names;
}

Maybe<AttrVal> FeedInputOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "FeedInput op has no attribute named " << attr_name;
}

Maybe<AttrVal> FetchOutputOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "FetchOutput op has no attribute named " << attr_name;
}

Maybe<AttrVal> FeedVariableOp::GetAttr(const std::string& attr_name) const {
  if (attr_name == "_l2") {
    return CastAttrValue(&_l2);
  } else {
    return Error::RuntimeError() << "FeedVariable op has no attribute named " << attr_name;
  }
}

const HashSet<std::string>& FeedVariableOp::AttrNames() const {
  static HashSet<std::string> attr_names{"_l2"};
  return attr_names;
}

Maybe<AttrVal> ImageDecoderRandomCropResizeOp::GetAttr(const std::string& attr_name) const {
  if (attr_name == "target_width") {
    return CastAttrValue(&target_width);
  } else if (attr_name == "target_height") {
    return CastAttrValue(&target_height);
  } else if (attr_name == "num_workers") {
    return CastAttrValue(&num_workers);
  } else if (attr_name == "max_num_pixels") {
    return CastAttrValue(&max_num_pixels);
  } else if (attr_name == "warmup_size") {
    return CastAttrValue(&warmup_size);
  } else if (attr_name == "seed") {
    return CastAttrValue(&seed);
  } else if (attr_name == "num_attempts") {
    return CastAttrValue(&num_attempts);
  } else if (attr_name == "random_area_min") {
    return CastAttrValue(&random_area_min);
  } else if (attr_name == "random_area_max") {
    return CastAttrValue(&random_area_max);
  } else if (attr_name == "random_aspect_ratio_min") {
    return CastAttrValue(&random_aspect_ratio_min);
  } else if (attr_name == "random_aspect_ratio_max") {
    return CastAttrValue(&random_aspect_ratio_max);
  } else {
    return Error::RuntimeError() << "FeedVariable op has no attribute named " << attr_name;
  }
}

const HashSet<std::string>& ImageDecoderRandomCropResizeOp::AttrNames() const {
  static HashSet<std::string> attr_names{"target_width",
                                         "target_height",
                                         "num_workers",
                                         "max_num_pixels",
                                         "warmup_size",
                                         "seed",
                                         "num_attempts",
                                         "random_area_min",
                                         "random_area_max",
                                         "random_aspect_ratio_min",
                                         "random_aspect_ratio_max"};
  return attr_names;
}

}  // namespace schema
}  // namespace oneflow
