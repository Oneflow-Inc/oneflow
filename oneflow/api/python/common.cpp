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

#include "oneflow/api/python/common.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

Maybe<void> ParsingDeviceTag(const std::string& device_tag, std::string* device_name,
                             int* device_index) {
  std::string::size_type pos = device_tag.find(':');
  if (pos == std::string::npos) {
    *device_name = device_tag;
    *device_index = -1;
  } else {
    std::string index_str = device_tag.substr(pos + 1);
    CHECK_OR_RETURN(IsStrInt(index_str)) << "Invalid device " << device_tag;
    *device_name = device_tag.substr(0, pos);
    *device_index = std::stoi(index_str);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
