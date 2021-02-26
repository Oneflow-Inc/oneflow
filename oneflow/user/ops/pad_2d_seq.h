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
#ifndef ONEFLOW_USER_OPS_PAD_2D_SEQ_H_
#define ONEFLOW_USER_OPS_PAD_2D_SEQ_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define PAD_2D_TYPE_SEQ                    \
  OF_PP_MAKE_TUPLE_SEQ("reflection_pad2d") \
  OF_PP_MAKE_TUPLE_SEQ("replication_pad2d")
}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_PAD_2D_SEQ_H_
