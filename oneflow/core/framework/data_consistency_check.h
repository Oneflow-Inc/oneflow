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
#ifndef ONEFLOW_CORE_FRAMEWORK_DATA_CONSISTENCY_CHECK_H_
#define ONEFLOW_CORE_FRAMEWORK_DATA_CONSISTENCY_CHECK_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

template<typename T>
Maybe<void> DataConsistencyCheck(const void* buffer_ptr, size_t elem_cnt,
                                 Symbol<ParallelDesc> placement);

#define MAKE_SWITCH_ENTRY(func_name, dtype) func_name<dtype>
DEFINE_STATIC_SWITCH_FUNC(Maybe<void>, DataConsistencyCheck, MAKE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ));

#undef MAKE_SWITCH_ENTRY
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_DATA_CONSISTENCY_CHECK_H_
