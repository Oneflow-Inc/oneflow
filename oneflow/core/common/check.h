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

// The functions in this header file are used to replace `CHECK` and `LOG(FATAL)` macros of glog
// in those header files included by oneflow/core/common/throw.h, so those header files
// do not need to include <glog/logging.h>, and we can undef CHECK series macro of
// glog in oneflow/core/common/throw.h and use another impl instead with less modification.
namespace oneflow {
void GLOGCHECK(bool);
void GLOGLOGFATAL(const char*);
}  // namespace oneflow