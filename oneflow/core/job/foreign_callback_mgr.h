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
#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLACK_MGR_H_
#define ONEFLOW_CORE_JOB_FOREIGN_CALLACK_MGR_H_

namespace oneflow {

class ForeignCallback;

void RegisterForeignCallbackOnlyOnce(ForeignCallback* callback);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_CALLACK_MGR_H_
