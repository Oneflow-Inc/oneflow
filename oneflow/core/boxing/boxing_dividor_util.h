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
#ifndef ONEFLOW_CORE_BOXING_BOXING_DIVIDOR_UTIL_H_
#define ONEFLOW_CORE_BOXING_BOXING_DIVIDOR_UTIL_H_

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/boxing/boxing_dividor.h"

namespace oneflow {

extern Maybe<BoxingDividor> (*ReplaceInDeviceType)(DeviceType device_type);
extern Maybe<BoxingDividor> (*ReplaceOutDeviceType)(DeviceType device_type);
extern Maybe<BoxingDividor> (*FlattenInHierarchy)();
extern Maybe<BoxingDividor> (*UnflattenInHierarchy)();
extern Maybe<BoxingDividor> (*UnflattenOutHierarchy)();
extern Maybe<BoxingDividor> (*OutPlacementAndPartialSum)();
extern Maybe<BoxingDividor> (*InPlacementAndBroadcast)();
extern Maybe<BoxingDividor> (*OutPlacementAndBroadcast)();
extern Maybe<BoxingDividor> (*InPlacementAndSplit)(int64_t axis);
extern Maybe<BoxingDividor> (*OutPlacementAndSplit)(int64_t axis);
extern Maybe<BoxingDividor> (*InFirstDeviceAndAllBroadcast)();
extern Maybe<BoxingDividor> (*OutFirstDeviceAndAllBroadcast)();
extern Maybe<BoxingDividor> (*InPlacementAndRepeatFirstSbp)();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BOXING_BOXING_DIVIDOR_UTIL_H_
