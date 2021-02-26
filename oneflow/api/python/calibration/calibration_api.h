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
#ifndef ONEFLOW_API_PYTHON_CALIBRATION_CALIBRATION_API_H_
#define ONEFLOW_API_PYTHON_CALIBRATION_CALIBRATION_API_H_

#include "oneflow/api/python/calibration/calibration.h"

inline void CacheInt8Calibration() { return oneflow::CacheInt8Calibration().GetOrThrow(); }

inline void WriteInt8Calibration(const std::string& path) {
  return oneflow::WriteInt8Calibration(path).GetOrThrow();
}

#endif  // ONEFLOW_API_PYTHON_CALIBRATION_CALIBRATION_ENV_H_
