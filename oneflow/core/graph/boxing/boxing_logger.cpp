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
#include "oneflow/core/graph/boxing/boxing_logger.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::unique_ptr<BoxingLogger> CreateBoxingLogger() {
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    return std::unique_ptr<BoxingLogger>(
        new CsvBoxingLogger(StrCat("boxing/logger/", GlobalJobDesc().job_id()) + ".csv"));
  } else {
    return std::unique_ptr<BoxingLogger>(new NullBoxingLogger());
  }
}

CsvBoxingLogger::CsvBoxingLogger(std::string path) {
  log_stream_ = TeePersistentLogStream::Create(path);
  log_stream_ << OF_BOXING_LOGGER_COLNUM_NAME_FIELD;
}

void CsvBoxingLogger::Log(const SubTskGphBuilderStatus& status) { 
  log_stream_ << status.ToString();
}

}  // namespace oneflow
