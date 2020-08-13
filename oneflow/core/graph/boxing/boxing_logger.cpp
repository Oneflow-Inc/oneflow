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

namespace oneflow {

std::unique_ptr<BoxingLogger> CreateBoxingLogger() {
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    return std::unique_ptr<BoxingLogger>(new CsvBoxingLogger());
  } else {
    return std::unique_ptr<BoxingLogger>(new NullBoxingLogger());
  }
}

Maybe<void> BoxingLogger::SetLogStream(std::string path) {
  log_stream_ = TeePersistentLogStream::Create(path);
  return Maybe<void>::Ok();
}

Maybe<void> BoxingLogger::OutputLogStream(std::string log_line) {
  log_stream_ << log_line;
  return Maybe<void>::Ok();
}

Maybe<void> CsvBoxingLogger::SetLogStream(std::string path) {
  return BoxingLogger::SetLogStream(path);
}

Maybe<void> CsvBoxingLogger::BoxingLoggerSave(std::string log_line) {
  return OutputLogStream(log_line);
}

}  // namespace oneflow