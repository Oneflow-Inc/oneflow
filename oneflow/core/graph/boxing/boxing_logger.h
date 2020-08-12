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
#ifndef ONEFLOW_CORE_GRAPH_BOXING_LOGGER_H_
#define ONEFLOW_CORE_GRAPH_BOXING_LOGGER_H_

#include <memory>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

class BoxingLogger;
std::unique_ptr<BoxingLogger> CreateBoxingLogger();

class BoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingLogger);
  BoxingLogger() = default;
  virtual ~BoxingLogger() {
    if (log_stream_ != nullptr) log_stream_->Flush();
  }
  Maybe<void> SetLogStream(std::string path);
  Maybe<void> OutputLogStream(std::string log_line);
  virtual Maybe<void> BoxingLoggerSave(std::string log_line) = 0;

 private:
  std::unique_ptr<TeePersistentLogStream> log_stream_;
};

class NullBoxingLogger final : public BoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NullBoxingLogger);
  NullBoxingLogger() = default;
  ~NullBoxingLogger() override = default;
  Maybe<void> BoxingLoggerSave(std::string log_line) override { return Maybe<void>::Ok(); }
};

class CsvBoxingLogger final : public BoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CsvBoxingLogger);
  CsvBoxingLogger() = default;
  ~CsvBoxingLogger() override = default;
  Maybe<void> BoxingLoggerSave(std::string log_line) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_LOGGER_H_