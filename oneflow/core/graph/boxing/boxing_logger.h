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
#include "glog/logging.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_status_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

class BoxingLog {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingLog);
  BoxingLog() = default;
  virtual ~BoxingLog() = default;

  virtual void Log(const SubTskGphBuilderStatus& status) = 0;
};

class NullBoxingLog final : public BoxingLog {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NullBoxingLog);
  NullBoxingLog() = default;
  ~NullBoxingLog() override = default;

  void Log(const SubTskGphBuilderStatus& status) override{};
};

class CsvBoxingLog final : public BoxingLog {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CsvBoxingLog);
  CsvBoxingLog() = delete;
  CsvBoxingLog(std::string path);
  ~CsvBoxingLog() override;

  void Log(const SubTskGphBuilderStatus& status) override;

 private:
  std::unique_ptr<TeePersistentLogStream> log_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_LOGGER_H_
