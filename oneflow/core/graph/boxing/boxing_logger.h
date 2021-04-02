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

#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_status_util.h"

namespace oneflow {

class BoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingLogger);
  BoxingLogger() = default;
  virtual ~BoxingLogger() = default;

  virtual void Log(const SubTskGphBuilderStatus& status, const std::string& src_op_name,
                   const std::string& dst_op_name, const ParallelDesc& src_parallel_desc,
                   const ParallelDesc& dst_parallel_desc,
                   const ParallelDistribution& src_parallel_distribution,
                   const ParallelDistribution& dst_parallel_distribution, const LogicalBlobId& lbi,
                   const BlobDesc& logical_blob_desc) = 0;
};

class NullBoxingLogger final : public BoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NullBoxingLogger);
  NullBoxingLogger() = default;
  ~NullBoxingLogger() override = default;

  void Log(const SubTskGphBuilderStatus& status, const std::string& src_op_name,
           const std::string& dst_op_name, const ParallelDesc& src_parallel_desc,
           const ParallelDesc& dst_parallel_desc,
           const ParallelDistribution& src_parallel_distribution,
           const ParallelDistribution& dst_parallel_distribution, const LogicalBlobId& lbi,
           const BlobDesc& logical_blob_desc) override{};
};

class CsvBoxingLogger final : public BoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CsvBoxingLogger);
  CsvBoxingLogger() = delete;
  CsvBoxingLogger(std::string path);
  ~CsvBoxingLogger() override;

  void Log(const SubTskGphBuilderStatus& status, const std::string& src_op_name,
           const std::string& dst_op_name, const ParallelDesc& src_parallel_desc,
           const ParallelDesc& dst_parallel_desc,
           const ParallelDistribution& src_parallel_distribution,
           const ParallelDistribution& dst_parallel_distribution, const LogicalBlobId& lbi,
           const BlobDesc& logical_blob_desc) override;

 private:
  std::unique_ptr<TeePersistentLogStream> log_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_LOGGER_H_
