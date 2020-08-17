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
#include "oneflow/core/graph/boxing/boxing_log.h"
#include <cstdint>
#include <string>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

#define OF_BOXING_LOGGER_CSV_COLNUM_NAME_FIELD                   \
  "src_op_name,dst_op_name,src_parallel_conf,dst_parallel_conf," \
  "src_sbp_conf,dst_sbp_conf,lbi,dtype,data_shape,comment\n"

std::string SbpParallelToString(const SbpParallel& sbp_parallel) {
  std::string serialized_sbp_parallel = "";
  if (sbp_parallel.has_broadcast_parallel()) {
    serialized_sbp_parallel = "B";
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    serialized_sbp_parallel = "P";

  } else if (sbp_parallel.has_split_parallel()) {
    serialized_sbp_parallel = "S(" + std::to_string(sbp_parallel.split_parallel().axis()) + ")";
  } else {
    UNIMPLEMENTED();
  }
  return serialized_sbp_parallel;
}

std::string ParallelDescToString(const ParallelDesc& parallel_desc) {
  std::string serialized_parallel_desc;
  serialized_parallel_desc += "device_type: " + DeviceType_Name(parallel_desc.device_type()) + " ";
  serialized_parallel_desc += "device_name: ";
  for (int64_t machine_id : parallel_desc.sorted_machine_ids()) {
    serialized_parallel_desc += std::to_string(machine_id) + ": ";
    int64_t min_id = parallel_desc.sorted_dev_phy_ids(machine_id).front();
    int64_t max_id = parallel_desc.sorted_dev_phy_ids(machine_id).back();
    serialized_parallel_desc += std::to_string(min_id) + "-" + std::to_string(max_id);
    serialized_parallel_desc += " ";
  }
  serialized_parallel_desc.pop_back();
  return serialized_parallel_desc;
}

std::string GetBlobInfo4LogicalBlobDesc(const BlobDesc& logical_blob_desc) {
  std::string blob_desc_info = "dtype: ";
  blob_desc_info += DataType_Name(logical_blob_desc.data_type()) + ",";
  auto shape_info = logical_blob_desc.shape().ToString();
  StringReplace(&shape_info, ',', ' ');
  blob_desc_info += " shape: " + shape_info;
  return blob_desc_info;
}

std::string SubTskGphBuilderStatusToCsvLine(const SubTskGphBuilderStatus& status) {
  std::string serialized_status("");
  serialized_status += status.src_op_name() + ",";
  serialized_status += status.dst_op_name() + ",";
  serialized_status += ParallelDescToString(status.src_parallel_desc()) + ",";
  serialized_status += ParallelDescToString(status.dst_parallel_desc()) + ",";
  serialized_status += SbpParallelToString(status.src_sbp_parallel()) + ",";
  serialized_status += SbpParallelToString(status.dst_sbp_parallel()) + ",";
  serialized_status += GenLogicalBlobName(status.lbi()) + ",";
  serialized_status += GetBlobInfo4LogicalBlobDesc(status.logical_blob_desc()) + ",";
  if (status.comment() == std::string("")) {
    serialized_status += status.builder_name() + ",";
  } else {
    serialized_status += status.builder_name() + ":" + status.comment();
  }
  serialized_status += "\n";
  return serialized_status;
}

}  // namespace

CsvBoxingLog::CsvBoxingLog(std::string path) {
  log_stream_ = TeePersistentLogStream::Create(path);
  log_stream_ << OF_BOXING_LOGGER_CSV_COLNUM_NAME_FIELD;
}

CsvBoxingLog::~CsvBoxingLog() { log_stream_->Flush(); }

void CsvBoxingLog::Log(const SubTskGphBuilderStatus& status) {
  log_stream_ << SubTskGphBuilderStatusToCsvLine(status);
}

}  // namespace oneflow
