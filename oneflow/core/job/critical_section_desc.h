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
#ifndef ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_
#define ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/critical_section.pb.h"

namespace oneflow {

class CriticalSectionDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionDesc);
  ~CriticalSectionDesc() = default;

  CriticalSection* AddCriticalSection(int64_t job_id);
  void Done();

  size_t CriticalSectionNum() const { return critical_sections_.size(); }
  const CriticalSection& GetCriticalSection(int64_t) const;
  CriticalSection* MutCriticalSection(int64_t) const;
  const std::vector<int64_t>& CriticalSectionIds4JobId(int64_t) const;
  void DumpCriticalSectionId2IntersectinIds(PbRpf<Int64List>* id2id_list) const;
  const std::vector<int64_t>& job_id2total_job_critical_section_id() const {
    return job_id2total_job_critical_section_id_;
  }

 private:
  friend class Global<CriticalSectionDesc>;
  CriticalSectionDesc() : inited_(false) {}
  void UpdateJobId2CriticalSectionIds();
  void UpdateJobId2TotalJobCriticalSectionId();
  void UpdateCriticalSectionIds2IntersectingIds();

  bool inited_;
  std::vector<std::unique_ptr<CriticalSection>> critical_sections_;
  std::vector<std::vector<int64_t>> job_id2critical_section_ids_;
  std::vector<int64_t> job_id2total_job_critical_section_id_;
  std::vector<HashSet<int64_t>> critical_section_id2intersecting_ids_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_
