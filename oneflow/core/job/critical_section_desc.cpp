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
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include <google/protobuf/text_format.h>
#include <cstdint>
#include <string>

namespace oneflow {

CriticalSection* CriticalSectionDesc::AddCriticalSection(int64_t job_id) {
  CHECK_EQ(inited_, false);
  auto critical_section = std::make_unique<CriticalSection>();
  CriticalSection* ret = critical_section.get();
  critical_section->set_job_id(job_id);
  critical_sections_.emplace_back(std::move(critical_section));
  return ret;
}

void CriticalSectionDesc::Done() {
  CHECK_EQ(inited_, false);
  UpdateJobId2CriticalSectionIds();
  UpdateJobId2TotalJobCriticalSectionId();
  UpdateCriticalSectionIds2IntersectingIds();
  CHECK_EQ(job_id2critical_section_ids_.size(), job_id2total_job_critical_section_id_.size());
  CHECK_EQ(critical_sections_.size(), critical_section_id2intersecting_ids_.size());
  inited_ = true;
  std::string all_output;
  int32_t i = 0;
  for (const auto& cs : critical_sections_) {
    all_output += "CriticalSection " + std::to_string(i) + "\n";
    std::string output;
    google::protobuf::TextFormat::PrintToString(*cs, &output);
    all_output += output;
    all_output += "\n";
    i++;
  }
  TeePersistentLogStream::Create("critical_section_desc")->Write(all_output);
}

const CriticalSection& CriticalSectionDesc::GetCriticalSection(int64_t critical_section_id) const {
  CHECK(inited_);
  return *critical_sections_.at(critical_section_id);
}

CriticalSection* CriticalSectionDesc::MutCriticalSection(int64_t critical_section_id) const {
  CHECK_EQ(inited_, false);
  return critical_sections_.at(critical_section_id).get();
}

const std::vector<int64_t>& CriticalSectionDesc::CriticalSectionIds4JobId(int64_t job_id) const {
  CHECK(inited_);
  return job_id2critical_section_ids_.at(job_id);
}

void CriticalSectionDesc::DumpCriticalSectionId2IntersectinIds(PbRpf<Int64List>* id2id_list) const {
  CHECK(inited_);
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    *id2id_list->Add()->mutable_value() = {critical_section_id2intersecting_ids_.at(i).begin(),
                                           critical_section_id2intersecting_ids_.at(i).end()};
  }
}

void CriticalSectionDesc::UpdateJobId2CriticalSectionIds() {
  CHECK_EQ(inited_, false);
  job_id2critical_section_ids_.resize(critical_sections_.size());
  int64_t max_job_id = -1;
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    const auto& critical_section = *critical_sections_.at(i);
    int64_t job_id = critical_section.job_id();
    job_id2critical_section_ids_[job_id].push_back(i);
    max_job_id = std::max(max_job_id, job_id);
  }
  job_id2critical_section_ids_.resize(max_job_id + 1);
}

void CriticalSectionDesc::UpdateJobId2TotalJobCriticalSectionId() {
  CHECK_EQ(inited_, false);
  HashSet<int64_t> unique_check;
  job_id2total_job_critical_section_id_.resize(critical_sections_.size());
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    const auto& critical_section = *critical_sections_.at(i);
    if (critical_section.has_total_job_critical_section()) {
      CHECK(unique_check.emplace(critical_section.job_id()).second);
      job_id2total_job_critical_section_id_.at(critical_section.job_id()) = i;
    }
  }
  job_id2total_job_critical_section_id_.resize(unique_check.size());
}

void CriticalSectionDesc::UpdateCriticalSectionIds2IntersectingIds() {
  CHECK_EQ(inited_, false);
  critical_section_id2intersecting_ids_.resize(critical_sections_.size());
  HashMap<int64_t, HashSet<int64_t>> mem_block_id2critical_section_ids;
  HashMap<int64_t, HashSet<int64_t>> chunk_id2critical_section_ids;
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    for (int64_t mem_block_id : critical_sections_.at(i)->mem_block_id()) {
      mem_block_id2critical_section_ids[mem_block_id].insert(i);
    }
    for (int64_t chunk_id : critical_sections_.at(i)->chunk_id()) {
      chunk_id2critical_section_ids[chunk_id].insert(i);
    }
  }
  for (const auto& pair : mem_block_id2critical_section_ids) {
    for (int64_t first_id : pair.second) {
      for (int64_t second_id : pair.second) {
        if (first_id != second_id) {
          critical_section_id2intersecting_ids_[first_id].insert(second_id);
        }
      }
    }
  }
  for (const auto& pair : chunk_id2critical_section_ids) {
    for (int64_t first_id : pair.second) {
      for (int64_t second_id : pair.second) {
        if (first_id != second_id) {
          critical_section_id2intersecting_ids_[first_id].insert(second_id);
        }
      }
    }
  }
}

}  // namespace oneflow
