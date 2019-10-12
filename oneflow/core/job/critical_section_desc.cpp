#include "oneflow/core/job/critical_section_desc.h"

namespace oneflow {

void CriticalSectionDesc::AddCriticalSection(std::unique_ptr<CriticalSection>&& critical_section) {
  CHECK_EQ(inited_, false);
  critical_sections_.emplace_back(std::move(critical_section));
}

void CriticalSectionDesc::Done() {
  CHECK_EQ(inited_, false);
  UpdateJobId2CriticalSectionIds();
  UpdateJobId2TotalJobCriticalSectionId();
  UpdateCriticalSectionIds2IntersectingIds();
  CHECK_EQ(job_id2critical_section_ids_.size(), job_id2total_job_critical_section_id_.size());
  CHECK_EQ(critical_sections_.size(), critical_section_id2intersecting_ids_.size());
  inited_ = true;
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
