#include "oneflow/core/job/critical_section_desc.h"

namespace oneflow {

void CriticalSectionDesc::AddCriticalSection(std::unique_ptr<CriticalSection>&& critical_section) {
  CHECK_EQ(inited_, false);
  critical_sections_.emplace_back(std::move(critical_section));
}

void CriticalSectionDesc::Done() {
  CHECK_EQ(inited_, false);
  UpdateJobId2CriticalSectionIndexes();
  UpdateTotalJobCriticalSectionIndexes();
  UpdateCriticalSectionIndexes2IntersectingIndexes();
  CHECK_EQ(job_id2critical_section_indexes_.size(), total_job_critical_section_indexes_.size());
  CHECK_EQ(critical_sections_.size(), critical_section_index2intersecting_indexes_.size());
  inited_ = true;
}

const CriticalSection& CriticalSectionDesc::GetCriticalSectionByIndex(int64_t index) const {
  CHECK(inited_);
  return *critical_sections_.at(index);
}

CriticalSection* CriticalSectionDesc::MutCriticalSectionByIndex(int64_t index) const {
  CHECK_EQ(inited_, false);
  return critical_sections_.at(index).get();
}

const std::vector<int64_t>& CriticalSectionDesc::CriticalSectionIndexes4JobId(int64_t idx) const {
  CHECK(inited_);
  return job_id2critical_section_indexes_.at(idx);
}

const HashSet<int64_t>& CriticalSectionDesc::GetIntersectingCriticalSectionIndexes(
    int64_t idx) const {
  CHECK(inited_);
  return critical_section_index2intersecting_indexes_.at(idx);
}

void CriticalSectionDesc::UpdateJobId2CriticalSectionIndexes() {
  CHECK_EQ(inited_, false);
  job_id2critical_section_indexes_.resize(critical_sections_.size());
  int64_t max_job_id = -1;
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    const auto& critical_section = *critical_sections_.at(i);
    int64_t job_id = critical_section.critical_section_id().job_id();
    job_id2critical_section_indexes_[job_id].push_back(i);
    max_job_id = std::max(max_job_id, job_id);
  }
  job_id2critical_section_indexes_.resize(max_job_id + 1);
}

void CriticalSectionDesc::UpdateTotalJobCriticalSectionIndexes() {
  CHECK_EQ(inited_, false);
  HashSet<int64_t> unique_check;
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    const auto& critical_section = *critical_sections_.at(i);
    if (critical_section.critical_section_type() == kTotalJobCriticalSection) {
      CHECK(unique_check.emplace(critical_section.critical_section_id().job_id()).second);
      total_job_critical_section_indexes_.push_back(i);
    }
  }
}

void CriticalSectionDesc::UpdateCriticalSectionIndexes2IntersectingIndexes() {
  CHECK_EQ(inited_, false);
  critical_section_index2intersecting_indexes_.resize(critical_sections_.size());
  HashMap<int64_t, HashSet<int64_t>> mem_block_id2critical_section_indexes;
  FOR_RANGE(int64_t, i, 0, critical_sections_.size()) {
    for (int64_t mem_block_id : critical_sections_.at(i)->mem_block_id()) {
      mem_block_id2critical_section_indexes[mem_block_id].insert(i);
    }
  }
  for (const auto& pair : mem_block_id2critical_section_indexes) {
    for (int64_t first_indexes : pair.second) {
      for (int64_t second_indexes : pair.second) {
        if (first_indexes != second_indexes) {
          critical_section_index2intersecting_indexes_[first_indexes].insert(second_indexes);
        }
      }
    }
  }
}

}  // namespace oneflow
