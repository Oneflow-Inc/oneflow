#ifndef ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_
#define ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/critical_section.pb.h"

namespace oneflow {

class CriticalSectionDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionDesc);
  ~CriticalSectionDesc() = default;

  void AddCriticalSection(std::unique_ptr<CriticalSection>&&);
  void Done();
  const CriticalSection& GetCriticalSectionByIndex(int64_t) const;
  const std::vector<int64_t>& CriticalSectionIndexes4JobId(int64_t) const;
  const HashSet<int64_t>& GetIntersectingCriticalSectionIndexes(int64_t) const;

 private:
  friend class Global<CriticalSectionDesc>;
  CriticalSectionDesc() : critical_section_relations_updated_(false) {}
  void UpdateCriticalSectionRelations();

  bool critical_section_relations_updated_;
  std::vector<std::unique_ptr<CriticalSection>> critical_sections_;
  HashMap<int64_t, std::vector<int64_t>> job_id2critical_section_indexes_;
  std::vector<int64_t> total_job_critical_section_indexes_;
  HashMap<int64_t, HashSet<int64_t>> critical_section_index2intersecting_indexes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_
