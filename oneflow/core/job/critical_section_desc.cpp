#include "oneflow/core/job/critical_section_desc.h"

namespace oneflow {

void CriticalSectionDesc::AddCriticalSection(std::unique_ptr<CriticalSection>&& critical_section) {
  critical_sections_.emplace_back(std::move(critical_section));
}

void CriticalSectionDesc::Done() { TODO(); }

const CriticalSection& CriticalSectionDesc::GetCriticalSectionByIndex(int64_t) const { TODO(); }

CriticalSection* CriticalSectionDesc::MutCriticalSectionByIndex(int64_t) const { TODO(); }

const std::vector<int64_t>& CriticalSectionDesc::CriticalSectionIndexes4JobId(int64_t) const {
  TODO();
}

const HashSet<int64_t>& CriticalSectionDesc::GetIntersectingCriticalSectionIndexes(int64_t) const {
  TODO();
}

void CriticalSectionDesc::UpdateCriticalSectionRelations() { TODO(); }

}  // namespace oneflow
