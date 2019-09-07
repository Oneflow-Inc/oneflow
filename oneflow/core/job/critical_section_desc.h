#ifndef ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_
#define ONEFLOW_CORE_JOB_CRITICAL_SECTION_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/critical_section.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class CriticalSectionDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionDesc);
  ~CriticalSectionDesc() = default;

  void AddCriticalSection(std::unique_ptr<CriticalSection>&&);
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
