#ifndef _COPY_TASK_DAG_H_
#define _COPY_TASK_DAG_H_
#include <string>
#include <cstdint>
#include "dag/task_dag.h"

namespace caffe {
template <typename Dtype>
class CopyTaskDag : public TaskDag<Dtype> {
 public:
  CopyTaskDag(const DagBuilder<Dtype>& path, TaskType type,
    int32_t task_id, PathType path_type, const std::string& actor_name,
    bool is_forward);
  ~CopyTaskDag();

  void AddProducedRegisterInfos() override;
  void AddConsumedRegisterInfosInPath() override;

  RegisterInfo CompleteConsumedRegisterInfoCrossPath(
    RegisterType consumer_register_type,
    int64_t produced_group_id) override;

  RegisterInfo ReplaceProducedRegisterInfoCrossPath(
    RegisterType my_register_type,
    int64_t other_group_id) override;

 private:
  void BuildForward() override;

  void InitH2D();

  std::vector<std::string> GetLogicalBlobsNeedCopied() const;
  std::string BuildProtoString(int32_t blob_num) const;

  std::vector<std::string> BuildInputTaskBlobs(
    const std::vector<std::string>& logical_blobs_copied) const;
  std::vector<std::string> BuildOutputTaskBlobs(
    const std::vector<std::string>& logical_blobs_copied) const;

  CopyTaskDag(const CopyTaskDag& other) = delete;
  CopyTaskDag operator=(const CopyTaskDag& other) = delete;
};
}  // namespace caffe
#endif  // _COPY_TASK_DAG_H_
