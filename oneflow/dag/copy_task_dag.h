#ifndef _COPY_TASK_DAG_H_
#define _COPY_TASK_DAG_H_
#include <string>
#include <cstdint>
#include "dag/task_dag.h"

namespace oneflow {
template <typename Dtype>
class CopyTaskDag : public TaskDag<Dtype> {
 public:
  using TaskDag<Dtype>::GetOpNode;
  using TaskDag<Dtype>::GetDataNode;
  using TaskDag<Dtype>::AddOpNode;
  using TaskDag<Dtype>::name_;
  using TaskDag<Dtype>::dag_builder_;
  using TaskDag<Dtype>::type_;
  using TaskDag<Dtype>::AddBlobsToConsumedRegisterInfoCrossPath;
  using TaskDag<Dtype>::AddBlobsToConsumedRegisterInfo;
  using TaskDag<Dtype>::AddEdges;
  using TaskDag<Dtype>::AddBlobsToProducedRegisterInfo;
  using TaskDag<Dtype>::task_blob_from_layer_blob;
  using TaskDag<Dtype>::AddDataNode;

  using DNode = DataNode<BlobMeta>;

  using TaskDag<Dtype>::task_id_;
  using TaskDag<Dtype>::is_h2d_;
  using TaskDag<Dtype>::is_forward_;
  using TaskDag<Dtype>::data_name_to_node_;
  using TaskDag<Dtype>::is_placeholder_;

  using TaskDag<Dtype>::path_type_;
  using TaskDag<Dtype>::blob_info_manager_;
  using TaskDag<Dtype>::register_info_manager_;
  using TaskDag<Dtype>::null_filter_;
  using TaskDag<Dtype>::op_name_to_node_;
  using TaskDag<Dtype>::GetFirstOpNames;
  using TaskDag<Dtype>::GetImmediateProducerNamesInPath; 



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
}  // namespace oneflow
#endif  // _COPY_TASK_DAG_H_
