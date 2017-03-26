#ifndef _COMPUTE_TASK_DAG_H_
#define _COMPUTE_TASK_DAG_H_
#include<string>
#include <vector>
#include "dag/task_dag.h"
#include "path/path_share_policy.h"

/*
Actually cover kDataTask and kComputeTask
*/
namespace oneflow {
template <typename Dtype>
class ComputeTaskDag : public TaskDag<Dtype> {
 public:
  using TaskDag<Dtype>::GetOpNode;
  using TaskDag<Dtype>::GetDataNode;
  using TaskDag<Dtype>::AddOpNode;
  using TaskDag<Dtype>::AddDataNode;
  using TaskDag<Dtype>::GetSucceedingOpNodeNamesOfDataNode;
  using TaskDag<Dtype>::AddBackwardOpNode;
  using TaskDag<Dtype>::AddBlobsToConsumedRegisterInfo;
  using TaskDag<Dtype>::name_;
  using TaskDag<Dtype>::dag_builder_;
  using TaskDag<Dtype>::type_;
  using TaskDag<Dtype>::AddBlobsToConsumedRegisterInfoCrossPath;
  using TaskDag<Dtype>::AddEdges;
 
  using DNode = DataNode<BlobMeta>;
  using TaskDag<Dtype>::IsFirstOpNode;
  using TaskDag<Dtype>::AddBlobsToProducedRegisterInfo;

  using TaskDag<Dtype>::task_id_;
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

  ComputeTaskDag(const DagBuilder<Dtype>& path, TaskType type,
    int32_t task_id, PathType path_type, const std::string& actor_name,
    bool is_forward);
  ~ComputeTaskDag();

  void AddProducedRegisterInfos() override;
  void AddConsumedRegisterInfosInPath() override;

  // Get the input blobs' names in LogicalDag.
  std::vector<std::string> GetInputLogicalBlobs() const;
  // Get the output blobs' names in LogicalDag.
  std::vector<std::string> GetOutputLogicalBlobs() const;

  // The following overridden function work for the specific case: kDataPath
  // consumes the model from kModelUpdatePath.
  RegisterInfo CompleteConsumedRegisterInfoCrossPath(
    RegisterType consumed_register_type, int64_t produced_group_id) override;

private:
  void BuildForward() override;
  void BuildFromLayerSet();
  void BuildFromLayer(const std::string& layer_name);

  void AddCopyD2DLayer();
  std::string BuildCopyProtoString(int32_t blob_num) const;
  std::vector<std::string> BuildCopyInputBlobNames(
    const std::vector<std::string>& task_blobs) const;

  // For kComputeTask, if some of its blobs need splitting
  std::vector<std::string> task_blobs_need_split_;
  // From the blob needed split to all the consumers of this blob
  std::unordered_map<std::string, std::vector<std::string>> task_blob_to_consumers_;
  // From the blob needed split to its producer;
  std::unordered_map<std::string, std::string> task_blob_to_producer_;
  // <blob_name, <consumer_layer_name, layer_blob_name_for_consumer_layer>>
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
    task_blob_to_consumer_to_layer_blob_;

  void AddSplitLayersIfNecessary();
  void CollectBlobsNeedSplit();
  void RemoveExistingEdges();
  void InsertSplitLayers();
  void InsertSplitLayer(const std::string& task_blob_split);
  void UpdateBlobToConsumer(
    const std::string& dag_blob_name,
    const std::string& consumer_layer_name,
    const std::string& layer_blob_name);

  void BuildBackward() override;

  std::string RectifyProtoStrForModelParallelism(const std::string& layer_name,
    const std::string& layer_type, const std::string& param_proto) const;

  void RegisterNonInputOutputBlobs() override;

  void ForwardAddProducedRegisterInfos();
  void BackwardAddProducedRegisterInfos();

  // TODO(jiyuan): use polymorphism instead.
  void DataPathForwardAddProducedRegisterInfos();
  void ModelUpdatePathForwardAddProducedRegisterInfos();
  void ModelLoadPathForwardAddProducedRegisterInfos();
  void ModelStorePathForwardAddProducedRegisterInfos();

  void ForwardAddConsumedRegisterInfoInPath();
  void BackwardAddConsumedRegisterInfoInPath();

  virtual void ForwardSetup() override;
  void ForwardSetupPrepareDataTask();

  void AddProducedRegisterInfoContainBlobsAcrossPath(
    std::shared_ptr<TaskDag<Dtype>> consumer_task_dag,
    int64_t group_id,
    BlobType blob_type);

  void AddEnvelopeBlobsToProducedRegisterInfoAcrossPath(
    const std::string& op_name,
    const std::vector<std::string>& layer_vars,
    RegisterInfo* register_info,
    bool model_flag);

  // Build task_blob name according the logical_blob name
  std::string build_task_blob_from_logical_blob(
    const std::string& logical_blob) const;

  std::string build_task_blob_from_layer_blob(
    const std::string& layer_blob) const;

  ComputeTaskDag(const ComputeTaskDag& other) = delete;
  ComputeTaskDag operator=(const ComputeTaskDag& other) = delete;
};
}  // namespace oneflow
#endif  // _COMPUTE_TASK_DAG_H_
