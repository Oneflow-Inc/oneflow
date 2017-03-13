#ifndef _BOXING_TASK_DAG_H_
#define _BOXING_TASK_DAG_H_
#include <string>
#include <unordered_map>
#include "dag/task_dag.h"
#include "dag/boxing_info.h"

namespace caffe {
template <typename Dtype>
class BoxingTaskDag : public TaskDag<Dtype> {
 public:
  BoxingTaskDag(const DagBuilder<Dtype>& dag_builder, TaskType type,
    int32_t task_id, PathType path_type, const std::string& actor_name,
    bool is_forward);
  ~BoxingTaskDag();

  int64_t GetImmediateProducedGroupIdInPath(
    const std::string& consumer_name) const override;

  void AddProducedRegisterInfos() override;
  void AddConsumedRegisterInfosInPath() override;

 private:
  void BuildForward() override;
  void SetBoxingProtoValue(
    bool is_in_boxing,
    const SegmentSegmentPair& segment_pair,
    const std::string& blob_name,
    const BoxingInfoElement& boxing_info_elem,
    BoxingProto *boxing_proto);

  void AddLayerForLogicalBlob(const std::string& boxing_pipe_name,
    const SegmentSegmentPair& segment_pair,
    const BoxingInfoElement& boxing_info_elem,
    const std::string& logical_blob);

  // Connect a blob named with |layer_blob| and |task_blob| to a RegisterInfo.
  // Add the blob to the RegisterInfo, 
  // which is the |idx|-th element of the vector std::vector<RegisterInfo>,
  // which is indexed by |second_segment| and has totally |output_num| elements.
  void UpdateRegisterInfo(
    const std::string& layer_blob,
    const std::string& task_blob,
    const std::string& second_segment,
    int32_t idx,
    int32_t output_num,
    std::unordered_map<std::string,
    std::vector<RegisterInfo>>* register_infos);

  void RegisterNonInputOutputBlobs() override;

  std::string build_layer_name(const std::string& pipe_name,
    const std::string& blob_name) const;

  BoxingTaskDag(const BoxingTaskDag& other) = delete;
  BoxingTaskDag operator=(const BoxingTaskDag& other) = delete;
};
}  // namespace caffe
#endif  // _BOXING_TASK_DAG_H_
