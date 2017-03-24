#ifndef _NET_TASK_DAG_H_
#define _NET_TASK_DAG_H_
#include <string>
#include "dag/task_dag.h"

namespace oneflow {
template <typename Dtype>
class NetTaskDag : public TaskDag<Dtype> {
 public:
  NetTaskDag(const DagBuilder<Dtype>& path, TaskType type,
    int32_t task_id, PathType path_type, const std::string& actor_name,
    bool is_forward);
  ~NetTaskDag();

  void AddProducedRegisterInfos() override;
  void AddConsumedRegisterInfosInPath() override;

  bool forward_is_sender() const { return forward_is_sender_; }
 private:
  bool forward_is_sender_;
  std::string envelope_name_;
  using BlobFilter = std::function<bool(const std::string&)>;
  const BlobFilter is_envelope_ = [](const std::string& blob_var) {
    return strings::Contains(blob_var, "envelope");
  };
  const BlobFilter is_not_envelope_ = [](const std::string& blob_var) {
    return !strings::Contains(blob_var, "envelope");
  };

  void BuildForward() override;

  std::vector<std::string> GetLogicalBlobsNeedTransferred() const;
  std::string BuildProtoString(int32_t blob_num) const;
  std::vector<std::string> BuildInputTaskBlobs(
    const std::vector<std::string>& logical_blobs) const;
  std::vector<std::string> BuildOutputTaskBlobs(
    const std::vector<std::string>& logical_blobs) const;

  std::string BuildEnvelopeName(const std::vector<std::string>& blobs) const;

  void ForwardSetup() override;
  void ForwardSetupInNetTask();

  NetTaskDag(const NetTaskDag& other) = delete;
  NetTaskDag operator=(const NetTaskDag& other) = delete;
};
}  // namespace oneflow
#endif  // _NET_TASK_DAG_H_
