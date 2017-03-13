#ifndef _BLOB_INFO_MANAGER_H_
#define _BLOB_INFO_MANAGER_H_
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_set>
#include "common/shape.h"

namespace caffe {
class BlobInfoManager {
public:
  BlobInfoManager() = default;
  ~BlobInfoManager() = default;

  void RegisterBlob(
    const std::string& layer_blob,
    const std::string& task_blob,
    const std::string& logical_blob,
    bool is_input);

  // Memorize that the produced |layer_blob| will locate at the register 
  // with |group_id| as its group_id.
  void AddProducedBlobToRegister(
    const std::string& layer_blob,
    int64_t group_id);
  // Revert the effect of |AddProducedBlobToRegister|.
  void RemoveProducedBlobFromRegister(
    const std::string& layer_blob,
    int64_t group_id);

  // Memorize that the consumed |layer_blob| will locates at the register
  // with |group_id| as its group_id, with |register_blob| as its name in the
  // register.
  void AddConsumedBlobToRegister(
    const std::string& layer_blob,
    const std::string& register_blob,
    int64_t group_id);

  void SetBlobShape(const std::string& task_blob, const Shape& shape);
  Shape GetBlobShape(const std::string& task_blob) const;

  bool IsProduced(const std::string& task_blob) const;

  // Specifically used in building kComputeTaskDag
  void EraseInputTaskBlob(const std::string& task_blob);
  void RemoveLayerAndTaskBlobPair(const std::string& layer_blob);

  std::vector<std::string> layer_blobs() const;
  std::vector<std::string> layer_blobs_in_execution() const;
  std::vector<std::string> input_task_blobs() const;
  std::vector<std::string> task_blobs() const;
  std::vector<std::string> produced_task_blobs() const;
  std::vector<std::string> consumed_task_blobs() const;

  int64_t group_id_of_task_blob(const std::string& task_blob) const;

  std::string logical_blob_from_task_blob(
    const std::string& task_blob) const;
  std::string task_blob_from_layer_blob(
    const std::string& layer_blob) const;
  std::string task_blob_from_logical_blob(
    int64_t group_id, const std::string& logical_blob) const;
  std::string register_blob_from_layer_blob(
    const std::string& layer_blob) const;
  std::string register_blob_from_task_blob(
    const std::string& task_blob) const;

private:
  using TaskBlobGroupIdMap = std::unordered_map<std::string, int64_t>;
  using BlobNameMap = std::unordered_map<std::string, std::string>;
  using LayerBlobTaskBlobMap = BlobNameMap;
  using TaskBlobLogicalBlobMap = BlobNameMap;
  using LogicalBlobTaskBlobMap = BlobNameMap;
  using LayerBlobRegisterBlobMap = BlobNameMap;
  using TaskBlobRegisterBlobMap = BlobNameMap;
  using TaskBlobProducerMap = BlobNameMap;

  std::vector<std::string> layer_blobs_;
  LayerBlobTaskBlobMap layer_blob_to_task_blob_;
  TaskBlobLogicalBlobMap task_blob_to_logical_blob_;
  LayerBlobRegisterBlobMap layer_blob_to_register_blob_;

  // Memorize all the layer_blobs which are once called as an input parameter 
  // of |AddProducedBlobToRegister| and |AddConsumedBlobToRegister|, indicating
  // that a particular layer_blob will be required in execution. Note that not
  // every layer_blob in |layer_blobs_| will be in |layer_blobs_in_execution_|.
  std::unordered_set<std::string> layer_blobs_in_execution_;
  // Map from a task_blob to its register_blob name. If the task_blob is
  // produced by another task, the register_blob name will be determined by its
  // task_blob name in the producer task.
  TaskBlobRegisterBlobMap task_blob_to_register_blob_;
  std::unordered_map<std::string, Shape> task_blob_to_shape_;
  std::unordered_set<std::string> input_task_blobs_;

  std::unordered_set<std::string> task_blobs_;
  std::unordered_set<std::string> produced_task_blobs_;
  std::unordered_set<std::string> consumed_task_blobs_;

  TaskBlobGroupIdMap task_blob_to_group_id_;
  TaskBlobGroupIdMap produced_task_blob_to_group_id_;
  TaskBlobGroupIdMap consumed_task_blob_to_group_id_;

  // In a task, there might be multiple task_blobs corresponding to a 
  // blob. However, within the range of a particular register, there is exactly
  // one task_blob correspondence to a logical_blob. Therefore, logical_blob 
  // could act as a key to find a unique task_blob correspondence.
  std::unordered_map<int64_t, LogicalBlobTaskBlobMap>
    produced_group_id_to_logical_task_map_;

  // Establish the correspondence from |logical_blob| to |task_blob| inside
  // |group_id|-th produced register. We ensure in each register, there is 
  // exactly one |task_blob| for each |logical_blob|.
  void AddLogicalBlobTaskBlobMap(const std::string& task_blob, int64_t group_id);
  void RemoveLogicalBlobTaskBlobMap(int64_t group_id);

  BlobInfoManager(const BlobInfoManager& other) = delete;
  BlobInfoManager& operator=(const BlobInfoManager& other) = delete;
};
}  // namespace caffe
#endif  // _BLOB_INFO_MANAGER_H_