#ifndef _REGISTER_INFO_MANAGER_H_
#define _REGISTER_INFO_MANAGER_H_
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdint>
#include "dag/register_info.h"

namespace caffe {
class RegisterInfoManager {
public:
  using GroupIDConsumerIDMap = std::unordered_map<int64_t, std::vector<int32_t>>;
public:
  RegisterInfoManager() = default;
  ~RegisterInfoManager() = default;

  // For non-kBoxingTask, there is at most one RegisterInfo corresponding to a
  // RegisterType value.
  void AddProducedRegisterInfoForNonBoxingTask(
    const RegisterInfo& register_info);
  // Query the produced group_id by RegisterType value |type|. 
  int64_t GetProducedGroupIdForNonBoxingTask(RegisterType type) const;
  void RemoveProducedRegisterInfoForNonBoxingTask(RegisterType type,
    int64_t group_id);

  // kBoxingTask may generate multiple RegisterInfos with the same RegisterType
  // value. Those RegisterInfos need to be distinguished from each other. Here,
  // we use the RegisterInfo's consumer and its order in the RegisterInfos with 
  // the same consumers. The |consumer_segment| indicates the segment name who
  // generates the consumer tasks.
  void AddProducedRegisterInfoForBoxingTask(const RegisterInfo& register_info,
    const std::string& consumer_segment);
  // Query the produced group_id by |consumer_segment| (the segment name which
  // generates the consumer) and its |order|. 
  int64_t GetProducedGroupIdForBoxingTask(
    const std::string& consumer_segment, int32_t order) const;

  RegisterInfo CompleteProducedRegisterInfoCrossPath(
    RegisterType produced_register_type,
    const RegisterInfo& consumed_register_info);

  const RegisterInfo& GetProducedRegisterInfo(int64_t group_id) const;

  void SetProducedGroupSize(int64_t group_id, int32_t group_size);
  int32_t GetProducedGroupSize(int64_t group_id) const;

  void AddConsumerOfGroupId(int32_t consumer_id, int64_t group_id);

  std::vector<int64_t> GetProducedGroupIds() const;
  std::vector<int64_t> GetGroupIdsConsumedByOthers() const;

  std::vector<int32_t> GetConsumersOfGroupId(int64_t group_id) const;

  void AddConsumedGroupId(int64_t group_id);
  std::vector<int64_t> GetConsumedGroupIds() const;

private:
  std::unordered_map<int64_t, RegisterInfo> produced_group_id_to_register_info_;
  std::unordered_map<int64_t, int32_t> produced_group_id_to_group_size_;
  GroupIDConsumerIDMap group_id_to_consumer_ids_;

  // RegisterInfo produced by current TaskDag.
  // (1) For non-kBoxingTask
  std::unordered_map<RegisterType, int64_t> register_type_to_produced_group_id_;
  // (2) For kBoxingTask
  std::unordered_map<std::string, std::vector<int64_t>>
    consumer_segment_to_produced_group_ids_;

  // RegisterInfo consumed by current TaskDag.
  std::unordered_set<int64_t> consumed_register_info_group_ids_;

  // RegisterInfo consumed by other TaskDags. The RegisterInfo may or may not be
  // produced by current TaskDag.
  // (1) One task may needs multiple group_ids, for example, a backward compute
  // task depends on two registers from the corresponding forward compute task,
  // that is, the kDataType RegisterInfo and the kModelType RegisterInfo.
  // (2) The depended RegisterInfo may not be produced by current TaskDag. For 
  // example, the kModelType RegisterInfo is actually produced by task in 
  // kModelUpdatePath, however, the message carrying the Register ID will be
  // sent by the forward compute TaskDag.
  std::unordered_map<int32_t, std::vector<int64_t>> consumer_id_to_group_ids_;

  RegisterInfoManager(const RegisterInfoManager& other) = delete;
  RegisterInfoManager& operator=(const RegisterInfoManager& other) = delete;
};
}  // namespace caffe
#endif  // _REGISTER_INFO_MANAGER_H_