#ifndef ONEFLOW_API_COMMON_FOLDER_RULE_TABLE_H_
#define ONEFLOW_API_COMMON_FOLDER_RULE_TABLE_H_

#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/folder_rule_table.h"

namespace oneflow {

inline std::vector<std::string>& GetFolderRuleTable() {
  auto folder_rule_table= Singleton<FolderRuleTable>::Get();
  return folder_rule_table->GetRules();
}

inline void AppendRuleToFolderRuleTable(std::string new_rule) {
  auto folder_rule_table= Singleton<FolderRuleTable>::Get();
  folder_rule_table->Append(new_rule);
}

inline void ResetFolderRuleTable() {
  auto folder_rule_table= Singleton<FolderRuleTable>::Get();
  folder_rule_table->Reset();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_COMMON_FOLDER_RULE_TABLE_H_