#ifndef ONEFLOW_CORE_FRAMEWORK_FOLDER_RULE_TABLE_H_
#define ONEFLOW_CORE_FRAMEWORK_FOLDER_RULE_TABLE_H_

#include <vector>
#include <string>
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T, typename Kind>
class Singleton;

class FolderRuleTable final {
  public:
    OF_DISALLOW_COPY_AND_MOVE(FolderRuleTable);
    ~FolderRuleTable() = default;
    void Append(std::string new_rule) {
      if(!infix_rules_.empty()){
        for(auto& rule : infix_rules_) {
            if(new_rule!=rule && new_rule.find(rule)!=std::string::npos) {
                rule = new_rule;
                return;
            }
        }
      }
      infix_rules_.push_back(new_rule);
    }
    void Reset() {
        infix_rules_.clear();
    }
    std::vector<std::string>& GetRules() {return infix_rules_;}
  private:
    friend class Singleton<FolderRuleTable>;
    FolderRuleTable() = default;
    std::vector<std::string> infix_rules_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_FOLDER_RULE_TABLE_H_