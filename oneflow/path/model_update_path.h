#ifndef ONEFLOW_PATH_MODEL_UPDATE_PATH_H_
#define ONEFLOW_PATH_MODEL_UPDATE_PATH_H_

#include "path/path.h"

namespace oneflow {

class ModelUpdatePath final : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdatePath);
  ModelUpdatePath() = default;
  ~ModelUpdatePath() = default;

  void Build(
      const ChainNode* data_chain,
      const std::vector<CompTaskNode*>& sorted_comptasks4data_chain);

 private:
  void BuildTaskGraph(const ChainNode* data_chain);
  void InitFaker2DataMap(
      const std::vector<CompTaskNode*>& sorted_comptasks4data_chain);
  
  std::unordered_map<CompTaskNode*, CompTaskNode*> faker2data;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_UPDATE_PATH_H_
