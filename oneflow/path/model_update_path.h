#ifndef ONEFLOW_PATH_MODEL_UPDATE_PATH_H_
#define ONEFLOW_PATH_MODEL_UPDATE_PATH_H_

#include "path/model_path.h"

namespace oneflow {

class ModelUpdatePath final : public ModelPath {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdatePath);
  ModelUpdatePath() = default;
  ~ModelUpdatePath() = default;
  
  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::ModelUpdateFwBuildExecAndProducedRegsts;
  }

  void Build(
      const ChainNode* data_chain,
      const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain);

 private:
  void BuildTaskGraph(const ChainNode* data_chain);
  void InitFaker2MccoyMapAndParallelIdUpdateMap(
      const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
      HashMap<int32_t, CompTaskNode*>* parallel_id2update_node);
  void SubscribeCrossPathWithoutFaker(
      const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
      const HashMap<int32_t, CompTaskNode*>& parallel_id2update_node);
  void SubscribeCrossPathWithFaker(
      const HashMap<int32_t, CompTaskNode*>& parallel_id2update_node);

};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_UPDATE_PATH_H_
