#include "path/model_update_path.h"

namespace oneflow {

void ModelUpdatePath::Build(const ChainNode* chain_in_data_path) {
  BuildTaskGraph(chain_in_data_path);
  TODO();
  // BuildFakerMap
  // ProducedRegister SubscribeRegister
  // data_path subscribe model_register
}

void ModelUpdatePath::BuildTaskGraph(const ChainNode* chain_in_data_path) {
  TODO();
}

} // namespace oneflow
