#include <iostream>
#include <string>
#include <unordered_map>
#include "oneflow/api/cpp/nn_graph.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: graph_exe model_path version saved_model_filename\n";
    return 0;
  }

  // Need StartOneFlow()
  // oneflow::Global<oneflow::ProcessCtx>::New();

  std::string model_path = std::string(argv[1]);
  std::string version = std::string(argv[2]);
  std::string saved_model_filename = std::string(argv[3]);
  oneflow_api::Graph graph;
  graph.Load(model_path, version, saved_model_filename);
  return 1;
}
