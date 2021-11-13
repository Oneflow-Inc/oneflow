#include <iostream>
#include <string>
#include <unordered_map>
#include "oneflow/api/cpp/nn_graph.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/api/cpp/utils.h"

// COMMAND(oneflow_api::StartOneFlow());
// Segmentation fault

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: graph_exe model_path version saved_model_filename\n";
    return 0;
  }

  oneflow_api::StartOneFlow();

  std::string model_path = std::string(argv[1]);
  std::string version = std::string(argv[2]);
  std::string saved_model_filename = std::string(argv[3]);
  oneflow_api::Graph graph;
  graph.Load(model_path, version, saved_model_filename);
  return 1;
}
