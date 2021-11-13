/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
