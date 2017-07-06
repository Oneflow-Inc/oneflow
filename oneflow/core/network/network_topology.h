#ifndef ONEFLOW_CORE_NETWORK_NETWORK_TOPOLOGY_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_TOPOLOGY_H_

#include <string>
#include <unordered_set>
#include <vector>

namespace oneflow {

// A graph to store connection relationship in network
// After Dag is build successfully, NetworkTopo can be generated from the dag.
//
//
// void Dag::GenerateNetworkTopo(NetworkTopo* result);
struct NetworkTopology {
  struct Node {
    int64_t machine_id;
    std::string address;
    int32_t port;
    std::unordered_set<int64_t> neighbors;
  };
  std::vector<Node> all_nodes;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_TOPOLOGY_H_
