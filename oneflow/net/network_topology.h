#ifndef THOR_NET_NETWORK_TOPOLOGY_H_
#define THOR_NET_NETWORK_TOPOLOGY_H_

#include <string>
#include <unordered_set>

namespace oneflow {

// A graph to store connection relationship in network
// After Dag is build successfully, NetworkTopo can be generated from the dag.
// 
// TODO(feiga): Some helper function like
// 
// void Dag::GenerateNetworkTopo(NetworkTopo* result);
struct NetworkTopology {
  struct Node {
    int32_t id;
    std::string address;
    int32_t port;
    std::unordered_set<int32_t> neighbors;
  };
  std::vector<Node> all_nodes;
};

}  // namespace caffe

#endif  // THOR_NET_NETWORK_TOPOLOGY_H_
