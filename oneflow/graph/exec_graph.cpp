#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
  pbn_ = "edge_" + std::to_string(edge_id()) + "/" + lbn;
}

void ExecNode::AddDtbnRegstPair(const std::string& dtbn, RegstDesc* regst) {
  TODO();
}

void ExecNode::AddIbnRegstPair(const std::string& ibn, RegstDesc* regst) {
  TODO();
}

void ExecNode::AddObnRegstPair(const std::string& obn, RegstDesc* regst) {
  TODO();
}

void ExecNode::AddMbnRegstPair(const std::string& mbn, RegstDesc* regst) {
  TODO();
}

void ExecNode::AddMtbnRegstPair(const std::string& mtbn, RegstDesc* regst) {
  TODO();
}

} // namespace oneflow
