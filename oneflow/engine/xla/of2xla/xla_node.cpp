#include <unordered_set>
#include "absl/strings/str_split.h"
#include "oneflow/engine/xla/of2xla/xla_utility.h"
#include "oneflow/engine/xla/of2xla/xla_node.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"

namespace oneflow {
namespace mola {

const SbpParallel &XlaEdge::sbp_policy(int index) const {
  CHECK_LT(index, 2);
  return sbp_policy_[index];
}

const Shape &XlaEdge::time_shape(int index) const {
  CHECK_LT(index, 2);
  return time_shape_[index];
}

void XlaEdge::set_sbp_policy(int index, const SbpParallel &policy) {
  CHECK_LT(index, 2);
  sbp_policy_[index] = policy;
}

void XlaEdge::set_time_shape(int index, const Shape &shape) {
  CHECK_LT(index, 2);
  time_shape_[index] = shape;
}

extern const std::string _XlaArgumentOpType;
extern const std::string _XlaInArgumentPrefix;
extern const std::string _XlaOutArgumentPrefix;

std::string DeviceTypeToBackend(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::kGPU:
      return "CUDA";
    case DeviceType::kCPU:
      return "CPU";
    default:
      DLOG(WARNING) << "Meet invalid DeviceType (" << device_type
                    << "). Use the default CPU backend instead.";
      return "CPU";
  }
}

DeviceType BackendToDeviceType(const std::string &backend) {
  if (backend == "CUDA") {
    return DeviceType::kGPU;
  } else if (backend == "CPU") {
    return DeviceType::kCPU;
  } else {
    return DeviceType::kCPU;
  }
}

XlaNode::XlaNode(const Operator *op) : op_(op), unique_id_(-1),
                                       cluster_id_(-1), sub_graph_(nullptr) {
  backend_ = DeviceTypeToBackend(op_->device_type());
  op_name_ = op_->op_name();
  op_type_ = ExtractOpTypeAsString(op_->op_conf());
  // Setup input and output logical blob ids
  for (const std::string &bn : op_->input_bns()) {
    const LogicalBlobId &lbi = op_->BnInOp2Lbi(bn);
    inputs_.emplace(bn, lbi);
  }
  for (const std::string &bn : op_->output_bns()) {
    const LogicalBlobId &lbi = op_->BnInOp2Lbi(bn);
    outputs_.emplace(bn, lbi);
  }
}

bool XlaNode::IsCompiled() const {
  return IsOpCompilerRegistered(backend_, op_type_);
}

void XlaNode::AddInEdge(const XlaEdge *edge) {
  in_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::AddOutEdge(const XlaEdge *edge) {
  out_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::EraseInEdge(const XlaEdge *edge) {
  in_edges_.remove_if([&](const XlaEdge *e) -> bool {
    return e->unique_id() == edge->unique_id() &&
           e->argument() == edge->argument();
  });
}

void XlaNode::EraseOutEdge(const XlaEdge *edge) {
  out_edges_.remove_if([&](const XlaEdge *e) -> bool {
    return e->unique_id() == edge->unique_id() &&
           e->argument() == edge->argument();
  });
}

bool XlaNode::IsSourceNode() const {
  return in_edges_.size() == 0;
}

bool XlaNode::IsFinishNode() const {
  return out_edges_.size() == 0;
}

bool XlaNode::IsArgumentNode() const {
  return op_type_ == _XlaArgumentOpType;
}

bool XlaNode::IsInArgumentNode() const {
  return IsArgumentNode() && absl::StartsWith(op_name_, _XlaInArgumentPrefix);
}

bool XlaNode::IsOutArgumentNode() const {
  return IsArgumentNode() && absl::StartsWith(op_name_, _XlaOutArgumentPrefix);
}

bool XlaNode::IsReachable(const XlaNode &dst_node) const {
  std::unordered_set<const XlaNode *> visited_nodes;
  std::stack<const XlaNode *> stack;
  for (const XlaEdge *edge : out_edges_) {
    stack.push(edge->end());
  }

  while (!stack.empty()) {
    const XlaNode *node = stack.top();
    stack.pop();
    if (node->unique_id() == dst_node.unique_id()) {
      return true;
    }
    for (const XlaEdge *edge : node->out_edges()) {
      const XlaNode *end = edge->end();
      if (visited_nodes.insert(end).second) {
        stack.push(end);
      }
    }
  }
  return false;
}

std::vector<std::string> XlaNode::input_bns() const {
  std::vector<std::string> input_bns(inputs_.size());
  std::transform(inputs_.begin(), inputs_.end(), input_bns.begin(),
                 [](const std::pair<std::string, LogicalBlobId> &in) {
                   return in.first;
                 });
  return input_bns;
}

std::vector<std::string> XlaNode::output_bns() const {
  std::vector<std::string> output_bns(outputs_.size());
  std::transform(outputs_.begin(), outputs_.end(), output_bns.begin(),
                 [](const std::pair<std::string, LogicalBlobId> &out) {
                   return out.first;
                 });
  return output_bns;
}

XlaArgumentNode::XlaArgumentNode(const XlaLaunchOpConf::Argument &arg_conf)
    : XlaNode(), arg_conf_(arg_conf) {
  this->op_type_ = _XlaArgumentOpType;
  this->op_name_ = arg_conf.name();
  this->backend_ = DeviceTypeToBackend(arg_conf.device_type());
  this->inputs_.emplace("in", BlobId(arg_conf.in()));
  this->outputs_.emplace("out", BlobId(arg_conf.out()));
}

bool IsNodeInput(const XlaNode *node, const LogicalBlobId &lbi) {
  for (XlaEdge *edge : node->in_edges()) {
    if (edge->argument().blob_id() == lbi) {
      return true;
    }
  }
  return false;
}

bool IsNodeOutput(const XlaNode *node, const LogicalBlobId &lbi) {
  for (XlaEdge *edge : node->out_edges()) {
    if (edge->argument().blob_id() == lbi) {
      return true;
    }
  }
  return false;
}

}  // namespace mola
}  // namespace oneflow
