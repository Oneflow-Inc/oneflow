#ifndef ONEFLOW_CORE_GRAPH_INPLACE_OBN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_INPLACE_OBN_GRAPH_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/graph.h"

namespace oneflow {

class InplaceObnEdge;

class InplaceObnNode : public Node<InplaceObnNode, InplaceObnEdge> {
 public:
  virtual ~InplaceObnNode() = default;

  const Operator& op() const { return *op_; }
  const std::string& obn() const { return obn_; }
  const LogicalBlobId& lbi() const { return op().BnInOp2Lbi(obn()); }
  virtual const std::string& ibn() const = 0;

 protected:
  OF_DISALLOW_COPY_AND_MOVE(InplaceObnNode);
  InplaceObnNode(const Operator* op, const std::string& obn) : op_(op), obn_(obn) {}

 private:
  const Operator* op_;
  const std::string obn_;
};

class NormalInplaceObnNode final : public InplaceObnNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalInplaceObnNode);
  NormalInplaceObnNode(const Operator* op, const std::string& obn) : InplaceObnNode(op, obn) {}
  ~NormalInplaceObnNode() = default;
  const std::string& ibn() const override;
};

class FakeObnInplaceObnNode final : public InplaceObnNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FakeObnInplaceObnNode);
  FakeObnInplaceObnNode(const Operator* op, const std::string& obn, const std::string& ibn)
      : InplaceObnNode(op, obn), ibn_(ibn) {}
  ~FakeObnInplaceObnNode() = default;

  const std::string& ibn() const override { return ibn_; }

 private:
  std::string ibn_;
};

class InplaceObnEdge final : public Edge<InplaceObnNode, InplaceObnEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InplaceObnEdge);
  InplaceObnEdge() = default;
  ~InplaceObnEdge() = default;
};

class InplaceObnGraph final : public Graph<const InplaceObnNode, const InplaceObnEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InplaceObnGraph);
  InplaceObnGraph(const OpBlobArgList& obas,
                  const std::function<const Operator*(const std::string&)>& Op4OpName) {
    Init(obas, Op4OpName);
  }
  ~InplaceObnGraph() = default;

  void ComputeSafeInplaceObns(OpBlobArgList* obas) const;

 private:
  void Init(const OpBlobArgList& obas,
            const std::function<const Operator*(const std::string&)>& Op4OpName);
  void InitNodes(HashMap<OpBlobArg, InplaceObnNode*>* oba2node, const OpBlobArgList& obas,
                 const std::function<const Operator*(const std::string&)>& Op4OpName);
  void CompleteVariableObnNodes(
      HashMap<OpBlobArg, InplaceObnNode*>* oba2node,
      const std::function<const Operator*(const std::string&)>& Op4OpName);
  void InitEdges(const HashMap<OpBlobArg, InplaceObnNode*>& oba2node, const OpBlobArgList& obas,
                 const std::function<const Operator*(const std::string&)>& Op4OpName);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_INPLACE_OBN_GRAPH_H_
