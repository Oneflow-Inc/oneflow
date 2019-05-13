#ifndef ONEFLOW_CORE_GRAPH_INPLACE_LBI_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_INPLACE_LBI_GRAPH_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/graph.h"

namespace oneflow {

class InplaceLbiEdge;

class InplaceLbiNode : public Node<InplaceLbiNode, InplaceLbiEdge> {
 public:
  virtual ~InplaceLbiNode() = default;

  const LogicalBlobId& lbi() const { return lbi_; }

 protected:
  OF_DISALLOW_COPY_AND_MOVE(InplaceLbiNode);
  explicit InplaceLbiNode(const LogicalBlobId& lbi) : lbi_(lbi) {}

 private:
  LogicalBlobId lbi_;
};

class NormalInplaceLbiNode final : public InplaceLbiNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalInplaceLbiNode);
  explicit NormalInplaceLbiNode(const LogicalBlobId& lbi) : InplaceLbiNode(lbi) {}
  ~NormalInplaceLbiNode() override = default;
};

class SourceOpInplaceLbiNode final : public InplaceLbiNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceOpInplaceLbiNode);
  explicit SourceOpInplaceLbiNode(const LogicalBlobId& lbi) : InplaceLbiNode(lbi) {}
  ~SourceOpInplaceLbiNode() = default;
};

class UpdateInplaceLbiNode final : public InplaceLbiNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpdateInplaceLbiNode);
  explicit UpdateInplaceLbiNode(const LogicalBlobId& lbi) : InplaceLbiNode(lbi) {}
  ~UpdateInplaceLbiNode() = default;
};

class InplaceLbiEdge final : public Edge<InplaceLbiNode, InplaceLbiEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InplaceLbiEdge);
  InplaceLbiEdge(const Operator* op, const std::string& ibn, const std::string& obn)
      : op_(op), ibn_(ibn), obn_(obn) {}
  ~InplaceLbiEdge() = default;

  const Operator& op() const { return *op_; }
  const std::string& ibn() const { return ibn_; }
  const std::string& obn() const { return obn_; }

 private:
  const Operator* op_;
  const std::string ibn_;
  const std::string obn_;
};

class InplaceLbiGraph final : public Graph<const InplaceLbiNode, const InplaceLbiEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InplaceLbiGraph);
  InplaceLbiGraph(const OpBlobArgList& obas,
                  const std::function<const Operator*(const std::string&)>& Op4OpName) {
    Init(obas, Op4OpName);
  }
  ~InplaceLbiGraph() = default;

  void ComputeSafeInplaceObns(OpBlobArgList* obas,
                              const std::function<bool(const LogicalBlobId&, const std::string&)>&
                                  IsReachableFromLbiToOpName) const;

 private:
  void Init(const OpBlobArgList& obas,
            const std::function<const Operator*(const std::string&)>& Op4OpName);
  std::function<InplaceLbiNode*(const LogicalBlobId&)> MakeMutFindOrCreateNode(
      std::function<const Operator*(const std::string&)> Op4OpName);
  void ComputeSafeInplaceObns(const std::function<bool(const LogicalBlobId&, const std::string&)>&
                                  IsReachableFromLbiToOpName,
                              const std::function<void(const InplaceLbiEdge*)>& Handler) const;
  void ComputeSafeInplaceObns(const HashSet<const InplaceLbiNode*>& nodes,
                              const std::function<bool(const LogicalBlobId&, const std::string&)>&
                                  IsReachableFromLbiToOpName,
                              const std::function<void(const InplaceLbiEdge*)>& Handler) const;
  void GetSafeInplaceObnEdges(const HashSet<const InplaceLbiNode*>& nodes,
                              const HashSet<const InplaceLbiEdge*>& disabled_edges,
                              HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const;
  void GetDisabledNodes(const HashSet<const InplaceLbiNode*>& nodes,
                        const HashSet<const InplaceLbiEdge*>& disabled_edges,
                        HashSet<const InplaceLbiNode*>* cur_disabled_nodes) const;
  void DisconnectDataMutableEdgeByReachability(
      const HashSet<const InplaceLbiNode*>& nodes,
      const HashSet<const InplaceLbiEdge*>& disabled_edges,
      const std::function<bool(const LogicalBlobId&, const std::string&)>&
          IsReachableFromLbiToOpName,
      HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const;
  void DisconnectDataMutableEdgeByReducingConficts(
      const HashSet<const InplaceLbiNode*>& nodes,
      const HashSet<const InplaceLbiEdge*>& disabled_edges,
      const std::function<bool(const LogicalBlobId&, const std::string&)>&
          IsReachableFromLbiToOpName,
      HashSet<const InplaceLbiEdge*>* cur_disabled_edges) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_INPLACE_LBI_GRAPH_H_
