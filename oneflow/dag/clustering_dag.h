#ifndef _DAG_CLUSTERING_DAG_H_
#define _DAG_CLUSTERING_DAG_H_
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include "dag/dag_node.h"
#include "dag/dag.h"
/*
ClusteringDag could cluster some operator nodes in an |LogicalDag| according to
some user-defined criteria. It supports two types of merging, non-iterative and 
iterative.
*/
namespace caffe {
class BlobMeta;

class ClusteringMeta;

template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;

template <typename Dtype>
class LogicalDag;

enum class MergeWayInClusteringDag {
  kMergeModelParallel = 0,             // merge model parallel layers 
  kMergeDataParallel,                  // merge data parallel layers
};

template <typename Dtype>
class ClusteringDag : public Dag<BlobMeta, ClusteringMeta> {
  friend class DagIterator<ClusteringDag<Dtype>>;
  friend class DagIterator<ClusteringDag<Dtype>, true>;
  friend class DagReverseIterator<ClusteringDag<Dtype>>;
  friend class DagReverseIterator<ClusteringDag<Dtype>, true>;
public:
  ClusteringDag(
    const LogicalDag<Dtype>& logical_dag,
    const std::vector<std::vector<std::string>>& inherited_clusters,
    const std::unordered_set<std::string>& inherited_unchanged_clusters,
    MergeWayInClusteringDag merge_way_in_clustring_dag);

  ~ClusteringDag() {}

  void Build();
  std::string GetClusterNameFromLayerName(const std::string& layer_name) const;
  std::vector<std::vector<std::string>> GetResultedClusters() const {
    return resulted_clusters_;
  }

private:
  const LogicalDag<Dtype>& logical_dag_;
  std::unordered_map<std::string, std::string> layer_name_to_cluster_name_;

  MergeWayInClusteringDag merge_way_in_clustring_dag_;

  // Clusters inherited from previous clustering result
  const std::vector<std::vector<std::string>>& inherited_clusters_;
  std::vector<std::vector<std::string>> resulted_clusters_;

  const std::unordered_set<std::string> inherited_unchanged_clusters_;
  // Constructed from |inherited_clusters_|, indicate the clusters which should
  // not be merged further.
  std::unordered_set<std::string> clusters_unchanged_;

  void Clear();
  void ProcessAndCheckInheritedClusters();

  void BuildFromInheritedClusters();
  void InitOpNodesFromClusters(
    const std::vector<std::vector<std::string>> & clusters);
  void InitDataNodes();

  void ReBuildFromMetaClusters(
    const std::vector<std::vector<std::string>>& meta_clusters);

  void MergeCluster();
  void MergeModelParallelClusters();
  void MergeDataParallelClusters();

  void FindMetaClustersOnePass(
    std::vector<std::vector<std::string>>* meta_clusters, bool merge_model_parallel);
  std::vector<std::string> FindMetaClusterFromSeedByModelParallelMerging(
    const DagNode* seed);
  std::vector<std::string> FindMetaClusterFromSeedByDataParallelMerging(
    const DagNode* seed,
    const std::unordered_set<std::string>& clusters_done);

  ONode* AddOpNode(const std::vector<std::string>& atom_names);
  DNode* AddDataNode(const std::string& blob_name);

  std::string build_cluster_name(
    const std::vector<std::string>& atom_names) const;

  ClusteringDag(const ClusteringDag& other) = delete;
  ClusteringDag& operator=(const ClusteringDag& other) = delete;
};
}  // namespace caffe
#endif  // _DAG_CLUSTERING_DAG_H_
