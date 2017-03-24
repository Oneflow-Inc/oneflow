#include "dag/clustering_dag.h"
#include "common/common.h"
#include "common/stl_util.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "dag/logical_dag.h"

namespace oneflow {
template <typename Dtype>
ClusteringDag<Dtype>::ClusteringDag(
  const LogicalDag<Dtype>& logical_dag,
  const std::vector<std::vector<std::string>>& inherited_clusters,
  const std::unordered_set<std::string>& inherited_unchanged_clusters,
  MergeWayInClusteringDag merge_way_in_clustring_dag)
  : logical_dag_(logical_dag),
  inherited_clusters_(inherited_clusters),
  inherited_unchanged_clusters_(inherited_unchanged_clusters),
  merge_way_in_clustring_dag_(merge_way_in_clustring_dag) {
  // Check whether the merge way is valid 
  CHECK(merge_way_in_clustring_dag_ == MergeWayInClusteringDag::kMergeModelParallel
    || merge_way_in_clustring_dag_ == MergeWayInClusteringDag::kMergeDataParallel)
    << "Unknown merge way in clustering DAG";
}

template <typename Dtype>
void ClusteringDag<Dtype>::Build() {
  ProcessAndCheckInheritedClusters();
  BuildFromInheritedClusters();
  MergeCluster();
}

template <typename Dtype>
void ClusteringDag<Dtype>::ProcessAndCheckInheritedClusters() {
  // Update clusters_unchanged_
  for (auto& cluster : inherited_clusters_) {
    // The cluster having more than 1 atoms is not allowed to be merged further
    if (cluster.size() > 1) {
      auto cluster_name = build_cluster_name(cluster);
      clusters_unchanged_.insert(cluster_name);
    }
  }
  for (auto& inherited_unchanged_cluster : inherited_unchanged_clusters_) {
    clusters_unchanged_.insert(inherited_unchanged_cluster);
  }
}

template <typename Dtype>
void ClusteringDag<Dtype>::BuildFromInheritedClusters() {
  InitOpNodesFromClusters(inherited_clusters_);
  InitDataNodes();
  AddStartAndEndNodes();
  PostProcessing();
}

template <typename Dtype>
void ClusteringDag<Dtype>::Clear() {
  layer_name_to_cluster_name_.clear();
  Dag<BlobMeta, ClusteringMeta>::Clear();
}

template <typename Dtype>
void ClusteringDag<Dtype>::InitOpNodesFromClusters(
  const std::vector<std::vector<std::string>> & clusters) {
  for (auto& cluster : clusters) {
    auto cluster_node = AddOpNode(cluster);
  }
  resulted_clusters_ = clusters;
}

template <typename Dtype>
std::string ClusteringDag<Dtype>::build_cluster_name(
  const std::vector<std::string>& atom_names) const {
  std::string cluster_name = "";
  for (auto& atom_name : atom_names) {
    if (!cluster_name.empty()) {
      cluster_name += "_";
    }
    cluster_name += atom_name;
  }
  return cluster_name;
}

template <typename Dtype>
OpNode<ClusteringMeta>* ClusteringDag<Dtype>::AddOpNode(
  const std::vector<std::string>& layer_names) {
  std::string cluster_name = build_cluster_name(layer_names);
  for (auto& layer_name : layer_names) {
    if (layer_name_to_cluster_name_.count(layer_name) > 0) {
      // the atom already exists
      layer_name_to_cluster_name_[layer_name] = cluster_name;
    }
    else {
      // a new atom
      layer_name_to_cluster_name_.insert({ layer_name, cluster_name });
    }
  }
  auto op_node = NewOpNode(cluster_name);
  auto& cluster_meta = op_node->mutable_op();
  cluster_meta = std::make_shared<ClusteringMeta>();
  cluster_meta->mutable_layer_names() = layer_names;
  auto it = op_name_to_node_.find(cluster_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << cluster_name;
  op_name_to_node_.insert({ cluster_name, op_node });
  return op_node;
}

template <typename Dtype>
std::string ClusteringDag<Dtype>::GetClusterNameFromLayerName(
  const std::string& layer_name) const {
  auto cluster_it = layer_name_to_cluster_name_.find(layer_name);
  CHECK(cluster_it != layer_name_to_cluster_name_.end());
  return cluster_it->second;
}

template <typename Dtype>
void ClusteringDag<Dtype>::InitDataNodes() {
  // For each layer in each cluster, finding its preceding layers. If the
  // the preceding layer is in another cluster, find the blob in-between. Create
  // a data node for the blob if it has no corresponding data node in the DAG.
  // Connect the edges if necessary.
  for (auto& name_node_pair : op_name_to_node_) {
    auto cluster_name = name_node_pair.first;
    auto cluster_node = name_node_pair.second;
    auto cluster_meta = cluster_node->op();
    auto layer_names = cluster_meta->layer_names();
    for (auto& layer_name : layer_names) {
      auto predecessors
        = logical_dag_.GetPrecedingOpNodeNames(layer_name);
      for (auto& predecessor : predecessors) {
        auto predecessor_cluster
          = GetClusterNameFromLayerName(predecessor);
        if (predecessor_cluster != cluster_name) {
          auto blob_names
            = logical_dag_.FindDataNodesInBetween(predecessor, layer_name);
          CHECK(blob_names.size() == 1);
          if (data_name_to_node_.count(blob_names[0]) > 0) {
            // The data_node of blob_names[0] already exists
            cluster_node->AddParent(data_name_to_node_[blob_names[0]]);
          }
          else {
            // Create the data_node of blob_names[0]
            auto data_node = AddDataNode(blob_names[0]);
            auto predecessor_cluster_node = GetOpNode(predecessor_cluster);
            data_node->AddParent(predecessor_cluster_node);
            cluster_node->AddParent(data_node);
          }
        }
      }
    }
  }
  // Add the data nodes of the blobs which directly connect to END node
  auto blob_names_without_successor
    = logical_dag_.GetPreceedingDataNodeNamesOfEndNode();
  for (auto& blob_name : blob_names_without_successor) {
    auto data_node = AddDataNode(blob_name);
    auto data_node_predecessors
      = logical_dag_.GetPreceedingOpNodeNamesOfDataNode(blob_name);
    CHECK(data_node_predecessors.size() == 1);
    auto preceeding_cluster_name
      = GetClusterNameFromLayerName(data_node_predecessors[0]);
    auto preceeding_cluster_node = GetOpNode(preceeding_cluster_name);
    data_node->AddParent(preceeding_cluster_node);
  }
}

template <typename Dtype>
DataNode<BlobMeta>* ClusteringDag<Dtype>::AddDataNode(
  const std::string& blob_name) {
  auto data_node = NewDataNode(blob_name);
  auto& blob_meta = data_node->mutable_data();
  blob_meta = std::make_shared<BlobMeta>();
  blob_meta->mutable_name() = blob_name;
  auto it = data_name_to_node_.find(blob_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << blob_name;
  data_name_to_node_.insert({ blob_name, data_node });
  return data_node;
}

template <typename Dtype>
void ClusteringDag<Dtype>::MergeCluster() {
  switch (merge_way_in_clustring_dag_)
  {
  case MergeWayInClusteringDag::kMergeModelParallel:
    MergeModelParallelClusters();
    break;
  case MergeWayInClusteringDag::kMergeDataParallel:
    MergeDataParallelClusters();
    break;
  default:
    LOG(FATAL) << "Unknown merge way in clustering DAG";
    break;
  }
}

template <typename Dtype>
void ClusteringDag<Dtype>::MergeModelParallelClusters() {
  std::vector<std::vector<std::string>> meta_clusters;
  FindMetaClustersOnePass(&meta_clusters, true);
  if (meta_clusters.empty()) return;
  ReBuildFromMetaClusters(meta_clusters);
}

template <typename Dtype>
void ClusteringDag<Dtype>::ReBuildFromMetaClusters(
  const std::vector<std::vector<std::string>>& meta_clusters) {
  std::vector<std::vector<std::string>> atoms_of_each_cluster;
  for (auto& meta_cluster : meta_clusters) {
    std::vector<std::string> atoms_in_current_meta_cluster;
    for (auto& cluster : meta_cluster) {
      auto cluster_node = GetOpNode(cluster);
      auto cluster_meta = cluster_node->op();
      auto atoms_in_cluster = cluster_meta->layer_names();
      for (auto& atom_in_cluster : atoms_in_cluster) {
        atoms_in_current_meta_cluster.push_back(atom_in_cluster);
      }
    }
    atoms_of_each_cluster.push_back(atoms_in_current_meta_cluster);
  }
  this->Clear();
  InitOpNodesFromClusters(atoms_of_each_cluster);
  InitDataNodes();
  AddStartAndEndNodes();
  PostProcessing();
}

template <typename Dtype>
void ClusteringDag<Dtype>::MergeDataParallelClusters() {
  int32_t iteration = 0;
  while (true) {
    std::vector<std::vector<std::string>> meta_clusters;
    FindMetaClustersOnePass(&meta_clusters, false);
    if (meta_clusters.empty()) break;
    ReBuildFromMetaClusters(meta_clusters);
    iteration++;
  }
}

template <typename Dtype>
void ClusteringDag<Dtype>::FindMetaClustersOnePass(
  std::vector<std::vector<std::string>>* meta_clusters,
  bool merge_model_parallel) {
  int32_t cluster_num = 0;
  std::unordered_set<std::string> clusters_done;
  DagIterator<ClusteringDag<Dtype>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    auto current_name = current_node->node_name();
    // Skip the non-kOpNode clusters
    if (current_node->Type() != NodeType::kOpNode) continue;
    cluster_num++;
    // Skip the clusters already being covered by a previously-found meta-cluster
    if (clusters_done.count(current_name) > 0) continue;

    // Using current_node as a seed to find all other clusters which could be
    // combined as a meta-cluster.
    std::vector<std::string> cluster_names;
    if (clusters_unchanged_.count(current_name) > 0) {
      cluster_names.push_back(current_name);
    } else {
      if (merge_model_parallel) {
        // merge model parallel layers
        cluster_names = FindMetaClusterFromSeedByModelParallelMerging(current_node);
      } else {
        // merge data parallel layers
        cluster_names = FindMetaClusterFromSeedByDataParallelMerging(
          current_node,
          clusters_done);
      }
    }
    // Add a meta-cluster into meta_clusters
    meta_clusters->push_back(cluster_names);

    // Mark the clusters in cluster_names as done, therefore they will not be
    // treated as seeds in the future.
    for (auto& cluster_name : cluster_names) {
      clusters_done.insert(cluster_name);
    }
  }
  // If the number of clusters does not change, clear the meta_clusters to
  // indicate no need to merge
  if (meta_clusters->size() == cluster_num) {
    meta_clusters->clear();
  }
}

template <typename Dtype>
std::vector<std::string>
ClusteringDag<Dtype>::FindMetaClusterFromSeedByModelParallelMerging(
const DagNode* seed) {
  auto seed_name = seed->node_name();
  std::vector<std::string> cluster_names;
  cluster_names.push_back(seed_name);

  auto seed_node
    = dynamic_cast<const OpNode<ClusteringMeta>*>(seed);
  CHECK_NOTNULL(seed_node);
  auto seed_meta = seed_node->op();
  auto seed_layer_names = seed_meta->layer_names();
  CHECK(seed_layer_names.size() == 1);
  auto seed_layer_node = logical_dag_.GetOpNode(seed_layer_names[0]);
  auto seed_layer_meta = seed_layer_node->op();
  auto& seed_placement_info = seed_layer_meta->placement_info();
  // seed's parallel policy needs to be model-parallel on multiple devices
  if (seed_placement_info.parallel_policy()
    == kModelParallelOnMultipleDevices) {
    // Loop until no further cluster can be merged to this seed
    std::string cluster_segment_end = seed_name;
    while (true) {
      auto succeeding_cluster_names
        = this->GetSucceedingOpNodeNames(cluster_segment_end);
      // cluster_segment_end must have only one succeeding cluster
      if (succeeding_cluster_names.size() != 1) break;
      auto succeeding_node
        = this->GetOpNode(succeeding_cluster_names[0]);
      auto succeeding_cluster_node
        = dynamic_cast<const OpNode<ClusteringMeta>*>(succeeding_node);
      CHECK_NOTNULL(succeeding_cluster_node);
      auto succeeding_cluster_meta = succeeding_cluster_node->op();
      auto succeeding_layer_names = succeeding_cluster_meta->layer_names();
      // cluster_segment_end must have only one succeeding layer
      if (succeeding_layer_names.size() != 1) break;
      auto succeeding_layer_node
        = logical_dag_.GetOpNode(succeeding_layer_names[0]);
      auto succeeding_layer_meta = succeeding_layer_node->op();
      auto& succeeding_placement_info
        = succeeding_layer_meta->placement_info();
      auto succeeding_layer = succeeding_layer_meta->layer();
      // cluster_segment_end's succeeding layer needs to be element-wise op
      if (!succeeding_layer->IsElemWise()) break;
      // must have the same device set and same parallel policy
      if (!seed_placement_info.EqualTo(succeeding_placement_info)) break;
      cluster_names.push_back(succeeding_cluster_names[0]);
      cluster_segment_end = succeeding_cluster_names[0];
    }
  }
  return cluster_names;
}

template <typename Dtype>
std::vector<std::string>
ClusteringDag<Dtype>::FindMetaClusterFromSeedByDataParallelMerging(
const DagNode* seed,
const std::unordered_set<std::string>& clusters_done) {
  std::vector<std::string> cluster_names;
  std::unordered_set<std::string> clusters_set;
  auto seed_name = seed->node_name();
  cluster_names.push_back(seed_name);
  clusters_set.insert(seed_name);

  auto seed_node
    = dynamic_cast<const OpNode<ClusteringMeta>*>(seed);
  CHECK_NOTNULL(seed_node);
  auto seed_meta = seed_node->op();
  auto seed_layer_names = seed_meta->layer_names();
  CHECK(seed_layer_names.size() >= 1);

  auto seed_layer_node = logical_dag_.GetOpNode(seed_layer_names[0]);
  auto seed_layer_meta = seed_layer_node->op();
  auto& seed_placement_info = seed_layer_meta->placement_info();
  // seed's parallel policy needs to be data-parallel
  if (seed_placement_info.parallel_policy() == kDataParallelOnMultipleDevices
    || seed_placement_info.parallel_policy() == kNaiveParallelOnSingleDevice) {
    // Init the ancestors and descendants of clusters in current meta-cluster
    std::unordered_set<std::string> ancestors_of_meta_cluster
      = this->GetOpAncestorsOfOpNode(seed_name);
    std::unordered_set<std::string> descendants_of_meta_cluster
      = this->GetOpDescendantsOfOpNode(seed_name);

    // Loop until no further cluster can be merged to this seed
    while (true) {
      bool found_new_node = false;
      bool found_seed = false;

      DagIterator<ClusteringDag<Dtype>, true> dag_iterator(*this);
      for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
        auto current_node = dag_iterator.CurrentNode();
        // Skip the non-kOpNode
        if (current_node->Type() != NodeType::kOpNode) continue;
        if (current_node->node_id() == seed->node_id()) {
          found_seed = true;
        }
        // Skip all the nodes before the seed node in topological order
        if (!found_seed) continue;
        auto current_name = current_node->node_name();
        // Skip the cluster node already being covered by other meta-cluster
        if (clusters_done.count(current_name) > 0) continue;
        // Skip the cluster node already being covered by current meta-cluster
        if (clusters_set.count(current_name) > 0) continue;

        auto current_clustering_node
          = dynamic_cast<const OpNode<ClusteringMeta>*>(current_node);
        CHECK_NOTNULL(current_clustering_node);
        auto current_clustering_node_meta = current_clustering_node->op();
        auto current_layer_names = current_clustering_node_meta->layer_names();
        CHECK(current_layer_names.size() >= 1);
        auto current_layer_node = logical_dag_.GetOpNode(current_layer_names[0]);
        auto current_layer_meta = current_layer_node->op();
        auto& current_placement_info = current_layer_meta->placement_info();
        // check whether the seed layer and current layer have the same device
        // set and same parallel policy
        if (!current_placement_info.EqualTo(seed_placement_info)) {
          continue;
        }

        std::unordered_set<std::string> ancestors_of_current_node
          = this->GetOpAncestorsOfOpNode(current_name);
        std::unordered_set<std::string> descendants_of_current_node
          = this->GetOpDescendantsOfOpNode(current_name);

        // The current node is either a descendant of segment or not, it can not
        // be an ancestor of the segment, since the current_node is added in the
        // topological order.
        if (descendants_of_meta_cluster.count(current_name) == 0) {
          // There is no path from seed to current_node
          if (stl::SetIsEqual(ancestors_of_meta_cluster, ancestors_of_current_node)
            && stl::SetIsEqual(
            descendants_of_meta_cluster, descendants_of_current_node)) {
            // Found a new node
            found_new_node = true;
            // Update segment
            cluster_names.push_back(current_name);
            clusters_set.insert(current_name);
            // No need to update the ancestor_of_segment and descendant_of_segment
          }
        } else {
          // There is a path from seed to current_node
          std::unordered_set<std::string> ancestors_and_meta_cluster
            = ancestors_of_meta_cluster;
          // Get the set consisting of the ancestor of the meta-cluster and the
          // clusters in the meta-cluster
          for (auto& cluster_in_meta_cluster : clusters_set) {
            ancestors_and_meta_cluster.insert(cluster_in_meta_cluster);
          }

          // Get the set consisting of the descendant of the current node
          // and the current node
          std::unordered_set<std::string> descendants_and_current_node
            = descendants_of_current_node;
          descendants_and_current_node.insert(current_name);

          if (stl::SetIsEqual(ancestors_of_current_node, ancestors_and_meta_cluster)
            && stl::SetIsEqual(
            descendants_and_current_node, descendants_of_meta_cluster)) {
            // Found a new node
            found_new_node = true;
            // Update segment
            cluster_names.push_back(current_name);
            clusters_set.insert(current_name);
            // Update the descendant of segment
            CHECK_EQ(descendants_of_meta_cluster.erase(current_name), 1);
          }
        }
      }
      // If no new cluster node can be found, break the loop and return the result.
      if (!found_new_node) break;
    }
  }
  return cluster_names;
}

template class ClusteringDag<float>;
template class ClusteringDag<double>;
}