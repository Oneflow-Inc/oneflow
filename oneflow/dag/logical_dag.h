#ifndef _DAG_LOGIC_DAG_H_
#define _DAG_LOGIC_DAG_H_
#include <ctype.h>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <algorithm>
#include "dag/dag_node.h"
#include "dag/dag.h"
#include "layers/layer_factory.h"
#include "proto/oneflow.pb.h"
#include "context/placement_info.h"
#include "dag/dag_iterator.h"

/*
A DAG consists of a set layers specified by users through external config file.
It describes the logical structure of the complete neural networks.
The layers MUST be declared in a topological sorting order. In other words,
while declaring a new layer, all its predecessors should have been declared
before this new layer.
*/
namespace oneflow {
class BlobMeta;

template <typename Dtype>
class LayerMeta;

class NetDescriptor;

/*
template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;
*/

template <typename Dtype>
class LogicalDag : public Dag<BlobMeta, LayerMeta<Dtype>> {
  friend class DagIterator<LogicalDag<Dtype>>;
  friend class DagIterator<LogicalDag<Dtype>, true>;
  friend class DagReverseIterator<LogicalDag<Dtype>>;
  friend class DagReverseIterator<LogicalDag<Dtype>, true>;
 public:
  LogicalDag(std::shared_ptr<NetDescriptor> net_descriptor, PathType path_type,
    const std::string& name = "logical_dag");
  ~LogicalDag();

  std::string DagBlobFromLayerBlob(const std::string& layer_blob) const;
  PlacementInfo GetPlacementInfo(const std::string& layer_name) const;
  bool DagBlobNeedsBP(const std::string& dag_blob) const;

private:
  // BlobDict is used to manage the mapping between the alias names of a blob.
  class BlobDict {
  public:
    BlobDict() {}
    ~BlobDict() {}

    bool HasKey(const std::string& key) const;
    void AddPair(
      const std::string& key, const std::string& value);
    const std::string& GetValueWithKey(const std::string& key) const;
  private:
    std::unordered_map<std::string, std::string> dict_;

    BlobDict(const BlobDict& other) = delete;
    BlobDict& operator=(const BlobDict& other) = delete;
  };

  enum class LayerBlobRole {
    kInput = 0,
    kOutput
  };

  struct LayerBlobTriple {
    std::string layer_name;
    std::string var_name;
    LayerBlobRole role;
  };
  using LayerBlobTriples = std::vector<LayerBlobTriple>;

  // A dag_blob can only have at most one producer, but can have more than one
  // consumers.
  class DagBlobToLayerBlobs {
  public:
    DagBlobToLayerBlobs() = default;
    ~DagBlobToLayerBlobs() = default;

    void AddTriple(const std::string& dag_blob,
      const std::string& layer_name,
      const std::string& var_name,
      LayerBlobRole role);
    LayerBlobTriples GetTriples(const std::string& dag_blob) const;
  private:
    std::unordered_map<std::string, LayerBlobTriples> dag_blob_to_triples_;
  };

private:
  std::shared_ptr<NetDescriptor> net_descriptor_;
  std::unordered_set<std::string> available_tops_;
  BlobDict layer_blob_to_dag_blob_;
  std::unordered_map<std::string, bool> dag_blob_to_need_bp_;
  DagBlobToLayerBlobs dag_blob_to_layer_blobs_;
  
  void Build();

  void ProcessLayer(int32_t layer_id);
  OpNode<LayerMeta<Dtype>>* AddOpNode(const std::string& op_name,
    const std::string& op_type,
    const LayerProto& layer_param);
  DataNode<BlobMeta>* AddDataNode(const std::string& data_name);

  void AddOtherLayerBlobAndDagBlobMap();
  void AddLayerBlobs(const std::string& layer_anme,
    const std::vector<std::string>& vars);

  void SetDagBlobNeedsBP();
  bool GetDagBlobNeedsBP(const LayerBlobTriples& triples) const;

  void GetMessageContentByKeyOrDie(const google::protobuf::Message& proto,
    const std::string& key, std::string *value);

  void GetStringValueByKeyOrDie(const google::protobuf::Message& proto,
    const std::string& key, std::string *value);

  // Verify the consistency inside layer_param
  const google::protobuf::Message& LayerParameterIntegrityCheck(
    const LayerProto& layer_param);

  //void Setup();
  //void SetupDataNode(DNode* dnode);
  //void SetupOpNode(ONode* onode);

  LogicalDag(const LogicalDag& other) = delete;
  LogicalDag& operator=(const LogicalDag& other) = delete;
};
}  // namespace oneflow
#endif  // _DAG_LOGIC_DAG_H_
