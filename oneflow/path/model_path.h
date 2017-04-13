#ifndef ONEFLOW_PATH_MODEL_PATH_H_
#define ONEFLOW_PATH_MODEL_PATH_H_

#include "path/path.h"

namespace oneflow {

class ModelPath : public Path {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelPath);
  ModelPath() = default;
  virtual ~ModelPath() = default;

  const ChainNode* GetDataChain() const override { return data_chain_; }
  CompTaskNode* Faker2Mccoy(CompTaskNode* faker) const override;

 protected:
  void set_data_chain(const ChainNode* data_chain);
  const HashMap<CompTaskNode*, CompTaskNode*>& faker2mccoy() const;
  void AddFakerMccoyPair(CompTaskNode* faker, CompTaskNode* mccoy);

 private:
  const ChainNode* data_chain_;
  HashMap<CompTaskNode*, CompTaskNode*> faker2mccoy_;

};

inline CompTaskNode* ModelPath::Faker2Mccoy(CompTaskNode* faker) const {
  return faker2mccoy_.at(faker);
}

inline void ModelPath::set_data_chain(const ChainNode* data_chain) {
  data_chain_ = data_chain;
}

inline const HashMap<CompTaskNode*, CompTaskNode*>&
ModelPath::faker2mccoy() const {
  return faker2mccoy_;
}

inline void ModelPath::AddFakerMccoyPair(CompTaskNode* faker,
                                         CompTaskNode* mccoy) {
  CHECK(faker2mccoy_.emplace(faker, mccoy).second);
}

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_PATH_H_
