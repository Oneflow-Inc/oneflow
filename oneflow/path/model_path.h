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

 protected:
  void set_data_chain(const ChainNode* data_chain) {
    data_chain_ = data_chain;
  }

 private:
  const ChainNode* data_chain_;

};

} // namespace oneflow

#endif // ONEFLOW_PATH_MODEL_PATH_H_
