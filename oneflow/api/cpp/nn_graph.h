#include <string>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_conf.pb.h"

namespace oneflow_api {

class Graph {
 public:
  void Save();
  void Load(const std::string& model_path, const std::string& version,
            const std::string& saved_model_filename);

 private:
  void CreateVariableOp(oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>>&
                            variable_op_name_to_tensor);

  oneflow::Job job_;
};

}  // namespace oneflow_api
