#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class OneRecDecoder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneRecDecoder);
  OneRecDecoder(std::unique_ptr<BufferedBatchedOneRecReader>&& reader, int32_t batch_size,
                std::vector<DecodeOneRecFieldConf> fields);
  ~OneRecDecoder();

  bool GetBatch(const std::vector<Blob*>& blobs);

 private:
  int32_t batch_size_;
  std::unique_ptr<BufferedBatchedOneRecReader> reader_;
  bool filled_ = false;
  bool closed_ = false;
  std::vector<DecodeOneRecFieldConf> field_vec_;
  std::vector<std::vector<char>> buffer_vec_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::thread decode_thread_;
  std::vector<Shape> instance_shape_vec_;
};

}  // namespace oneflow
