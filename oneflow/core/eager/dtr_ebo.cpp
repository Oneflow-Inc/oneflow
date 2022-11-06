#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace vm {

class DtrEbo : public EagerBlobObject {
 public:
  const std::shared_ptr<const one::MutLocalTensorMeta>& mut_tensor_meta() override {
    return content_->mut_tensor_meta();
  }
  // Getters
  const Symbol<one::LocalTensorMeta>& tensor_meta() const override { return content_->tensor_meta(); }

  // user_op::TensorDesc overrides
  const Shape& shape() const override { return content_->shape(); }
  const Stride& stride() const override { return content_->stride(); }
  DataType data_type() const override { return content_->data_type(); }
  bool is_dynamic() const override { return content_->is_dynamic(); }

  void set_shape(const Shape& shape) override { return content_->set_shape(shape); }
  void set_stride(const Stride& stride) override { return content_->set_stride(stride); }
  void set_data_type(DataType data_type) override { return content_->set_data_type(data_type); }
  void set_is_dynamic(bool is_dynamic) override { return content_->set_is_dynamic(is_dynamic); }

  // user_op::Tensor overrides
  ShapeView shape_view() const override { return content_->shape_view(); }
  MutShapeView mut_shape_view() override { return content_->mut_shape_view(); }
  const MemoryCase& mem_case() const override { return content_->mem_case(); }
  const void* raw_dptr() const override { return content_->raw_dptr(); }
  void* mut_raw_dptr() override { return content_->mut_raw_dptr(); }

  void set_storage_offset(const int64_t offset) override {
    return content_->set_storage_offset(offset);
  }

  Maybe<void> TryAllocateBlobBodyMemory(vm::Allocator* allocator) override {
    return content_->TryAllocateBlobBodyMemory(allocator);
  }
  Maybe<void> DeallocateBlobDataPtr() override { return content_->DeallocateBlobDataPtr(); }
  void RegisterStorageDeleteHook(const std::function<void()>& hook) override {
    return content_->RegisterStorageDeleteHook(hook);
  }

  Maybe<LocalDepObject*> compute_local_dep_object() const override {
    return content_->compute_local_dep_object();
  }

  std::shared_ptr<TensorStorage>& tensor_storage() override { return content_->tensor_storage(); }

  const Optional<Symbol<::oneflow::Stream>>& producer_stream() const override {
    return content_->producer_stream();
  }
  Maybe<void> init_producer_stream(Symbol<::oneflow::Stream> producer_stream) override {
    return content_->init_producer_stream(producer_stream);
  }

  const Optional<Symbol<::oneflow::Stream>>& last_used_stream() const override {
    return content_->last_used_stream();
  }
  void set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream) override {
    return content_->set_last_used_stream(last_used_stream);
  }

  std::shared_ptr<const Shape> shape_ptr() const override { return content_->shape_ptr(); }
  std::shared_ptr<const Stride> stride_ptr() const override { return content_->stride_ptr(); }

  size_t ByteSizeOfBlobBody() const override { return content_->ByteSizeOfBlobBody(); }
  size_t AlignedByteSizeOfBlobBody() const override { return content_->AlignedByteSizeOfBlobBody(); }
  size_t ByteSizeOfBlobHeader() const override { return content_->ByteSizeOfBlobHeader(); }
  size_t AlignedByteSizeOfBlobHeader() const override {
    return content_->AlignedByteSizeOfBlobHeader();
  }

  const char* header_ptr() const override { return content_->header_ptr(); }
  char* mut_header_ptr() override { return content_->mut_header_ptr(); }

  void InitOrCheckMemPtrForAllocationComputationPipelining() override {
    return content_->InitOrCheckMemPtrForAllocationComputationPipelining();
  }

  void TryInitNonPODTypeEagerBlobObjectIfNeed() override {
    return content_->TryInitNonPODTypeEagerBlobObjectIfNeed();
  }

  void set_content(const std::shared_ptr<EagerBlobObject>& content) { content_ = content; }

 private:
  std::shared_ptr<EagerBlobObject> content_;
};

}  // namespace vm
}  // namespace oneflow
