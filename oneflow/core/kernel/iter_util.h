#ifndef ONEFLOW_CORE_KERNEL_ITER_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ITER_UTIL_H_

namespace oneflow {

using CopyBlobFieldMthd = void (Blob::*)(DeviceCtx*, const Blob*);

template<typename Iter, DeviceType device_type>
void CopyFromIterToIter(DeviceCtx* ctx, Iter& src_it, Iter& dst_it) {
  const char* src_ptr = nullptr;
  size_t src_size = 0;
  char* dst_ptr = nullptr;
  size_t dst_size = 0;
  while (true) {
    if (src_size == 0) { std::tie(src_ptr, src_size) = src_it.GetNext(); }
    if (dst_size == 0) { std::tie(dst_ptr, dst_size) = dst_it.GetNext(); }
    if (src_size == 0) {
      CHECK_EQ(src_size, dst_size);
      break;
    }
    size_t cp_size = std::min(src_size, dst_size);
    if (dst_ptr != nullptr) {
      if (src_ptr != nullptr) {
        Memcpy<device_type>(ctx, dst_ptr, src_ptr, cp_size);
      } else {
        Memset<device_type>(ctx, dst_ptr, 0, cp_size);
      }
      dst_ptr += cp_size;
    }
    if (src_ptr != nullptr) { src_ptr += cp_size; }
    src_size -= cp_size;
    dst_size -= cp_size;
  }
}

class DataContentIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataContentIterator);
  DataContentIterator() = delete;
  ~DataContentIterator() = default;

  DataContentIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    seg_num_ = BnInOp2Blob(bns->Get(0))->shape().Count(0, axis);
    seg_idx_ = 0;
    bns_ = bns;
    bn_idx_ = 0;
    axis_ = axis;
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (seg_idx_ == seg_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_));
    int64_t elem_num = blob->shape().Count(axis_);
    std::get<1>(ret) = elem_num * GetSizeOfDataType(blob->data_type());
    if (blob->IsColValid()) {
      std::get<0>(ret) = blob->mut_dptr<char>() + seg_idx_ * std::get<1>(ret);
    }
    bn_idx_ += 1;
    if (bn_idx_ == bns_->size()) {
      bn_idx_ = 0;
      seg_idx_ += 1;
    }
    return ret;
  }

  static CopyBlobFieldMthd GetCopyBlobFieldMthd() {
    return &Blob::CopyDataContentFrom;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  int64_t seg_num_;
  int64_t seg_idx_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t axis_;
};

class FieldIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FieldIterator);
  FieldIterator() = delete;
  virtual ~FieldIterator() = default;

  FieldIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    bns_ = bns;
    bn_idx_ = 0;
    if (axis == 0) {
      bn_num_ = bns_->size();
    } else {
      bn_num_ = 1;
    }
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (bn_idx_ == bn_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_++));
    std::get<0>(ret) = GetMutPtr(blob);
    std::get<1>(ret) = GetSizeOfField(blob);
    return ret;
  }

 protected:
  virtual char* GetMutPtr(Blob* blob) = 0;
  virtual size_t GetSizeOfField(Blob* blob) const = 0;

  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t bn_num_;
};

class DataIdIterator final : public FieldIterator {
 public:
  DataIdIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() {
    return &Blob::CopyDataIdFrom;
  }

 private:
  char* GetMutPtr(Blob* blob) override { return blob->mut_data_id(); }

  size_t GetSizeOfField(Blob* blob) const override {
    return blob->ByteSizeOfDataIdField();
  }
};

class ColNumIterator final : public FieldIterator {
 public:
  ColNumIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() {
    return &Blob::CopyColNumFrom;
  }

 private:
  char* GetMutPtr(Blob* blob) override {
    return reinterpret_cast<char*>(blob->mut_col_num());
  }

  size_t GetSizeOfField(Blob* blob) const override {
    return blob->ByteSizeOfColNumField();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ITER_UTIL_H_
