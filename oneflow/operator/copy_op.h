#ifndef ONEFLOW_OPERATOR_COPY_OP_H_
#define ONEFLOW_OPERATOR_COPY_OP_H_

namespace oneflow {

class CopyDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(CopyDataBlobDescSet);
  CopyDataBlobDescSet() = default;
  ~CopyDataBlobDescSet() = default;

  void Init(const google::protobuf::RepeatedPtrField<std::string>& logical_blob_names) {
    DataBlobDescSet::Init();

    input_blobs_.resize(logical_blob_names.size());
    output_blobs_.resize(logical_blob_names.size());
    for (int i = 0; i < logical_blob_names.size(); ++i) {
      RegisterInputBlobPptr(logical_blob_names.Get(i), &(input_blobs_[i]));
      RegisterOutputBlobPptr(logical_blob_names.Get(i), &(output_blobs_[i]));
    }
  }

 private:
  std::vector<BlobDescriptor*> input_blobs_;
  std::vector<BlobDescriptor*> output_blobs_;

};

class CopyModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(CopyModelBlobDescSet);
  CopyModelBlobDescSet() = default;
  ~CopyModelBlobDescSet() = default;

  void Init() {
    ModelBlobDescSet::Init();
  }

 private:

};

class CopyOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(CopyOp);
  CopyOp() = default;
  ~CopyOp() = default;

  void Init(const OperatorConf& op_conf) override {
    mutable_op_name() = op_conf.name();

    CHECK(op_conf.has_copy_op_conf());
    auto cnf_ptr = new CopyOpConf(op_conf.copy_op_conf());
    mutable_pb_op_conf().reset(cnf_ptr);
    
    auto data_ptr = new CopyDataBlobDescSet();
    data_ptr->Init(cnf_ptr->logical_blob_names());
    mutable_data_blob_desc_set().reset(data_ptr);

    auto model_ptr = new CopyModelBlobDescSet();
    model_ptr->Init();
    mutable_model_blob_desc_set().reset(model_ptr);
  }

  std::string ibn2lbn(const std::string& input_blob_name) const override {
    return input_blob_name;
  }
  std::string obn2lbn(const std::string& output_blob_name) const override {
    return output_blob_name;
  }

  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_COPY_OP_H_
