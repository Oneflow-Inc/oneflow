
  using ShapeInfo = std::vector<int>;
  using DtypeInfo = std::vector<dtype>;
  using OpBlobInfo = std::pair<ShapeInfo, DtypeInfo>;

  std::map<std::string, OpBlobInfo> inferface_name2info_;

  input_info(std::string input_name, std::string job_name = "");
  output_info(std::string input_name, std::string job_name = "");

  OpBlobInfo GetOpBlobInfo(std::string job_name, std::string op_name, std::string blob_name);

std::pair<std::vector<int>, const std::shared_ptr<const DType>&>
InferenceSession::GetOpBlobInfo(std::string job_name, 
                                std::string op_name, 
                                std::string blob_name) {
  std::vector<SessionStatus> status = {SessionStatus::OPEN, SessionStatus::RUNNING};
  this->CheckStatus(status);

  if (std::find(std::begin(this->inferface_name2info_), 
                std::end(this->inferface_name2info_), 
                op_name) != std::end(this->inferface_name2info_) {
    return this->inferface_name2info_[op_name];
  }

  if(job_name.empty()) job_name = this->cur_job_name_;
  CHECK_OR_RETURN(!job_name.empty()) << Error::ValueError(std::string("please specify job_name")); 

  std::string lbn = JobBuildAndInferCtx_GetOpBlobLbn(job_name, op_name, blob_name);
  Int64List int64_list;
  CHECK_OR_RETURN(TxtString2PbMessage(
    JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn), &int64_list))
    << "Int64List parse failed";
  std::vector<int> shape = int64_list.value();
  auto dtype = DType::Get(static_cast<DataType>(
      JobBuildAndInferCtx_GetDataType(job_name, lbn)
    )).GetOrThrow().get();
  auto info = std::pair<std::vector<int>, const std::shared_ptr<const DType>&>(shape, dtype);
  this->inferface_name2info_[op_name] = info;
  return info;
}

InferenceSession::input_info(std::string input_name, std::string job_name = "") {
  return this->GetOpBlobInfo(job_name, input_name, "out");
}

InferenceSession::output_info(std::string input_name, std::string job_name = "") {
  return this->GetOpBlobInfo(job_name, output_name, "in");
}

void SignatureProtoToCfg(const JobSignatureDef& signature_proto, 
                         cfg::JobSignatureDef& mut_signature_cfg) {}
  for (auto pair : signature_proto.inputs()) {
    std::string input_name = pair.first;
    const JobInputDef& input_def = pair.second;
    cfg::JobInputDef input_def_cfg;
    input_def_cfg.mutable_lbi()->set_op_name(input_def.lbi().op_name());
    input_def_cfg.mutable_lbi().set_blob_name(input_def.lbi().blob_name());
    InferfaceBlobConfProtoToCfg(input_def.blob_conf(), *input_def_cfg.mutable_blob_conf())
    mut_signature_cfg.mutable_inputs()->at(input_name).CopyFrom(input_def_cfg);
  }

  for (output_name, output_def) in signature_proto.outputs()) {
    std::string output_name = pair.first;
    const JobOutputDef& output_def = pair.second;
    cfg::JobOutputDef output_def_cfg;
    output_def_cfg.mutable_lbi()->set_op_name(output_def.lbi().op_name());
    output_def_cfg.mutable_lbi()->set_blob_name(output_def.lbi().blob_name());
    mut_signature_cfg.mutable_outputs()->at(output_name).CopyFrom(output_def_cfg);
  }
}

void InferfaceBlobConfProtoToCfg(const InterfaceBlobConf& inferface_blob_conf_proto,
                                 cfg::InterfaceBlobConf& mut_inferface_blob_conf_cfg) {
    shape = shape_proto_cfg.ShapeProto()
    for dim in inferface_blob_conf_proto.shape.dim:
        shape.add_dim(dim)
    mut_inferface_blob_conf_cfg.mutable_shape().CopyFrom(shape)
    dtype = dtype_proto_cfg.DataType(int(inferface_blob_conf_proto.data_type))
    mut_inferface_blob_conf_cfg.set_data_type(dtype)
    if inferface_blob_conf_proto.HasField("parallel_distribution"):
        assert len(inferface_blob_conf_proto.parallel_distribution.sbp_parallel) == 1
        sbp_proto = inferface_blob_conf_proto.parallel_distribution.sbp_parallel[0]
        if sbp_proto.HasField("split_parallel"):
            split_axis = sbp_proto.split_parallel.axis
            sbp = sbp_parallel_cfg.SbpParallel()
            sbp.mutable_split_parallel().set_axis(split_axis)
            mut_inferface_blob_conf_cfg.mutable_parallel_distribution().mutable_sbp_parallel().Add().CopyFrom(
                sbp
            )
    mut_inferface_blob_conf_cfg.set_is_dynamic(inferface_blob_conf_proto.is_dynamic)
}