/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// This file is generated automatically. Please DO NOT EDIT!

#ifdef DEFINE_OP_INTERP_CTX_CLASS

class COCOReaderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "session_id") {
      return CastAttr(&session_id);
    } else if (attr_name == "annotation_file") {
      return CastAttr(&annotation_file);
    } else if (attr_name == "image_dir") {
      return CastAttr(&image_dir);
    } else if (attr_name == "batch_size") {
      return CastAttr(&batch_size);
    } else if (attr_name == "shuffle_after_epoch") {
      return CastAttr(&shuffle_after_epoch);
    } else if (attr_name == "random_seed") {
      return CastAttr(&random_seed);
    } else if (attr_name == "group_by_ratio") {
      return CastAttr(&group_by_ratio);
    } else if (attr_name == "remove_images_without_annotations") {
      return CastAttr(&remove_images_without_annotations);
    } else if (attr_name == "stride_partition") {
      return CastAttr(&stride_partition);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "COCOReader op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "session_id",          "annotation_file",
        "image_dir",           "batch_size",
        "shuffle_after_epoch", "random_seed",
        "group_by_ratio",      "remove_images_without_annotations",
        "stride_partition",    "nd_sbp"};
    return attr_names;
  }

 public:
  int64_t session_id;
  std::string annotation_file;
  std::string image_dir;
  int64_t batch_size;
  bool shuffle_after_epoch;
  int64_t random_seed;
  bool group_by_ratio;
  bool remove_images_without_annotations;
  bool stride_partition;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.COCOReader", COCOReaderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CategoricalOrdinalEncodeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "hash_precomputed") {
      return CastAttr(&hash_precomputed);
    } else {
      return Error::RuntimeError()
             << "CategoricalOrdinalEncode op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"hash_precomputed"};
    return attr_names;
  }

 public:
  bool hash_precomputed;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.CategoricalOrdinalEncode", CategoricalOrdinalEncodeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OFRecordReaderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "data_dir") {
      return CastAttr(&data_dir);
    } else if (attr_name == "data_part_num") {
      return CastAttr(&data_part_num);
    } else if (attr_name == "batch_size") {
      return CastAttr(&batch_size);
    } else if (attr_name == "part_name_prefix") {
      return CastAttr(&part_name_prefix);
    } else if (attr_name == "part_name_suffix_length") {
      return CastAttr(&part_name_suffix_length);
    } else if (attr_name == "random_shuffle") {
      return CastAttr(&random_shuffle);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "shuffle_buffer_size") {
      return CastAttr(&shuffle_buffer_size);
    } else if (attr_name == "shuffle_after_epoch") {
      return CastAttr(&shuffle_after_epoch);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "OFRecordReader op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"data_dir",
                                           "data_part_num",
                                           "batch_size",
                                           "part_name_prefix",
                                           "part_name_suffix_length",
                                           "random_shuffle",
                                           "seed",
                                           "shuffle_buffer_size",
                                           "shuffle_after_epoch",
                                           "nd_sbp"};
    return attr_names;
  }

 public:
  std::string data_dir;
  int32_t data_part_num;
  int32_t batch_size;
  std::string part_name_prefix;
  int32_t part_name_suffix_length;
  bool random_shuffle;
  int64_t seed;
  int32_t shuffle_buffer_size;
  bool shuffle_after_epoch;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.OFRecordReader", OFRecordReaderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OneRecReaderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "files") {
      return CastAttr(&files);
    } else if (attr_name == "batch_size") {
      return CastAttr(&batch_size);
    } else if (attr_name == "random_shuffle") {
      return CastAttr(&random_shuffle);
    } else if (attr_name == "shuffle_mode") {
      return CastAttr(&shuffle_mode);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "shuffle_buffer_size") {
      return CastAttr(&shuffle_buffer_size);
    } else if (attr_name == "shuffle_after_epoch") {
      return CastAttr(&shuffle_after_epoch);
    } else if (attr_name == "verify_example") {
      return CastAttr(&verify_example);
    } else {
      return Error::RuntimeError() << "OneRecReader op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "files", "batch_size",          "random_shuffle",      "shuffle_mode",
        "seed",  "shuffle_buffer_size", "shuffle_after_epoch", "verify_example"};
    return attr_names;
  }

 public:
  std::vector<std::string> files;
  int32_t batch_size;
  bool random_shuffle;
  std::string shuffle_mode;
  int64_t seed;
  int32_t shuffle_buffer_size;
  bool shuffle_after_epoch;
  bool verify_example;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.OneRecReader", OneRecReaderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestDataTypeAttrOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_type") {
      return CastAttr(&output_type);
    } else {
      return Error::RuntimeError() << "TestDataTypeAttr op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_type"};
    return attr_names;
  }

 public:
  DataType output_type;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestDataTypeAttr", TestDataTypeAttrOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestDynamicSourceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TestDynamicSource op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestDynamicSource", TestDynamicSourceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestListDataTypeAndListShapeAndListStringAttrOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_shapes") {
      return CastAttr(&out_shapes);
    } else if (attr_name == "out_types") {
      return CastAttr(&out_types);
    } else if (attr_name == "string_list") {
      return CastAttr(&string_list);
    } else {
      return Error::RuntimeError()
             << "TestListDataTypeAndListShapeAndListStringAttr op has no attribute named "
             << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_shapes", "out_types", "string_list"};
    return attr_names;
  }

 public:
  std::vector<Shape> out_shapes;
  std::vector<DataType> out_types;
  std::vector<std::string> string_list;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestListDataTypeAndListShapeAndListStringAttr",
                       TestListDataTypeAndListShapeAndListStringAttrOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestMultiInputOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TestMultiInput op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestMultiInput", TestMultiInputOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestMultiInputGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TestMultiInputGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestMultiInputGrad", TestMultiInputGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestMultiOutputOrderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TestMultiOutputOrder op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestMultiOutputOrder", TestMultiOutputOrderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestRandomSourceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "seed") {
      return CastAttr(&seed);
    } else {
      return Error::RuntimeError() << "TestRandomSource op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"seed"};
    return attr_names;
  }

 public:
  int64_t seed;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestRandomSource", TestRandomSourceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestReshapeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "TestReshape op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"shape"};
    return attr_names;
  }

 public:
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestReshape", TestReshapeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestSourceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TestSource op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestSource", TestSourceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestSourceMultiGpuFixedOutNumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_num") {
      return CastAttr(&out_num);
    } else {
      return Error::RuntimeError()
             << "TestSourceMultiGpuFixedOutNum op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_num"};
    return attr_names;
  }

 public:
  int64_t out_num;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.TestSourceMultiGpuFixedOutNum",
                       TestSourceMultiGpuFixedOutNumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogical_2DSameDim0All2allOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_dim1_split_axis") {
      return CastAttr(&in_dim1_split_axis);
    } else if (attr_name == "out_dim1_split_axis") {
      return CastAttr(&out_dim1_split_axis);
    } else {
      return Error::RuntimeError()
             << "_ncclLogical_2DSameDim0All2all op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_dim1_split_axis", "out_dim1_split_axis"};
    return attr_names;
  }

 public:
  int64_t in_dim1_split_axis;
  int64_t out_dim1_split_axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all2all",
                       _ncclLogical_2DSameDim0All2allOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogical_2DSameDim0AllGatherOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "_ncclLogical_2DSameDim0AllGather op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all_gather",
                       _ncclLogical_2DSameDim0AllGatherOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogical_2DSameDim0AllGatherNoncontinuousOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_dim1_split_axis") {
      return CastAttr(&in_dim1_split_axis);
    } else {
      return Error::RuntimeError()
             << "_ncclLogical_2DSameDim0AllGatherNoncontinuous op has no attribute named "
             << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_dim1_split_axis"};
    return attr_names;
  }

 public:
  int64_t in_dim1_split_axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all_gather_noncontinuous",
                       _ncclLogical_2DSameDim0AllGatherNoncontinuousOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogical_2DSameDim0AllReduceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "_ncclLogical_2DSameDim0AllReduce op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all_reduce",
                       _ncclLogical_2DSameDim0AllReduceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogical_2DSameDim1AllReduceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "_ncclLogical_2DSameDim1AllReduce op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim1_all_reduce",
                       _ncclLogical_2DSameDim1AllReduceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogicalAllGatherOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "_ncclLogicalAllGather op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_all_gather", _ncclLogicalAllGatherOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogicalAllGatherNoncontinuousOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_split_axis") {
      return CastAttr(&in_split_axis);
    } else {
      return Error::RuntimeError()
             << "_ncclLogicalAllGatherNoncontinuous op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_split_axis"};
    return attr_names;
  }

 public:
  int64_t in_split_axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_all_gather_noncontinuous",
                       _ncclLogicalAllGatherNoncontinuousOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogicalAllReduceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "_ncclLogicalAllReduce op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_all_reduce", _ncclLogicalAllReduceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogicalReduceScatterOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "_ncclLogicalReduceScatter op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_reduce_scatter", _ncclLogicalReduceScatterOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class _ncclLogicalS2sOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_split_axis") {
      return CastAttr(&in_split_axis);
    } else if (attr_name == "out_split_axis") {
      return CastAttr(&out_split_axis);
    } else {
      return Error::RuntimeError() << "_ncclLogicalS2s op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_split_axis", "out_split_axis"};
    return attr_names;
  }

 public:
  int64_t in_split_axis;
  int64_t out_split_axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user._nccl_logical_s2s", _ncclLogicalS2sOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AbsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Abs op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.abs", AbsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AbsGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AbsGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.abs_grad", AbsGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AccOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "max_acc_num") {
      return CastAttr(&max_acc_num);
    } else {
      return Error::RuntimeError() << "Acc op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"max_acc_num"};
    return attr_names;
  }

 public:
  int32_t max_acc_num;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.acc", AccOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AcosOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Acos op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.acos", AcosOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AcosGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AcosGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.acos_grad", AcosGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AcoshOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Acosh op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.acosh", AcoshOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AcoshGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AcoshGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.acosh_grad", AcoshGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdagradUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "train_step_val") {
      return CastAttr(&train_step_val);
    } else if (attr_name == "learning_rate_val") {
      return CastAttr(&learning_rate_val);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "lr_decay") {
      return CastAttr(&lr_decay);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError() << "AdagradUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "train_step_val", "learning_rate_val", "scale",  "l1", "l2",
        "lr_decay",       "weight_decay",      "epsilon"};
    return attr_names;
  }

 public:
  int32_t train_step_val;
  float learning_rate_val;
  double scale;
  float l1;
  float l2;
  float lr_decay;
  float weight_decay;
  float epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adagrad_update", AdagradUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdamBiasCorrectionFactorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "beta") {
      return CastAttr(&beta);
    } else {
      return Error::RuntimeError()
             << "AdamBiasCorrectionFactor op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"beta"};
    return attr_names;
  }

 public:
  float beta;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adam_bias_correction_factor", AdamBiasCorrectionFactorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdamUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "learning_rate_val") {
      return CastAttr(&learning_rate_val);
    } else if (attr_name == "bias_correction1_val") {
      return CastAttr(&bias_correction1_val);
    } else if (attr_name == "bias_correction2_val") {
      return CastAttr(&bias_correction2_val);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "beta1") {
      return CastAttr(&beta1);
    } else if (attr_name == "beta2") {
      return CastAttr(&beta2);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else if (attr_name == "amsgrad") {
      return CastAttr(&amsgrad);
    } else if (attr_name == "do_bias_correction") {
      return CastAttr(&do_bias_correction);
    } else {
      return Error::RuntimeError() << "AdamUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"learning_rate_val",
                                           "bias_correction1_val",
                                           "bias_correction2_val",
                                           "scale",
                                           "l1",
                                           "l2",
                                           "beta1",
                                           "beta2",
                                           "epsilon",
                                           "weight_decay",
                                           "amsgrad",
                                           "do_bias_correction"};
    return attr_names;
  }

 public:
  float learning_rate_val;
  float bias_correction1_val;
  float bias_correction2_val;
  double scale;
  float l1;
  float l2;
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
  bool amsgrad;
  bool do_bias_correction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adam_update", AdamUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdaptiveAvgPool1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else {
      return Error::RuntimeError() << "AdaptiveAvgPool1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size"};
    return attr_names;
  }

 public:
  std::vector<int64_t> output_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool1d", AdaptiveAvgPool1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdaptiveAvgPool1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else {
      return Error::RuntimeError()
             << "AdaptiveAvgPool1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size"};
    return attr_names;
  }

 public:
  std::vector<int64_t> output_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool1d_grad", AdaptiveAvgPool1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdaptiveAvgPool2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else {
      return Error::RuntimeError() << "AdaptiveAvgPool2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size"};
    return attr_names;
  }

 public:
  std::vector<int64_t> output_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool2d", AdaptiveAvgPool2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdaptiveAvgPool2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else {
      return Error::RuntimeError()
             << "AdaptiveAvgPool2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size"};
    return attr_names;
  }

 public:
  std::vector<int64_t> output_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool2d_grad", AdaptiveAvgPool2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdaptiveAvgPool3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else {
      return Error::RuntimeError() << "AdaptiveAvgPool3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size"};
    return attr_names;
  }

 public:
  std::vector<int64_t> output_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool3d", AdaptiveAvgPool3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AdaptiveAvgPool3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else {
      return Error::RuntimeError()
             << "AdaptiveAvgPool3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size"};
    return attr_names;
  }

 public:
  std::vector<int64_t> output_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool3d_grad", AdaptiveAvgPool3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AddNOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AddN op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.add_n", AddNOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AffineGridOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "size") {
      return CastAttr(&size);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else {
      return Error::RuntimeError() << "AffineGrid op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"size", "align_corners"};
    return attr_names;
  }

 public:
  Shape size;
  bool align_corners;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.affine_grid", AffineGridOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AffineGridGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "size") {
      return CastAttr(&size);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else {
      return Error::RuntimeError() << "AffineGridGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"size", "align_corners"};
    return attr_names;
  }

 public:
  Shape size;
  bool align_corners;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.affine_grid_grad", AffineGridGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AmpWhiteIdentityOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AmpWhiteIdentity op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.amp_white_identity", AmpWhiteIdentityOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ArangeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "integer_start") {
      return CastAttr(&integer_start);
    } else if (attr_name == "integer_delta") {
      return CastAttr(&integer_delta);
    } else if (attr_name == "integer_limit") {
      return CastAttr(&integer_limit);
    } else if (attr_name == "float_start") {
      return CastAttr(&float_start);
    } else if (attr_name == "float_delta") {
      return CastAttr(&float_delta);
    } else if (attr_name == "float_limit") {
      return CastAttr(&float_limit);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Arange op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"integer_start", "integer_delta", "integer_limit",
                                           "float_start",   "float_delta",   "float_limit",
                                           "dtype",         "nd_sbp"};
    return attr_names;
  }

 public:
  int64_t integer_start;
  int64_t integer_delta;
  int64_t integer_limit;
  double float_start;
  double float_delta;
  double float_limit;
  DataType dtype;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.arange", ArangeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ArgSortOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "direction") {
      return CastAttr(&direction);
    } else {
      return Error::RuntimeError() << "ArgSort op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"direction"};
    return attr_names;
  }

 public:
  std::string direction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.arg_sort", ArgSortOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ArgmaxOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Argmax op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.argmax", ArgmaxOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ArgwhereOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else {
      return Error::RuntimeError() << "Argwhere op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dtype"};
    return attr_names;
  }

 public:
  DataType dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.argwhere", ArgwhereOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AsinOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Asin op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.asin", AsinOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AsinGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AsinGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.asin_grad", AsinGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AsinhOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Asinh op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.asinh", AsinhOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AsinhGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AsinhGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.asinh_grad", AsinhGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AssignOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Assign op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.assign", AssignOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AssignIfOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AssignIf op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.assign_if", AssignIfOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AssignIfNotOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AssignIfNot op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.assign_if_not", AssignIfNotOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AtanOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Atan op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atan", AtanOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Atan2OpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Atan2 op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atan2", Atan2OpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Atan2XGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Atan2XGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atan2_x_grad", Atan2XGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Atan2YGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Atan2YGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atan2_y_grad", Atan2YGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AtanGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AtanGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atan_grad", AtanGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AtanhOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Atanh op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atanh", AtanhOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AtanhGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "AtanhGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.atanh_grad", AtanhGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AvgPool1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else if (attr_name == "count_include_pad") {
      return CastAttr(&count_include_pad);
    } else if (attr_name == "divisor_override") {
      return CastAttr(&divisor_override);
    } else {
      return Error::RuntimeError() << "AvgPool1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",         "data_format", "kernel_size",
                                           "stride",          "ceil_mode",   "count_include_pad",
                                           "divisor_override"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.avgpool_1d", AvgPool1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AvgPool1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else if (attr_name == "count_include_pad") {
      return CastAttr(&count_include_pad);
    } else if (attr_name == "divisor_override") {
      return CastAttr(&divisor_override);
    } else {
      return Error::RuntimeError() << "AvgPool1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",         "data_format", "kernel_size",
                                           "stride",          "ceil_mode",   "count_include_pad",
                                           "divisor_override"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.avgpool_1d_grad", AvgPool1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AvgPool2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else if (attr_name == "count_include_pad") {
      return CastAttr(&count_include_pad);
    } else if (attr_name == "divisor_override") {
      return CastAttr(&divisor_override);
    } else {
      return Error::RuntimeError() << "AvgPool2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",         "data_format", "kernel_size",
                                           "stride",          "ceil_mode",   "count_include_pad",
                                           "divisor_override"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.avgpool_2d", AvgPool2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AvgPool2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else if (attr_name == "count_include_pad") {
      return CastAttr(&count_include_pad);
    } else if (attr_name == "divisor_override") {
      return CastAttr(&divisor_override);
    } else {
      return Error::RuntimeError() << "AvgPool2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",         "data_format", "kernel_size",
                                           "stride",          "ceil_mode",   "count_include_pad",
                                           "divisor_override"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.avgpool_2d_grad", AvgPool2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AvgPool3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else if (attr_name == "count_include_pad") {
      return CastAttr(&count_include_pad);
    } else if (attr_name == "divisor_override") {
      return CastAttr(&divisor_override);
    } else {
      return Error::RuntimeError() << "AvgPool3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",         "data_format", "kernel_size",
                                           "stride",          "ceil_mode",   "count_include_pad",
                                           "divisor_override"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.avgpool_3d", AvgPool3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class AvgPool3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else if (attr_name == "count_include_pad") {
      return CastAttr(&count_include_pad);
    } else if (attr_name == "divisor_override") {
      return CastAttr(&divisor_override);
    } else {
      return Error::RuntimeError() << "AvgPool3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",         "data_format", "kernel_size",
                                           "stride",          "ceil_mode",   "count_include_pad",
                                           "divisor_override"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode;
  bool count_include_pad;
  int64_t divisor_override;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.avgpool_3d_grad", AvgPool3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BatchGatherOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BatchGather op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.batch_gather", BatchGatherOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BatchMatmulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "transpose_a") {
      return CastAttr(&transpose_a);
    } else if (attr_name == "transpose_b") {
      return CastAttr(&transpose_b);
    } else if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "BatchMatmul op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"transpose_a", "transpose_b", "alpha"};
    return attr_names;
  }

 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.batch_matmul", BatchMatmulOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BernoulliOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "has_seed") {
      return CastAttr(&has_seed);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else {
      return Error::RuntimeError() << "Bernoulli op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"seed", "has_seed", "dtype"};
    return attr_names;
  }

 public:
  int64_t seed;
  bool has_seed;
  DataType dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.bernoulli", BernoulliOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BiasAddOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError() << "BiasAdd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int32_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.bias_add", BiasAddOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BinaryCrossEntropyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else {
      return Error::RuntimeError() << "BinaryCrossEntropy op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"reduction"};
    return attr_names;
  }

 public:
  std::string reduction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.binary_cross_entropy", BinaryCrossEntropyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BinaryCrossEntropyGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else {
      return Error::RuntimeError()
             << "BinaryCrossEntropyGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"reduction"};
    return attr_names;
  }

 public:
  std::string reduction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.binary_cross_entropy_grad", BinaryCrossEntropyGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BinaryCrossEntropyWithLogitsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_pos_weight") {
      return CastAttr(&has_pos_weight);
    } else if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else {
      return Error::RuntimeError()
             << "BinaryCrossEntropyWithLogits op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_pos_weight", "reduction"};
    return attr_names;
  }

 public:
  bool has_pos_weight;
  std::string reduction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.binary_cross_entropy_with_logits",
                       BinaryCrossEntropyWithLogitsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BinaryCrossEntropyWithLogitsGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_pos_weight") {
      return CastAttr(&has_pos_weight);
    } else if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else {
      return Error::RuntimeError()
             << "BinaryCrossEntropyWithLogitsGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_pos_weight", "reduction"};
    return attr_names;
  }

 public:
  bool has_pos_weight;
  std::string reduction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.binary_cross_entropy_with_logits_grad",
                       BinaryCrossEntropyWithLogitsGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastAddOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastAdd op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_add", BroadcastAddOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastDivOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastDiv op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_div", BroadcastDivOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastDivGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastDivGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_div_grad", BroadcastDivGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastEqual op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_equal", BroadcastEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastFloorModOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastFloorMod op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_floor_mod", BroadcastFloorModOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastFmodOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastFmod op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_fmod", BroadcastFmodOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastGreaterOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastGreater op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_greater", BroadcastGreaterOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastGreaterEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastGreaterEqual op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_greater_equal", BroadcastGreaterEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastLessOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastLess op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_less", BroadcastLessOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastLessEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastLessEqual op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_less_equal", BroadcastLessEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "broadcast_axes") {
      return CastAttr(&broadcast_axes);
    } else {
      return Error::RuntimeError() << "BroadcastLike op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"broadcast_axes"};
    return attr_names;
  }

 public:
  std::vector<int32_t> broadcast_axes;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_like", BroadcastLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastLogicalAndOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastLogicalAnd op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_logical_and", BroadcastLogicalAndOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastLogicalOrOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastLogicalOr op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_logical_or", BroadcastLogicalOrOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastLogicalXorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastLogicalXor op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_logical_xor", BroadcastLogicalXorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastMatmulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "transpose_a") {
      return CastAttr(&transpose_a);
    } else if (attr_name == "transpose_b") {
      return CastAttr(&transpose_b);
    } else if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "BroadcastMatmul op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"transpose_a", "transpose_b", "alpha"};
    return attr_names;
  }

 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_matmul", BroadcastMatmulOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastMatmulGradBOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError()
             << "BroadcastMatmulGradB op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_matmul_grad_b", BroadcastMatmulGradBOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastMaximumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastMaximum op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_maximum", BroadcastMaximumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastMinimumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastMinimum op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_minimum", BroadcastMinimumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastMulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastMul op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_mul", BroadcastMulOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastNotEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastNotEqual op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_not_equal", BroadcastNotEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastPowOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastPow op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_pow", BroadcastPowOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastPowXGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastPowXGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_pow_x_grad", BroadcastPowXGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastPowYGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastPowYGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_pow_y_grad", BroadcastPowYGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class BroadcastSubOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "BroadcastSub op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.broadcast_sub", BroadcastSubOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CastOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else {
      return Error::RuntimeError() << "Cast op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dtype"};
    return attr_names;
  }

 public:
  DataType dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cast", CastOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CastLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CastLike op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cast_like", CastLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CastToStaticShapeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CastToStaticShape op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cast_to_static_shape", CastToStaticShapeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CastToTickOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CastToTick op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cast_to_tick", CastToTickOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CcreluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Ccrelu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ccrelu", CcreluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CcreluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CcreluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ccrelu_grad", CcreluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CeilOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Ceil op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ceil", CeilOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CeilGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CeilGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ceil_grad", CeilGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CeluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "Celu op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.celu", CeluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CeluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "CeluGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.celu_grad", CeluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ClipByScalarOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_min") {
      return CastAttr(&floating_min);
    } else if (attr_name == "integral_min") {
      return CastAttr(&integral_min);
    } else if (attr_name == "floating_max") {
      return CastAttr(&floating_max);
    } else if (attr_name == "integral_max") {
      return CastAttr(&integral_max);
    } else {
      return Error::RuntimeError() << "ClipByScalar op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_min", "integral_min", "floating_max",
                                           "integral_max"};
    return attr_names;
  }

 public:
  double floating_min;
  int64_t integral_min;
  double floating_max;
  int64_t integral_max;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.clip_by_scalar", ClipByScalarOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ClipByScalarGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_min") {
      return CastAttr(&floating_min);
    } else if (attr_name == "integral_min") {
      return CastAttr(&integral_min);
    } else if (attr_name == "floating_max") {
      return CastAttr(&floating_max);
    } else if (attr_name == "integral_max") {
      return CastAttr(&integral_max);
    } else {
      return Error::RuntimeError() << "ClipByScalarGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_min", "integral_min", "floating_max",
                                           "integral_max"};
    return attr_names;
  }

 public:
  double floating_min;
  int64_t integral_min;
  double floating_max;
  int64_t integral_max;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.clip_by_scalar_grad", ClipByScalarGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ClipByScalarMaxOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_max") {
      return CastAttr(&floating_max);
    } else if (attr_name == "integral_max") {
      return CastAttr(&integral_max);
    } else {
      return Error::RuntimeError() << "ClipByScalarMax op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_max", "integral_max"};
    return attr_names;
  }

 public:
  double floating_max;
  int64_t integral_max;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.clip_by_scalar_max", ClipByScalarMaxOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ClipByScalarMaxGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_max") {
      return CastAttr(&floating_max);
    } else if (attr_name == "integral_max") {
      return CastAttr(&integral_max);
    } else {
      return Error::RuntimeError() << "ClipByScalarMaxGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_max", "integral_max"};
    return attr_names;
  }

 public:
  double floating_max;
  int64_t integral_max;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.clip_by_scalar_max_grad", ClipByScalarMaxGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ClipByScalarMinOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_min") {
      return CastAttr(&floating_min);
    } else if (attr_name == "integral_min") {
      return CastAttr(&integral_min);
    } else {
      return Error::RuntimeError() << "ClipByScalarMin op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_min", "integral_min"};
    return attr_names;
  }

 public:
  double floating_min;
  int64_t integral_min;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.clip_by_scalar_min", ClipByScalarMinOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ClipByScalarMinGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_min") {
      return CastAttr(&floating_min);
    } else if (attr_name == "integral_min") {
      return CastAttr(&integral_min);
    } else {
      return Error::RuntimeError() << "ClipByScalarMinGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_min", "integral_min"};
    return attr_names;
  }

 public:
  double floating_min;
  int64_t integral_min;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.clip_by_scalar_min_grad", ClipByScalarMinGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CoinFlipOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "probability") {
      return CastAttr(&probability);
    } else if (attr_name == "batch_size") {
      return CastAttr(&batch_size);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "has_seed") {
      return CastAttr(&has_seed);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "CoinFlip op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"probability", "batch_size", "seed", "has_seed",
                                           "nd_sbp"};
    return attr_names;
  }

 public:
  float probability;
  int64_t batch_size;
  int64_t seed;
  bool has_seed;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.coin_flip", CoinFlipOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CombinedMarginLossOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "m1") {
      return CastAttr(&m1);
    } else if (attr_name == "m2") {
      return CastAttr(&m2);
    } else if (attr_name == "m3") {
      return CastAttr(&m3);
    } else if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError() << "CombinedMarginLoss op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"m1", "m2", "m3", "depth"};
    return attr_names;
  }

 public:
  float m1;
  float m2;
  float m3;
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.combined_margin_loss", CombinedMarginLossOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CombinedMarginLossGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "m1") {
      return CastAttr(&m1);
    } else if (attr_name == "m2") {
      return CastAttr(&m2);
    } else if (attr_name == "m3") {
      return CastAttr(&m3);
    } else if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "CombinedMarginLossGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"m1", "m2", "m3", "depth"};
    return attr_names;
  }

 public:
  float m1;
  float m2;
  float m3;
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.combined_margin_loss_grad", CombinedMarginLossGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConcatOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "max_dim_size") {
      return CastAttr(&max_dim_size);
    } else {
      return Error::RuntimeError() << "Concat op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "max_dim_size"};
    return attr_names;
  }

 public:
  int64_t axis;
  int64_t max_dim_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.concat", ConcatOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integer_value") {
      return CastAttr(&integer_value);
    } else if (attr_name == "is_floating_value") {
      return CastAttr(&is_floating_value);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Constant op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"floating_value", "integer_value", "is_floating_value",
                                           "dtype",          "shape",         "nd_sbp"};
    return attr_names;
  }

 public:
  double floating_value;
  int64_t integer_value;
  bool is_floating_value;
  DataType dtype;
  Shape shape;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant", ConstantOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantPad1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integral_value") {
      return CastAttr(&integral_value);
    } else {
      return Error::RuntimeError() << "ConstantPad1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "floating_value", "integral_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
  double floating_value;
  int64_t integral_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant_pad1d", ConstantPad1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantPad1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integral_value") {
      return CastAttr(&integral_value);
    } else {
      return Error::RuntimeError() << "ConstantPad1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "floating_value", "integral_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
  double floating_value;
  int64_t integral_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant_pad1d_grad", ConstantPad1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantPad2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integral_value") {
      return CastAttr(&integral_value);
    } else {
      return Error::RuntimeError() << "ConstantPad2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "floating_value", "integral_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
  double floating_value;
  int64_t integral_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant_pad2d", ConstantPad2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantPad2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integral_value") {
      return CastAttr(&integral_value);
    } else {
      return Error::RuntimeError() << "ConstantPad2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "floating_value", "integral_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
  double floating_value;
  int64_t integral_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant_pad2d_grad", ConstantPad2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantPad3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integral_value") {
      return CastAttr(&integral_value);
    } else {
      return Error::RuntimeError() << "ConstantPad3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "floating_value", "integral_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
  double floating_value;
  int64_t integral_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant_pad3d", ConstantPad3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConstantPad3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "floating_value") {
      return CastAttr(&floating_value);
    } else if (attr_name == "integral_value") {
      return CastAttr(&integral_value);
    } else {
      return Error::RuntimeError() << "ConstantPad3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "floating_value", "integral_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
  double floating_value;
  int64_t integral_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.constant_pad3d_grad", ConstantPad3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Conv1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "filters") {
      return CastAttr(&filters);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "Conv1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"filters",     "padding_before", "data_format",
                                           "kernel_size", "strides",        "dilation_rate",
                                           "groups"};
    return attr_names;
  }

 public:
  int32_t filters;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.conv1d", Conv1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Conv2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "filters") {
      return CastAttr(&filters);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "Conv2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"filters",     "padding_before", "data_format",
                                           "kernel_size", "strides",        "dilation_rate",
                                           "groups"};
    return attr_names;
  }

 public:
  int32_t filters;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.conv2d", Conv2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Conv3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "filters") {
      return CastAttr(&filters);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "Conv3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"filters",     "padding_before", "data_format",
                                           "kernel_size", "strides",        "dilation_rate",
                                           "groups"};
    return attr_names;
  }

 public:
  int32_t filters;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.conv3d", Conv3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConvBiasGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "num_spatial_dims") {
      return CastAttr(&num_spatial_dims);
    } else {
      return Error::RuntimeError() << "ConvBiasGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"data_format", "num_spatial_dims"};
    return attr_names;
  }

 public:
  std::string data_format;
  int32_t num_spatial_dims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.conv_bias_grad", ConvBiasGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConvDataGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "num_spatial_dims") {
      return CastAttr(&num_spatial_dims);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "ConvDataGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "num_spatial_dims", "padding_before", "data_format", "kernel_size",
        "strides",          "dilation_rate",  "groups"};
    return attr_names;
  }

 public:
  int32_t num_spatial_dims;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.conv_data_grad", ConvDataGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ConvFilterGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "num_spatial_dims") {
      return CastAttr(&num_spatial_dims);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "ConvFilterGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "num_spatial_dims", "padding_before", "data_format", "kernel_size",
        "strides",          "dilation_rate",  "groups"};
    return attr_names;
  }

 public:
  int32_t num_spatial_dims;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.conv_filter_grad", ConvFilterGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CopyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "device_type") {
      return CastAttr(&device_type);
    } else if (attr_name == "device_id") {
      return CastAttr(&device_id);
    } else {
      return Error::RuntimeError() << "Copy op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"device_type", "device_id"};
    return attr_names;
  }

 public:
  std::string device_type;
  int64_t device_id;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.copy", CopyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CosOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Cos op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cos", CosOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CosGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CosGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cos_grad", CosGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CoshOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Cosh op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cosh", CoshOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CoshGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CoshGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cosh_grad", CoshGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CountNotFiniteOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CountNotFinite op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.count_not_finite", CountNotFiniteOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CpuOnlyReluTestOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "CpuOnlyReluTest op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cpu_only_relu_test", CpuOnlyReluTestOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CreateSummaryWriterOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "logdir") {
      return CastAttr(&logdir);
    } else {
      return Error::RuntimeError() << "CreateSummaryWriter op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"logdir"};
    return attr_names;
  }

 public:
  std::string logdir;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.create_summary_writer", CreateSummaryWriterOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CropMirrorNormalizeFromTensorbufferOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "color_space") {
      return CastAttr(&color_space);
    } else if (attr_name == "output_layout") {
      return CastAttr(&output_layout);
    } else if (attr_name == "mean") {
      return CastAttr(&mean);
    } else if (attr_name == "std") {
      return CastAttr(&std);
    } else if (attr_name == "crop_h") {
      return CastAttr(&crop_h);
    } else if (attr_name == "crop_w") {
      return CastAttr(&crop_w);
    } else if (attr_name == "crop_pos_x") {
      return CastAttr(&crop_pos_x);
    } else if (attr_name == "crop_pos_y") {
      return CastAttr(&crop_pos_y);
    } else if (attr_name == "output_dtype") {
      return CastAttr(&output_dtype);
    } else {
      return Error::RuntimeError()
             << "CropMirrorNormalizeFromTensorbuffer op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"color_space", "output_layout", "mean",
                                           "std",         "crop_h",        "crop_w",
                                           "crop_pos_x",  "crop_pos_y",    "output_dtype"};
    return attr_names;
  }

 public:
  std::string color_space;
  std::string output_layout;
  std::vector<float> mean;
  std::vector<float> std;
  int64_t crop_h;
  int64_t crop_w;
  float crop_pos_x;
  float crop_pos_y;
  DataType output_dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.crop_mirror_normalize_from_tensorbuffer",
                       CropMirrorNormalizeFromTensorbufferOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CropMirrorNormalizeFromUint8OpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "color_space") {
      return CastAttr(&color_space);
    } else if (attr_name == "output_layout") {
      return CastAttr(&output_layout);
    } else if (attr_name == "mean") {
      return CastAttr(&mean);
    } else if (attr_name == "std") {
      return CastAttr(&std);
    } else if (attr_name == "crop_h") {
      return CastAttr(&crop_h);
    } else if (attr_name == "crop_w") {
      return CastAttr(&crop_w);
    } else if (attr_name == "crop_pos_x") {
      return CastAttr(&crop_pos_x);
    } else if (attr_name == "crop_pos_y") {
      return CastAttr(&crop_pos_y);
    } else if (attr_name == "output_dtype") {
      return CastAttr(&output_dtype);
    } else {
      return Error::RuntimeError()
             << "CropMirrorNormalizeFromUint8 op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"color_space", "output_layout", "mean",
                                           "std",         "crop_h",        "crop_w",
                                           "crop_pos_x",  "crop_pos_y",    "output_dtype"};
    return attr_names;
  }

 public:
  std::string color_space;
  std::string output_layout;
  std::vector<float> mean;
  std::vector<float> std;
  int64_t crop_h;
  int64_t crop_w;
  float crop_pos_x;
  float crop_pos_y;
  DataType output_dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.crop_mirror_normalize_from_uint8",
                       CropMirrorNormalizeFromUint8OpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CtcGreedyDecoderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "merge_repeated") {
      return CastAttr(&merge_repeated);
    } else {
      return Error::RuntimeError() << "CtcGreedyDecoder op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"merge_repeated"};
    return attr_names;
  }

 public:
  bool merge_repeated;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ctc_greedy_decoder", CtcGreedyDecoderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CtcLossOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "max_target_length") {
      return CastAttr(&max_target_length);
    } else if (attr_name == "blank") {
      return CastAttr(&blank);
    } else if (attr_name == "zero_infinity") {
      return CastAttr(&zero_infinity);
    } else {
      return Error::RuntimeError() << "CtcLoss op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"max_target_length", "blank", "zero_infinity"};
    return attr_names;
  }

 public:
  int64_t max_target_length;
  int32_t blank;
  bool zero_infinity;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ctc_loss", CtcLossOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CtcLossGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "max_target_length") {
      return CastAttr(&max_target_length);
    } else if (attr_name == "blank") {
      return CastAttr(&blank);
    } else if (attr_name == "zero_infinity") {
      return CastAttr(&zero_infinity);
    } else {
      return Error::RuntimeError() << "CtcLossGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"max_target_length", "blank", "zero_infinity"};
    return attr_names;
  }

 public:
  int64_t max_target_length;
  int32_t blank;
  bool zero_infinity;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ctc_loss_grad", CtcLossGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CudnnFusedNormalizationAddReluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "momentum") {
      return CastAttr(&momentum);
    } else {
      return Error::RuntimeError()
             << "CudnnFusedNormalizationAddRelu op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon", "momentum"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
  float momentum;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cudnn_fused_normalization_add_relu",
                       CudnnFusedNormalizationAddReluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class CudnnFusedNormalizationAddReluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError()
             << "CudnnFusedNormalizationAddReluGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.cudnn_fused_normalization_add_relu_grad",
                       CudnnFusedNormalizationAddReluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Deconv1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "filters") {
      return CastAttr(&filters);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "output_padding") {
      return CastAttr(&output_padding);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "Deconv1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"filters",       "padding_before", "data_format",
                                           "kernel_size",   "output_padding", "strides",
                                           "dilation_rate", "groups"};
    return attr_names;
  }

 public:
  int32_t filters;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> output_padding;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.deconv1d", Deconv1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Deconv2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "filters") {
      return CastAttr(&filters);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "output_padding") {
      return CastAttr(&output_padding);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "Deconv2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"filters",       "padding_before", "data_format",
                                           "kernel_size",   "output_padding", "strides",
                                           "dilation_rate", "groups"};
    return attr_names;
  }

 public:
  int32_t filters;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> output_padding;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.deconv2d", Deconv2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Deconv3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "filters") {
      return CastAttr(&filters);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "output_padding") {
      return CastAttr(&output_padding);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else if (attr_name == "groups") {
      return CastAttr(&groups);
    } else {
      return Error::RuntimeError() << "Deconv3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"filters",       "padding_before", "data_format",
                                           "kernel_size",   "output_padding", "strides",
                                           "dilation_rate", "groups"};
    return attr_names;
  }

 public:
  int32_t filters;
  std::vector<int32_t> padding_before;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> output_padding;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.deconv3d", Deconv3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DiagOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else {
      return Error::RuntimeError() << "Diag op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal"};
    return attr_names;
  }

 public:
  int32_t diagonal;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.diag", DiagOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DiagGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else {
      return Error::RuntimeError() << "DiagGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal"};
    return attr_names;
  }

 public:
  int32_t diagonal;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.diag_grad", DiagGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimGatherOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimGather op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim"};
    return attr_names;
  }

 public:
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_gather", DimGatherOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterAddOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimScatterAdd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim"};
    return attr_names;
  }

 public:
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_add", DimScatterAddOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterAddLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimScatterAddLike op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim"};
    return attr_names;
  }

 public:
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_add_like", DimScatterAddLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterAddScalarOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "src_scalar") {
      return CastAttr(&src_scalar);
    } else if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimScatterAddScalar op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"src_scalar", "dim"};
    return attr_names;
  }

 public:
  float src_scalar;
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_add_scalar", DimScatterAddScalarOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterMulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimScatterMul op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim"};
    return attr_names;
  }

 public:
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_mul", DimScatterMulOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterMulScalarOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "src_scalar") {
      return CastAttr(&src_scalar);
    } else if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimScatterMulScalar op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"src_scalar", "dim"};
    return attr_names;
  }

 public:
  float src_scalar;
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_mul_scalar", DimScatterMulScalarOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError() << "DimScatterUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim"};
    return attr_names;
  }

 public:
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_update", DimScatterUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DimScatterUpdateScalarOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "src_scalar") {
      return CastAttr(&src_scalar);
    } else if (attr_name == "dim") {
      return CastAttr(&dim);
    } else {
      return Error::RuntimeError()
             << "DimScatterUpdateScalar op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"src_scalar", "dim"};
    return attr_names;
  }

 public:
  float src_scalar;
  int32_t dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dim_scatter_update_scalar", DimScatterUpdateScalarOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DistributedPartialFcSampleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "num_sample") {
      return CastAttr(&num_sample);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else {
      return Error::RuntimeError()
             << "DistributedPartialFcSample op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"num_sample", "seed"};
    return attr_names;
  }

 public:
  int64_t num_sample;
  int64_t seed;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.distributed_partial_fc_sample", DistributedPartialFcSampleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DistributedPartialFcSampleDisableBoxingOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError()
           << "DistributedPartialFcSampleDisableBoxing op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.distributed_partial_fc_sample_disable_boxing",
                       DistributedPartialFcSampleDisableBoxingOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DropoutOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale") {
      return CastAttr(&scale);
    } else {
      return Error::RuntimeError() << "Dropout op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale"};
    return attr_names;
  }

 public:
  float scale;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dropout", DropoutOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DropoutGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale") {
      return CastAttr(&scale);
    } else {
      return Error::RuntimeError() << "DropoutGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale"};
    return attr_names;
  }

 public:
  float scale;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dropout_grad", DropoutGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class DynamicLossScaleScheduleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "increment_period") {
      return CastAttr(&increment_period);
    } else if (attr_name == "multiplier") {
      return CastAttr(&multiplier);
    } else {
      return Error::RuntimeError()
             << "DynamicLossScaleSchedule op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"increment_period", "multiplier"};
    return attr_names;
  }

 public:
  int64_t increment_period;
  float multiplier;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.dynamic_loss_scale_schedule", DynamicLossScaleScheduleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerBToSOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_split_axis") {
      return CastAttr(&out_split_axis);
    } else if (attr_name == "in_parallel_conf") {
      return CastAttr(&in_parallel_conf);
    } else if (attr_name == "out_parallel_conf") {
      return CastAttr(&out_parallel_conf);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "EagerBToS op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_split_axis", "in_parallel_conf",
                                           "out_parallel_conf", "shape"};
    return attr_names;
  }

 public:
  int64_t out_split_axis;
  std::string in_parallel_conf;
  std::string out_parallel_conf;
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_b_to_s", EagerBToSOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNaiveSToSOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_split_axis") {
      return CastAttr(&in_split_axis);
    } else if (attr_name == "out_split_axis") {
      return CastAttr(&out_split_axis);
    } else if (attr_name == "in_parallel_conf") {
      return CastAttr(&in_parallel_conf);
    } else if (attr_name == "out_parallel_conf") {
      return CastAttr(&out_parallel_conf);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "EagerNaiveSToS op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_split_axis", "out_split_axis", "in_parallel_conf",
                                           "out_parallel_conf", "shape"};
    return attr_names;
  }

 public:
  int64_t in_split_axis;
  int64_t out_split_axis;
  std::string in_parallel_conf;
  std::string out_parallel_conf;
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_naive_s_to_s", EagerNaiveSToSOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNcclAllGatherOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else {
      return Error::RuntimeError() << "EagerNcclAllGather op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"parallel_conf"};
    return attr_names;
  }

 public:
  std::string parallel_conf;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_nccl_all_gather", EagerNcclAllGatherOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNcclAllReduceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else if (attr_name == "async_launch") {
      return CastAttr(&async_launch);
    } else {
      return Error::RuntimeError() << "EagerNcclAllReduce op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"parallel_conf", "async_launch"};
    return attr_names;
  }

 public:
  std::string parallel_conf;
  bool async_launch;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_nccl_all_reduce", EagerNcclAllReduceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNcclBroadcastOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else if (attr_name == "root") {
      return CastAttr(&root);
    } else {
      return Error::RuntimeError() << "EagerNcclBroadcast op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"parallel_conf", "root"};
    return attr_names;
  }

 public:
  std::string parallel_conf;
  int64_t root;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_nccl_broadcast", EagerNcclBroadcastOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNcclReduceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else if (attr_name == "root") {
      return CastAttr(&root);
    } else {
      return Error::RuntimeError() << "EagerNcclReduce op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"parallel_conf", "root"};
    return attr_names;
  }

 public:
  std::string parallel_conf;
  int64_t root;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_nccl_reduce", EagerNcclReduceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNcclReduceScatterOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else if (attr_name == "op_type") {
      return CastAttr(&op_type);
    } else {
      return Error::RuntimeError()
             << "EagerNcclReduceScatter op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"parallel_conf", "op_type"};
    return attr_names;
  }

 public:
  std::string parallel_conf;
  std::string op_type;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_nccl_reduce_scatter", EagerNcclReduceScatterOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerNcclS2sOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_split_axis") {
      return CastAttr(&in_split_axis);
    } else if (attr_name == "out_split_axis") {
      return CastAttr(&out_split_axis);
    } else if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else {
      return Error::RuntimeError() << "EagerNcclS2s op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_split_axis", "out_split_axis", "parallel_conf"};
    return attr_names;
  }

 public:
  int64_t in_split_axis;
  int64_t out_split_axis;
  std::string parallel_conf;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_nccl_s2s", EagerNcclS2sOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerPToBOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_parallel_conf") {
      return CastAttr(&in_parallel_conf);
    } else if (attr_name == "out_parallel_conf") {
      return CastAttr(&out_parallel_conf);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "EagerPToB op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_parallel_conf", "out_parallel_conf", "shape"};
    return attr_names;
  }

 public:
  std::string in_parallel_conf;
  std::string out_parallel_conf;
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_p_to_b", EagerPToBOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerPToSOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_split_axis") {
      return CastAttr(&out_split_axis);
    } else if (attr_name == "in_parallel_conf") {
      return CastAttr(&in_parallel_conf);
    } else if (attr_name == "out_parallel_conf") {
      return CastAttr(&out_parallel_conf);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "EagerPToS op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_split_axis", "in_parallel_conf",
                                           "out_parallel_conf", "shape"};
    return attr_names;
  }

 public:
  int64_t out_split_axis;
  std::string in_parallel_conf;
  std::string out_parallel_conf;
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_p_to_s", EagerPToSOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerSToBOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_split_axis") {
      return CastAttr(&in_split_axis);
    } else if (attr_name == "in_parallel_conf") {
      return CastAttr(&in_parallel_conf);
    } else if (attr_name == "out_parallel_conf") {
      return CastAttr(&out_parallel_conf);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "EagerSToB op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_split_axis", "in_parallel_conf", "out_parallel_conf",
                                           "shape"};
    return attr_names;
  }

 public:
  int64_t in_split_axis;
  std::string in_parallel_conf;
  std::string out_parallel_conf;
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_s_to_b", EagerSToBOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EagerSymmetricSToPOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "in_split_axis") {
      return CastAttr(&in_split_axis);
    } else if (attr_name == "parallel_conf") {
      return CastAttr(&parallel_conf);
    } else {
      return Error::RuntimeError() << "EagerSymmetricSToP op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"in_split_axis", "parallel_conf"};
    return attr_names;
  }

 public:
  int64_t in_split_axis;
  std::string parallel_conf;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eager_symmetric_s_to_p", EagerSymmetricSToPOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ElementwiseMaximumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ElementwiseMaximum op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.elementwise_maximum", ElementwiseMaximumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ElementwiseMaximumBackwardOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ElementwiseMaximumBackward op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.elementwise_maximum_backward", ElementwiseMaximumBackwardOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ElementwiseMinimumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ElementwiseMinimum op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.elementwise_minimum", ElementwiseMinimumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ElementwiseMinimumBackwardOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ElementwiseMinimumBackward op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.elementwise_minimum_backward", ElementwiseMinimumBackwardOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "Elu op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.elu", EluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "EluGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.elu_grad", EluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EmptyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Empty op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dtype", "shape", "nd_sbp"};
    return attr_names;
  }

 public:
  DataType dtype;
  Shape shape;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.empty", EmptyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ErfOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Erf op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.erf", ErfOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ErfGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ErfGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.erf_grad", ErfGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ErfcOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Erfc op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.erfc", ErfcOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ErfcGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ErfcGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.erfc_grad", ErfcGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ExpOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Exp op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.exp", ExpOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ExpGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ExpGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.exp_grad", ExpGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ExpandOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "logical_in_shape") {
      return CastAttr(&logical_in_shape);
    } else if (attr_name == "logical_expand_shape") {
      return CastAttr(&logical_expand_shape);
    } else {
      return Error::RuntimeError() << "Expand op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"logical_in_shape", "logical_expand_shape"};
    return attr_names;
  }

 public:
  std::vector<int32_t> logical_in_shape;
  std::vector<int32_t> logical_expand_shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.expand", ExpandOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ExpandDimsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError() << "ExpandDims op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int32_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.expand_dims", ExpandDimsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ExpandGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "logical_out_shape") {
      return CastAttr(&logical_out_shape);
    } else if (attr_name == "logical_expand_shape") {
      return CastAttr(&logical_expand_shape);
    } else {
      return Error::RuntimeError() << "ExpandGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"logical_out_shape", "logical_expand_shape"};
    return attr_names;
  }

 public:
  std::vector<int32_t> logical_out_shape;
  std::vector<int32_t> logical_expand_shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.expand_grad", ExpandGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Expm1OpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Expm1 op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.expm1", Expm1OpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Expm1GradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Expm1Grad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.expm1_grad", Expm1GradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class EyeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "rows") {
      return CastAttr(&rows);
    } else if (attr_name == "cols") {
      return CastAttr(&cols);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Eye op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"rows", "cols", "dtype", "nd_sbp"};
    return attr_names;
  }

 public:
  int64_t rows;
  int64_t cols;
  DataType dtype;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.eye", EyeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FakeQuantizationOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "quantization_formula") {
      return CastAttr(&quantization_formula);
    } else if (attr_name == "quantization_bit") {
      return CastAttr(&quantization_bit);
    } else if (attr_name == "quantization_scheme") {
      return CastAttr(&quantization_scheme);
    } else {
      return Error::RuntimeError() << "FakeQuantization op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"quantization_formula", "quantization_bit",
                                           "quantization_scheme"};
    return attr_names;
  }

 public:
  std::string quantization_formula;
  int32_t quantization_bit;
  std::string quantization_scheme;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fake_quantization", FakeQuantizationOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FlattenOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "start_dim") {
      return CastAttr(&start_dim);
    } else if (attr_name == "end_dim") {
      return CastAttr(&end_dim);
    } else {
      return Error::RuntimeError() << "Flatten op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"start_dim", "end_dim"};
    return attr_names;
  }

 public:
  int32_t start_dim;
  int32_t end_dim;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.flatten", FlattenOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FlipOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dims") {
      return CastAttr(&dims);
    } else {
      return Error::RuntimeError() << "Flip op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> dims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.flip", FlipOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FlipGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dims") {
      return CastAttr(&dims);
    } else {
      return Error::RuntimeError() << "FlipGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> dims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.flip_grad", FlipGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FloorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Floor op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.floor", FloorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FloorGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "FloorGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.floor_grad", FloorGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FloordivOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Floordiv op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.floordiv", FloordivOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FloordivXGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "FloordivXGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.floordiv_x_grad", FloordivXGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FloordivYGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "FloordivYGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.floordiv_y_grad", FloordivYGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FlushSummaryWriterOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "FlushSummaryWriter op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.flush_summary_writer", FlushSummaryWriterOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FoldOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "output_size") {
      return CastAttr(&output_size);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else {
      return Error::RuntimeError() << "Fold op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"output_size", "kernel_size", "strides", "padding",
                                           "dilation_rate"};
    return attr_names;
  }

 public:
  std::vector<int32_t> output_size;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> padding;
  std::vector<int32_t> dilation_rate;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fold", FoldOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedBiasAddGeluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError() << "FusedBiasAddGelu op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int32_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_bias_add_gelu", FusedBiasAddGeluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedBiasAddGeluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError()
             << "FusedBiasAddGeluGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int32_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_bias_add_gelu_grad", FusedBiasAddGeluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedBiasAddMaskScaleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else {
      return Error::RuntimeError()
             << "FusedBiasAddMaskScale op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "scale"};
    return attr_names;
  }

 public:
  int32_t axis;
  float scale;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_bias_add_mask_scale", FusedBiasAddMaskScaleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedCastScaleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale") {
      return CastAttr(&scale);
    } else {
      return Error::RuntimeError() << "FusedCastScale op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale"};
    return attr_names;
  }

 public:
  double scale;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_cast_scale", FusedCastScaleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedScaleMaskSoftmaxOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_value") {
      return CastAttr(&scale_value);
    } else if (attr_name == "mask_fill_value") {
      return CastAttr(&mask_fill_value);
    } else {
      return Error::RuntimeError()
             << "FusedScaleMaskSoftmax op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_value", "mask_fill_value"};
    return attr_names;
  }

 public:
  float scale_value;
  float mask_fill_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax", FusedScaleMaskSoftmaxOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedScaleMaskSoftmaxDropoutOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_value") {
      return CastAttr(&scale_value);
    } else if (attr_name == "mask_fill_value") {
      return CastAttr(&mask_fill_value);
    } else if (attr_name == "dropout_scale_value") {
      return CastAttr(&dropout_scale_value);
    } else {
      return Error::RuntimeError()
             << "FusedScaleMaskSoftmaxDropout op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_value", "mask_fill_value", "dropout_scale_value"};
    return attr_names;
  }

 public:
  float scale_value;
  float mask_fill_value;
  float dropout_scale_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax_dropout",
                       FusedScaleMaskSoftmaxDropoutOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedScaleMaskSoftmaxDropoutGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_value") {
      return CastAttr(&scale_value);
    } else if (attr_name == "dropout_scale_value") {
      return CastAttr(&dropout_scale_value);
    } else {
      return Error::RuntimeError()
             << "FusedScaleMaskSoftmaxDropoutGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_value", "dropout_scale_value"};
    return attr_names;
  }

 public:
  float scale_value;
  float dropout_scale_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax_dropout_grad",
                       FusedScaleMaskSoftmaxDropoutGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedScaleMaskSoftmaxGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_value") {
      return CastAttr(&scale_value);
    } else {
      return Error::RuntimeError()
             << "FusedScaleMaskSoftmaxGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_value"};
    return attr_names;
  }

 public:
  float scale_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax_grad", FusedScaleMaskSoftmaxGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedScaleTrilOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else if (attr_name == "floating_fill_value") {
      return CastAttr(&floating_fill_value);
    } else if (attr_name == "integer_fill_value") {
      return CastAttr(&integer_fill_value);
    } else if (attr_name == "is_floating_fill_value") {
      return CastAttr(&is_floating_fill_value);
    } else if (attr_name == "floating_scale_value") {
      return CastAttr(&floating_scale_value);
    } else if (attr_name == "integer_scale_value") {
      return CastAttr(&integer_scale_value);
    } else if (attr_name == "is_floating_scale_value") {
      return CastAttr(&is_floating_scale_value);
    } else {
      return Error::RuntimeError() << "FusedScaleTril op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal",
                                           "floating_fill_value",
                                           "integer_fill_value",
                                           "is_floating_fill_value",
                                           "floating_scale_value",
                                           "integer_scale_value",
                                           "is_floating_scale_value"};
    return attr_names;
  }

 public:
  int64_t diagonal;
  double floating_fill_value;
  int64_t integer_fill_value;
  bool is_floating_fill_value;
  double floating_scale_value;
  int64_t integer_scale_value;
  bool is_floating_scale_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_scale_tril", FusedScaleTrilOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedSelfAttentionQueryMulKeyAndValueOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "head_size") {
      return CastAttr(&head_size);
    } else if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError()
             << "FusedSelfAttentionQueryMulKeyAndValue op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"head_size", "alpha"};
    return attr_names;
  }

 public:
  int64_t head_size;
  float alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_self_attention_query_mul_key_and_value",
                       FusedSelfAttentionQueryMulKeyAndValueOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedSelfAttentionQueryMulKeyAndValueGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError()
             << "FusedSelfAttentionQueryMulKeyAndValueGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  float alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_self_attention_query_mul_key_and_value_grad",
                       FusedSelfAttentionQueryMulKeyAndValueGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedTrilScaleSoftmaxMaskScaleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else if (attr_name == "tril_fill_value") {
      return CastAttr(&tril_fill_value);
    } else if (attr_name == "tril_scale_value") {
      return CastAttr(&tril_scale_value);
    } else if (attr_name == "mask_scale_value") {
      return CastAttr(&mask_scale_value);
    } else {
      return Error::RuntimeError()
             << "FusedTrilScaleSoftmaxMaskScale op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal", "tril_fill_value", "tril_scale_value",
                                           "mask_scale_value"};
    return attr_names;
  }

 public:
  int64_t diagonal;
  float tril_fill_value;
  float tril_scale_value;
  float mask_scale_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_tril_scale_softmax_mask_scale",
                       FusedTrilScaleSoftmaxMaskScaleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class FusedTrilScaleSoftmaxMaskScaleGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else if (attr_name == "tril_scale_value") {
      return CastAttr(&tril_scale_value);
    } else if (attr_name == "mask_scale_value") {
      return CastAttr(&mask_scale_value);
    } else {
      return Error::RuntimeError()
             << "FusedTrilScaleSoftmaxMaskScaleGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal", "tril_scale_value", "mask_scale_value"};
    return attr_names;
  }

 public:
  int64_t diagonal;
  float tril_scale_value;
  float mask_scale_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.fused_tril_scale_softmax_mask_scale_grad",
                       FusedTrilScaleSoftmaxMaskScaleGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GatherOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError() << "Gather op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int64_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.gather", GatherOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GatherNdOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "GatherNd op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.gather_nd", GatherNdOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GeluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Gelu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.gelu", GeluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GeluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "GeluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.gelu_grad", GeluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GenTensorBufferOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "shape_list") {
      return CastAttr(&shape_list);
    } else if (attr_name == "value_list") {
      return CastAttr(&value_list);
    } else if (attr_name == "data_type") {
      return CastAttr(&data_type);
    } else if (attr_name == "dynamic_out") {
      return CastAttr(&dynamic_out);
    } else {
      return Error::RuntimeError() << "GenTensorBuffer op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"shape", "shape_list", "value_list", "data_type",
                                           "dynamic_out"};
    return attr_names;
  }

 public:
  Shape shape;
  std::vector<Shape> shape_list;
  std::vector<float> value_list;
  DataType data_type;
  bool dynamic_out;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.gen_tensor_buffer", GenTensorBufferOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GenerateRandomBatchPermutationIndicesOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "seed") {
      return CastAttr(&seed);
    } else {
      return Error::RuntimeError()
             << "GenerateRandomBatchPermutationIndices op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"seed"};
    return attr_names;
  }

 public:
  int64_t seed;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.generate_random_batch_permutation_indices",
                       GenerateRandomBatchPermutationIndicesOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GridSampleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "interpolation_mode") {
      return CastAttr(&interpolation_mode);
    } else if (attr_name == "padding_mode") {
      return CastAttr(&padding_mode);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else {
      return Error::RuntimeError() << "GridSample op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"interpolation_mode", "padding_mode", "align_corners"};
    return attr_names;
  }

 public:
  std::string interpolation_mode;
  std::string padding_mode;
  bool align_corners;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.grid_sample", GridSampleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class GridSampleGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "interpolation_mode") {
      return CastAttr(&interpolation_mode);
    } else if (attr_name == "padding_mode") {
      return CastAttr(&padding_mode);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else {
      return Error::RuntimeError() << "GridSampleGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"interpolation_mode", "padding_mode", "align_corners"};
    return attr_names;
  }

 public:
  std::string interpolation_mode;
  std::string padding_mode;
  bool align_corners;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.grid_sample_grad", GridSampleGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HardsigmoidOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Hardsigmoid op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hardsigmoid", HardsigmoidOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HardsigmoidGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "HardsigmoidGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hardsigmoid_grad", HardsigmoidGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HardswishOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Hardswish op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hardswish", HardswishOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HardswishGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "HardswishGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hardswish_grad", HardswishGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HardtanhOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "min_val") {
      return CastAttr(&min_val);
    } else if (attr_name == "max_val") {
      return CastAttr(&max_val);
    } else {
      return Error::RuntimeError() << "Hardtanh op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"min_val", "max_val"};
    return attr_names;
  }

 public:
  double min_val;
  double max_val;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hardtanh", HardtanhOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HardtanhGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "min_val") {
      return CastAttr(&min_val);
    } else if (attr_name == "max_val") {
      return CastAttr(&max_val);
    } else {
      return Error::RuntimeError() << "HardtanhGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"min_val", "max_val"};
    return attr_names;
  }

 public:
  double min_val;
  double max_val;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hardtanh_grad", HardtanhGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HierarchicalParallelCastOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else if (attr_name == "grad_mode") {
      return CastAttr(&grad_mode);
    } else if (attr_name == "grad_nd_sbp") {
      return CastAttr(&grad_nd_sbp);
    } else {
      return Error::RuntimeError()
             << "HierarchicalParallelCast op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"nd_sbp", "grad_mode", "grad_nd_sbp"};
    return attr_names;
  }

 public:
  std::vector<std::string> nd_sbp;
  std::string grad_mode;
  std::vector<std::string> grad_nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hierarchical_parallel_cast", HierarchicalParallelCastOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class HierarchicalParallelCastLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "HierarchicalParallelCastLike op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.hierarchical_parallel_cast_like",
                       HierarchicalParallelCastLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class IdentityOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Identity op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.identity", IdentityOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class IdentityBufferOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "buffer_size") {
      return CastAttr(&buffer_size);
    } else {
      return Error::RuntimeError() << "IdentityBuffer op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"buffer_size"};
    return attr_names;
  }

 public:
  int64_t buffer_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.identity_buffer", IdentityBufferOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageBatchAlignOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "data_type") {
      return CastAttr(&data_type);
    } else if (attr_name == "alignment") {
      return CastAttr(&alignment);
    } else if (attr_name == "dynamic_out") {
      return CastAttr(&dynamic_out);
    } else {
      return Error::RuntimeError() << "ImageBatchAlign op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"shape", "data_type", "alignment", "dynamic_out"};
    return attr_names;
  }

 public:
  Shape shape;
  DataType data_type;
  int32_t alignment;
  bool dynamic_out;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_batch_align", ImageBatchAlignOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageDecodeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "color_space") {
      return CastAttr(&color_space);
    } else if (attr_name == "data_type") {
      return CastAttr(&data_type);
    } else {
      return Error::RuntimeError() << "ImageDecode op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"color_space", "data_type"};
    return attr_names;
  }

 public:
  std::string color_space;
  DataType data_type;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_decode", ImageDecodeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageFlipOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ImageFlip op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_flip", ImageFlipOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageNormalizeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "std") {
      return CastAttr(&std);
    } else if (attr_name == "mean") {
      return CastAttr(&mean);
    } else {
      return Error::RuntimeError() << "ImageNormalize op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"std", "mean"};
    return attr_names;
  }

 public:
  std::vector<float> std;
  std::vector<float> mean;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_normalize", ImageNormalizeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageRandomCropOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "num_attempts") {
      return CastAttr(&num_attempts);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "has_seed") {
      return CastAttr(&has_seed);
    } else if (attr_name == "random_area") {
      return CastAttr(&random_area);
    } else if (attr_name == "random_aspect_ratio") {
      return CastAttr(&random_aspect_ratio);
    } else {
      return Error::RuntimeError() << "ImageRandomCrop op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"num_attempts", "seed", "has_seed", "random_area",
                                           "random_aspect_ratio"};
    return attr_names;
  }

 public:
  int32_t num_attempts;
  int64_t seed;
  bool has_seed;
  std::vector<float> random_area;
  std::vector<float> random_aspect_ratio;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_random_crop", ImageRandomCropOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageResizeKeepAspectRatioOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "target_size") {
      return CastAttr(&target_size);
    } else if (attr_name == "min_size") {
      return CastAttr(&min_size);
    } else if (attr_name == "max_size") {
      return CastAttr(&max_size);
    } else if (attr_name == "resize_longer") {
      return CastAttr(&resize_longer);
    } else if (attr_name == "interpolation_type") {
      return CastAttr(&interpolation_type);
    } else {
      return Error::RuntimeError()
             << "ImageResizeKeepAspectRatio op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"target_size", "min_size", "max_size", "resize_longer",
                                           "interpolation_type"};
    return attr_names;
  }

 public:
  int32_t target_size;
  int32_t min_size;
  int32_t max_size;
  bool resize_longer;
  std::string interpolation_type;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_resize_keep_aspect_ratio",
                       ImageResizeKeepAspectRatioOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageResizeToFixedOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "target_width") {
      return CastAttr(&target_width);
    } else if (attr_name == "target_height") {
      return CastAttr(&target_height);
    } else if (attr_name == "channels") {
      return CastAttr(&channels);
    } else if (attr_name == "data_type") {
      return CastAttr(&data_type);
    } else if (attr_name == "interpolation_type") {
      return CastAttr(&interpolation_type);
    } else {
      return Error::RuntimeError() << "ImageResizeToFixed op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"target_width", "target_height", "channels", "data_type",
                                           "interpolation_type"};
    return attr_names;
  }

 public:
  int64_t target_width;
  int64_t target_height;
  int64_t channels;
  DataType data_type;
  std::string interpolation_type;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_resize_to_fixed", ImageResizeToFixedOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ImageTargetResizeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "target_size") {
      return CastAttr(&target_size);
    } else if (attr_name == "max_size") {
      return CastAttr(&max_size);
    } else {
      return Error::RuntimeError() << "ImageTargetResize op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"target_size", "max_size"};
    return attr_names;
  }

 public:
  int32_t target_size;
  int32_t max_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.image_target_resize", ImageTargetResizeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class InTopKOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "k") {
      return CastAttr(&k);
    } else {
      return Error::RuntimeError() << "InTopK op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"k"};
    return attr_names;
  }

 public:
  int32_t k;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.in_top_k", InTopKOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class IndexedSlicesAdamUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "learning_rate_val") {
      return CastAttr(&learning_rate_val);
    } else if (attr_name == "beta1") {
      return CastAttr(&beta1);
    } else if (attr_name == "beta2") {
      return CastAttr(&beta2);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else if (attr_name == "amsgrad") {
      return CastAttr(&amsgrad);
    } else if (attr_name == "do_bias_correction") {
      return CastAttr(&do_bias_correction);
    } else {
      return Error::RuntimeError()
             << "IndexedSlicesAdamUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"learning_rate_val", "beta1",        "beta2",
                                           "epsilon",           "weight_decay", "amsgrad",
                                           "do_bias_correction"};
    return attr_names;
  }

 public:
  float learning_rate_val;
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
  bool amsgrad;
  bool do_bias_correction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.indexed_slices_adam_update", IndexedSlicesAdamUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class IndexedSlicesMomentumUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "beta") {
      return CastAttr(&beta);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError()
             << "IndexedSlicesMomentumUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"beta", "weight_decay"};
    return attr_names;
  }

 public:
  float beta;
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.indexed_slices_momentum_update",
                       IndexedSlicesMomentumUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class IndexedSlicesReduceSumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "IndexedSlicesReduceSum op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.indexed_slices_reduce_sum", IndexedSlicesReduceSumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class IndexedSlicesSgdUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError()
             << "IndexedSlicesSgdUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"weight_decay"};
    return attr_names;
  }

 public:
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.indexed_slices_sgd_update", IndexedSlicesSgdUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class KlDivLossOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else if (attr_name == "log_target") {
      return CastAttr(&log_target);
    } else {
      return Error::RuntimeError() << "KlDivLoss op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"reduction", "log_target"};
    return attr_names;
  }

 public:
  std::string reduction;
  bool log_target;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.kl_div_loss", KlDivLossOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class KlDivLossGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else if (attr_name == "log_target") {
      return CastAttr(&log_target);
    } else {
      return Error::RuntimeError() << "KlDivLossGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"reduction", "log_target"};
    return attr_names;
  }

 public:
  std::string reduction;
  bool log_target;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.kl_div_loss_grad", KlDivLossGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class L1L2RegularizeGradientOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else {
      return Error::RuntimeError()
             << "L1L2RegularizeGradient op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"l1", "l2"};
    return attr_names;
  }

 public:
  float l1;
  float l2;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.l1_l2_regularize_gradient", L1L2RegularizeGradientOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class L2NormalizeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError() << "L2Normalize op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.l2_normalize", L2NormalizeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class L2NormalizeGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError() << "L2NormalizeGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.l2_normalize_grad", L2NormalizeGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LambUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "beta1") {
      return CastAttr(&beta1);
    } else if (attr_name == "beta2") {
      return CastAttr(&beta2);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError() << "LambUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"beta1", "beta2", "epsilon",     "scale",
                                           "l1",    "l2",    "weight_decay"};
    return attr_names;
  }

 public:
  float beta1;
  float beta2;
  float epsilon;
  double scale;
  float l1;
  float l2;
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.lamb_update", LambUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LarsUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "momentum_beta") {
      return CastAttr(&momentum_beta);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "lars_coefficient") {
      return CastAttr(&lars_coefficient);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError() << "LarsUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "scale", "l1", "l2", "momentum_beta", "epsilon", "lars_coefficient", "weight_decay"};
    return attr_names;
  }

 public:
  double scale;
  float l1;
  float l2;
  float momentum_beta;
  float epsilon;
  float lars_coefficient;
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.lars_update", LarsUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LayerNormOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "center") {
      return CastAttr(&center);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "begin_norm_axis") {
      return CastAttr(&begin_norm_axis);
    } else if (attr_name == "begin_params_axis") {
      return CastAttr(&begin_params_axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError() << "LayerNorm op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"center", "scale", "begin_norm_axis",
                                           "begin_params_axis", "epsilon"};
    return attr_names;
  }

 public:
  bool center;
  bool scale;
  int64_t begin_norm_axis;
  int64_t begin_params_axis;
  double epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.layer_norm", LayerNormOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LayerNormGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "begin_norm_axis") {
      return CastAttr(&begin_norm_axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError() << "LayerNormGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"begin_norm_axis", "epsilon"};
    return attr_names;
  }

 public:
  int64_t begin_norm_axis;
  double epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.layer_norm_grad", LayerNormGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LayerNormParamGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "begin_params_axis") {
      return CastAttr(&begin_params_axis);
    } else {
      return Error::RuntimeError() << "LayerNormParamGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"begin_params_axis"};
    return attr_names;
  }

 public:
  int64_t begin_params_axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.layer_norm_param_grad", LayerNormParamGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LeakyReluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "LeakyRelu op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  float alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.leaky_relu", LeakyReluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LeakyReluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "LeakyReluGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"alpha"};
    return attr_names;
  }

 public:
  float alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.leaky_relu_grad", LeakyReluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LgammaOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Lgamma op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.lgamma", LgammaOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LgammaGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LgammaGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.lgamma_grad", LgammaGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Log op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log", LogOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Log1pOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Log1p op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log1p", Log1pOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class Log1pGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Log1pGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log1p_grad", Log1pGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LogGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log_grad", LogGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogSigmoidOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LogSigmoid op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log_sigmoid", LogSigmoidOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogSigmoidGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LogSigmoidGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log_sigmoid_grad", LogSigmoidGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogSoftmaxOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LogSoftmax op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log_softmax", LogSoftmaxOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogSoftmaxGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LogSoftmaxGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.log_softmax_grad", LogSoftmaxGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogicalNotOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "LogicalNot op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.logical_not", LogicalNotOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogicalSliceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "stop") {
      return CastAttr(&stop);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "LogicalSlice op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"start", "stop", "step"};
    return attr_names;
  }

 public:
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.logical_slice", LogicalSliceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class LogicalSliceAssignOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "stop") {
      return CastAttr(&stop);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "LogicalSliceAssign op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"start", "stop", "step"};
    return attr_names;
  }

 public:
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.logical_slice_assign", LogicalSliceAssignOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaskedFillOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "MaskedFill op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.masked_fill", MaskedFillOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MatmulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "transpose_a") {
      return CastAttr(&transpose_a);
    } else if (attr_name == "transpose_b") {
      return CastAttr(&transpose_b);
    } else if (attr_name == "alpha") {
      return CastAttr(&alpha);
    } else {
      return Error::RuntimeError() << "Matmul op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"transpose_a", "transpose_b", "alpha"};
    return attr_names;
  }

 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.matmul", MatmulOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaxPool1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "dilation") {
      return CastAttr(&dilation);
    } else if (attr_name == "return_indices") {
      return CastAttr(&return_indices);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "MaxPool1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",  "data_format",    "kernel_size", "stride",
                                           "dilation", "return_indices", "ceil_mode"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.maxpool_1d", MaxPool1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaxPool1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "dilation") {
      return CastAttr(&dilation);
    } else if (attr_name == "return_indices") {
      return CastAttr(&return_indices);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "MaxPool1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",  "data_format",    "kernel_size", "stride",
                                           "dilation", "return_indices", "ceil_mode"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.maxpool_1d_grad", MaxPool1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaxPool2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "dilation") {
      return CastAttr(&dilation);
    } else if (attr_name == "return_indices") {
      return CastAttr(&return_indices);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "MaxPool2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",  "data_format",    "kernel_size", "stride",
                                           "dilation", "return_indices", "ceil_mode"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.maxpool_2d", MaxPool2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaxPool2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "dilation") {
      return CastAttr(&dilation);
    } else if (attr_name == "return_indices") {
      return CastAttr(&return_indices);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "MaxPool2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",  "data_format",    "kernel_size", "stride",
                                           "dilation", "return_indices", "ceil_mode"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.maxpool_2d_grad", MaxPool2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaxPool3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "dilation") {
      return CastAttr(&dilation);
    } else if (attr_name == "return_indices") {
      return CastAttr(&return_indices);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "MaxPool3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",  "data_format",    "kernel_size", "stride",
                                           "dilation", "return_indices", "ceil_mode"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.maxpool_3d", MaxPool3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MaxPool3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "stride") {
      return CastAttr(&stride);
    } else if (attr_name == "dilation") {
      return CastAttr(&dilation);
    } else if (attr_name == "return_indices") {
      return CastAttr(&return_indices);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "MaxPool3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",  "data_format",    "kernel_size", "stride",
                                           "dilation", "return_indices", "ceil_mode"};
    return attr_names;
  }

 public:
  std::vector<int32_t> padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.maxpool_3d_grad", MaxPool3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MegatronGptMmapDataLoaderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "data_file_prefix") {
      return CastAttr(&data_file_prefix);
    } else if (attr_name == "seq_length") {
      return CastAttr(&seq_length);
    } else if (attr_name == "label_length") {
      return CastAttr(&label_length);
    } else if (attr_name == "num_samples") {
      return CastAttr(&num_samples);
    } else if (attr_name == "batch_size") {
      return CastAttr(&batch_size);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "split_sizes") {
      return CastAttr(&split_sizes);
    } else if (attr_name == "split_index") {
      return CastAttr(&split_index);
    } else if (attr_name == "shuffle") {
      return CastAttr(&shuffle);
    } else if (attr_name == "random_seed") {
      return CastAttr(&random_seed);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError()
             << "MegatronGptMmapDataLoader op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "data_file_prefix", "seq_length",  "label_length", "num_samples", "batch_size", "dtype",
        "split_sizes",      "split_index", "shuffle",      "random_seed", "nd_sbp"};
    return attr_names;
  }

 public:
  std::string data_file_prefix;
  int64_t seq_length;
  int64_t label_length;
  int64_t num_samples;
  int64_t batch_size;
  DataType dtype;
  std::vector<int64_t> split_sizes;
  int64_t split_index;
  bool shuffle;
  int64_t random_seed;
  std::vector<std::string> nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.megatron_gpt_mmap_data_loader", MegatronGptMmapDataLoaderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MinMaxObserverOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "quantization_formula") {
      return CastAttr(&quantization_formula);
    } else if (attr_name == "quantization_bit") {
      return CastAttr(&quantization_bit);
    } else if (attr_name == "quantization_scheme") {
      return CastAttr(&quantization_scheme);
    } else if (attr_name == "per_layer_quantization") {
      return CastAttr(&per_layer_quantization);
    } else {
      return Error::RuntimeError() << "MinMaxObserver op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"quantization_formula", "quantization_bit",
                                           "quantization_scheme", "per_layer_quantization"};
    return attr_names;
  }

 public:
  std::string quantization_formula;
  int32_t quantization_bit;
  std::string quantization_scheme;
  bool per_layer_quantization;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.min_max_observer", MinMaxObserverOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MishOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Mish op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.mish", MishOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MishGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "MishGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.mish_grad", MishGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MomentumUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "learning_rate_val") {
      return CastAttr(&learning_rate_val);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "beta") {
      return CastAttr(&beta);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError() << "MomentumUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"learning_rate_val", "scale", "l1", "l2", "beta",
                                           "weight_decay"};
    return attr_names;
  }

 public:
  float learning_rate_val;
  double scale;
  float l1;
  float l2;
  float beta;
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.momentum_update", MomentumUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MovingAverageMinMaxObserverOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "training") {
      return CastAttr(&training);
    } else if (attr_name == "quantization_formula") {
      return CastAttr(&quantization_formula);
    } else if (attr_name == "stop_update_after_iters") {
      return CastAttr(&stop_update_after_iters);
    } else if (attr_name == "quantization_bit") {
      return CastAttr(&quantization_bit);
    } else if (attr_name == "quantization_scheme") {
      return CastAttr(&quantization_scheme);
    } else if (attr_name == "momentum") {
      return CastAttr(&momentum);
    } else {
      return Error::RuntimeError()
             << "MovingAverageMinMaxObserver op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "training",         "quantization_formula", "stop_update_after_iters",
        "quantization_bit", "quantization_scheme",  "momentum"};
    return attr_names;
  }

 public:
  bool training;
  std::string quantization_formula;
  int64_t stop_update_after_iters;
  int32_t quantization_bit;
  std::string quantization_scheme;
  float momentum;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.moving_average_min_max_observer",
                       MovingAverageMinMaxObserverOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MultiCountNotFiniteOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "MultiCountNotFinite op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.multi_count_not_finite", MultiCountNotFiniteOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MultiSquareSumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "MultiSquareSum op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.multi_square_sum", MultiSquareSumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class MultiplyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Multiply op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.multiply", MultiplyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NarrowOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "length") {
      return CastAttr(&length);
    } else {
      return Error::RuntimeError() << "Narrow op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim", "start", "length"};
    return attr_names;
  }

 public:
  int64_t dim;
  int64_t start;
  int64_t length;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.narrow", NarrowOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NarrowGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dim") {
      return CastAttr(&dim);
    } else if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "length") {
      return CastAttr(&length);
    } else {
      return Error::RuntimeError() << "NarrowGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dim", "start", "length"};
    return attr_names;
  }

 public:
  int64_t dim;
  int64_t start;
  int64_t length;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.narrow_grad", NarrowGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NegativeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Negative op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.negative", NegativeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NegativeGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "NegativeGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.negative_grad", NegativeGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NllOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "ignore_index") {
      return CastAttr(&ignore_index);
    } else if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else {
      return Error::RuntimeError() << "Nll op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"ignore_index", "reduction"};
    return attr_names;
  }

 public:
  int64_t ignore_index;
  std::string reduction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.nll", NllOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NllGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "ignore_index") {
      return CastAttr(&ignore_index);
    } else if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else {
      return Error::RuntimeError() << "NllGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"ignore_index", "reduction"};
    return attr_names;
  }

 public:
  int64_t ignore_index;
  std::string reduction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.nll_grad", NllGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NmsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "iou_threshold") {
      return CastAttr(&iou_threshold);
    } else if (attr_name == "keep_n") {
      return CastAttr(&keep_n);
    } else {
      return Error::RuntimeError() << "Nms op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"iou_threshold", "keep_n"};
    return attr_names;
  }

 public:
  float iou_threshold;
  int32_t keep_n;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.nms", NmsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NormalOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "mean") {
      return CastAttr(&mean);
    } else if (attr_name == "std") {
      return CastAttr(&std);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Normal op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"mean", "std", "seed", "dtype", "shape", "nd_sbp"};
    return attr_names;
  }

 public:
  double mean;
  double std;
  int64_t seed;
  DataType dtype;
  Shape shape;
  std::string nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.normal", NormalOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NormalizationOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "training") {
      return CastAttr(&training);
    } else if (attr_name == "momentum") {
      return CastAttr(&momentum);
    } else {
      return Error::RuntimeError() << "Normalization op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon", "training", "momentum"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
  bool training;
  float momentum;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.normalization", NormalizationOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NormalizationAddReluBaseOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "training") {
      return CastAttr(&training);
    } else if (attr_name == "momentum") {
      return CastAttr(&momentum);
    } else {
      return Error::RuntimeError()
             << "NormalizationAddReluBase op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon", "training", "momentum"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
  bool training;
  float momentum;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.normalization_add_relu", NormalizationAddReluBaseOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NormalizationAddReluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError()
             << "NormalizationAddReluGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.normalization_add_relu_grad", NormalizationAddReluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NormalizationGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else {
      return Error::RuntimeError() << "NormalizationGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "epsilon"};
    return attr_names;
  }

 public:
  int32_t axis;
  float epsilon;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.normalization_grad", NormalizationGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NvtxEndOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "mark_prefix") {
      return CastAttr(&mark_prefix);
    } else {
      return Error::RuntimeError() << "NvtxEnd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"mark_prefix"};
    return attr_names;
  }

 public:
  std::string mark_prefix;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.nvtx_end", NvtxEndOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class NvtxStartOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "mark_prefix") {
      return CastAttr(&mark_prefix);
    } else {
      return Error::RuntimeError() << "NvtxStart op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"mark_prefix"};
    return attr_names;
  }

 public:
  std::string mark_prefix;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.nvtx_start", NvtxStartOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ObjectBboxFlipOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ObjectBboxFlip op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.object_bbox_flip", ObjectBboxFlipOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ObjectBboxScaleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ObjectBboxScale op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.object_bbox_scale", ObjectBboxScaleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ObjectSegmentationPolygonFlipOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ObjectSegmentationPolygonFlip op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.object_segmentation_polygon_flip",
                       ObjectSegmentationPolygonFlipOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ObjectSegmentationPolygonScaleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ObjectSegmentationPolygonScale op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.object_segmentation_polygon_scale",
                       ObjectSegmentationPolygonScaleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ObjectSegmentationPolygonToMaskOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ObjectSegmentationPolygonToMask op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.object_segmentation_polygon_to_mask",
                       ObjectSegmentationPolygonToMaskOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OfrecordBytesDecoderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "name") {
      return CastAttr(&name);
    } else {
      return Error::RuntimeError()
             << "OfrecordBytesDecoder op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"name"};
    return attr_names;
  }

 public:
  std::string name;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ofrecord_bytes_decoder", OfrecordBytesDecoderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OfrecordImageClassificationReaderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "data_dir") {
      return CastAttr(&data_dir);
    } else if (attr_name == "data_part_num") {
      return CastAttr(&data_part_num);
    } else if (attr_name == "batch_size") {
      return CastAttr(&batch_size);
    } else if (attr_name == "part_name_prefix") {
      return CastAttr(&part_name_prefix);
    } else if (attr_name == "part_name_suffix_length") {
      return CastAttr(&part_name_suffix_length);
    } else if (attr_name == "random_shuffle") {
      return CastAttr(&random_shuffle);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "shuffle_buffer_size") {
      return CastAttr(&shuffle_buffer_size);
    } else if (attr_name == "shuffle_after_epoch") {
      return CastAttr(&shuffle_after_epoch);
    } else if (attr_name == "color_space") {
      return CastAttr(&color_space);
    } else if (attr_name == "image_feature_name") {
      return CastAttr(&image_feature_name);
    } else if (attr_name == "label_feature_name") {
      return CastAttr(&label_feature_name);
    } else if (attr_name == "decode_buffer_size_per_thread") {
      return CastAttr(&decode_buffer_size_per_thread);
    } else if (attr_name == "num_decode_threads_per_machine") {
      return CastAttr(&num_decode_threads_per_machine);
    } else {
      return Error::RuntimeError()
             << "OfrecordImageClassificationReader op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"data_dir",
                                           "data_part_num",
                                           "batch_size",
                                           "part_name_prefix",
                                           "part_name_suffix_length",
                                           "random_shuffle",
                                           "seed",
                                           "shuffle_buffer_size",
                                           "shuffle_after_epoch",
                                           "color_space",
                                           "image_feature_name",
                                           "label_feature_name",
                                           "decode_buffer_size_per_thread",
                                           "num_decode_threads_per_machine"};
    return attr_names;
  }

 public:
  std::string data_dir;
  int32_t data_part_num;
  int32_t batch_size;
  std::string part_name_prefix;
  int32_t part_name_suffix_length;
  bool random_shuffle;
  int64_t seed;
  int32_t shuffle_buffer_size;
  bool shuffle_after_epoch;
  std::string color_space;
  std::string image_feature_name;
  std::string label_feature_name;
  int32_t decode_buffer_size_per_thread;
  int32_t num_decode_threads_per_machine;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ofrecord_image_classification_reader",
                       OfrecordImageClassificationReaderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OfrecordImageDecoderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "name") {
      return CastAttr(&name);
    } else if (attr_name == "color_space") {
      return CastAttr(&color_space);
    } else {
      return Error::RuntimeError()
             << "OfrecordImageDecoder op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"name", "color_space"};
    return attr_names;
  }

 public:
  std::string name;
  std::string color_space;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ofrecord_image_decoder", OfrecordImageDecoderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OfrecordImageDecoderRandomCropOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "name") {
      return CastAttr(&name);
    } else if (attr_name == "color_space") {
      return CastAttr(&color_space);
    } else if (attr_name == "num_attempts") {
      return CastAttr(&num_attempts);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "has_seed") {
      return CastAttr(&has_seed);
    } else if (attr_name == "random_area") {
      return CastAttr(&random_area);
    } else if (attr_name == "random_aspect_ratio") {
      return CastAttr(&random_aspect_ratio);
    } else {
      return Error::RuntimeError()
             << "OfrecordImageDecoderRandomCrop op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"name",     "color_space", "num_attempts",       "seed",
                                           "has_seed", "random_area", "random_aspect_ratio"};
    return attr_names;
  }

 public:
  std::string name;
  std::string color_space;
  int32_t num_attempts;
  int64_t seed;
  bool has_seed;
  std::vector<float> random_area;
  std::vector<float> random_aspect_ratio;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ofrecord_image_decoder_random_crop",
                       OfrecordImageDecoderRandomCropOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OfrecordRawDecoderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "name") {
      return CastAttr(&name);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "data_type") {
      return CastAttr(&data_type);
    } else if (attr_name == "dim1_varying_length") {
      return CastAttr(&dim1_varying_length);
    } else if (attr_name == "truncate") {
      return CastAttr(&truncate);
    } else {
      return Error::RuntimeError() << "OfrecordRawDecoder op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"name", "shape", "data_type", "dim1_varying_length",
                                           "truncate"};
    return attr_names;
  }

 public:
  std::string name;
  Shape shape;
  DataType data_type;
  bool dim1_varying_length;
  bool truncate;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ofrecord_raw_decoder", OfrecordRawDecoderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OneHotOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else if (attr_name == "floating_on_value") {
      return CastAttr(&floating_on_value);
    } else if (attr_name == "integer_on_value") {
      return CastAttr(&integer_on_value);
    } else if (attr_name == "floating_off_value") {
      return CastAttr(&floating_off_value);
    } else if (attr_name == "integer_off_value") {
      return CastAttr(&integer_off_value);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else {
      return Error::RuntimeError() << "OneHot op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "depth", "floating_on_value", "integer_on_value", "floating_off_value", "integer_off_value",
        "dtype"};
    return attr_names;
  }

 public:
  int64_t depth;
  double floating_on_value;
  int64_t integer_on_value;
  double floating_off_value;
  int64_t integer_off_value;
  DataType dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.one_hot", OneHotOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OnerecDecoderOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "key") {
      return CastAttr(&key);
    } else if (attr_name == "data_type") {
      return CastAttr(&data_type);
    } else if (attr_name == "static_shape") {
      return CastAttr(&static_shape);
    } else if (attr_name == "is_dynamic") {
      return CastAttr(&is_dynamic);
    } else if (attr_name == "has_reshape") {
      return CastAttr(&has_reshape);
    } else if (attr_name == "reshape") {
      return CastAttr(&reshape);
    } else if (attr_name == "has_batch_padding") {
      return CastAttr(&has_batch_padding);
    } else if (attr_name == "batch_padding") {
      return CastAttr(&batch_padding);
    } else {
      return Error::RuntimeError() << "OnerecDecoder op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "key",         "data_type", "static_shape",      "is_dynamic",
        "has_reshape", "reshape",   "has_batch_padding", "batch_padding"};
    return attr_names;
  }

 public:
  std::string key;
  DataType data_type;
  Shape static_shape;
  bool is_dynamic;
  bool has_reshape;
  Shape reshape;
  bool has_batch_padding;
  Shape batch_padding;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.onerec_decoder", OnerecDecoderOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class OnesLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "OnesLike op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ones_like", OnesLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PackOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "pack_num") {
      return CastAttr(&pack_num);
    } else {
      return Error::RuntimeError() << "Pack op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"pack_num"};
    return attr_names;
  }

 public:
  int32_t pack_num;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.pack", PackOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PadOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "floating_constant_value") {
      return CastAttr(&floating_constant_value);
    } else if (attr_name == "integral_constant_value") {
      return CastAttr(&integral_constant_value);
    } else {
      return Error::RuntimeError() << "Pad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding_before", "padding_after",
                                           "floating_constant_value", "integral_constant_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding_before;
  std::vector<int64_t> padding_after;
  double floating_constant_value;
  int64_t integral_constant_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.pad", PadOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PadGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "floating_constant_value") {
      return CastAttr(&floating_constant_value);
    } else if (attr_name == "integral_constant_value") {
      return CastAttr(&integral_constant_value);
    } else {
      return Error::RuntimeError() << "PadGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding_before", "padding_after",
                                           "floating_constant_value", "integral_constant_value"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding_before;
  std::vector<int64_t> padding_after;
  double floating_constant_value;
  int64_t integral_constant_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.pad_grad", PadGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ParallelCastOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "sbp_parallel") {
      return CastAttr(&sbp_parallel);
    } else if (attr_name == "grad_sbp_parallel") {
      return CastAttr(&grad_sbp_parallel);
    } else {
      return Error::RuntimeError() << "ParallelCast op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"sbp_parallel", "grad_sbp_parallel"};
    return attr_names;
  }

 public:
  std::string sbp_parallel;
  std::string grad_sbp_parallel;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.parallel_cast", ParallelCastOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PowOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Pow op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.pow", PowOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PowXGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "PowXGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.pow_x_grad", PowXGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PowYGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "PowYGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.pow_y_grad", PowYGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PreluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Prelu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.prelu", PreluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class PreluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "PreluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.prelu_grad", PreluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class QuantizationOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "quantization_formula") {
      return CastAttr(&quantization_formula);
    } else if (attr_name == "quantization_bit") {
      return CastAttr(&quantization_bit);
    } else if (attr_name == "quantization_scheme") {
      return CastAttr(&quantization_scheme);
    } else {
      return Error::RuntimeError() << "Quantization op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"quantization_formula", "quantization_bit",
                                           "quantization_scheme"};
    return attr_names;
  }

 public:
  std::string quantization_formula;
  int32_t quantization_bit;
  std::string quantization_scheme;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.quantization", QuantizationOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RandomMaskLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "rate") {
      return CastAttr(&rate);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else {
      return Error::RuntimeError() << "RandomMaskLike op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"rate", "seed"};
    return attr_names;
  }

 public:
  float rate;
  int64_t seed;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.random_mask_like", RandomMaskLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RandpermOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "n") {
      return CastAttr(&n);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Randperm op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"n", "seed", "nd_sbp"};
    return attr_names;
  }

 public:
  int32_t n;
  int64_t seed;
  std::string nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.randperm", RandpermOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReciprocalOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Reciprocal op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reciprocal", ReciprocalOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReciprocalGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ReciprocalGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reciprocal_grad", ReciprocalGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReciprocalNoNanOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ReciprocalNoNan op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reciprocal_no_nan", ReciprocalNoNanOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReciprocalNoNanGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ReciprocalNoNanGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reciprocal_no_nan_grad", ReciprocalNoNanGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RecvOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "src_process_id") {
      return CastAttr(&src_process_id);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "device_type") {
      return CastAttr(&device_type);
    } else if (attr_name == "device_id") {
      return CastAttr(&device_id);
    } else {
      return Error::RuntimeError() << "Recv op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"src_process_id", "dtype", "shape", "device_type",
                                           "device_id"};
    return attr_names;
  }

 public:
  int64_t src_process_id;
  DataType dtype;
  Shape shape;
  std::string device_type;
  int64_t device_id;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.recv", RecvOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceAllOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError() << "ReduceAll op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_all", ReduceAllOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceAnyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError() << "ReduceAny op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_any", ReduceAnyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMaxOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError() << "ReduceMax op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_max", ReduceMaxOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMaxDeviceStageOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError()
             << "ReduceMaxDeviceStage op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_max_device_stage", ReduceMaxDeviceStageOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMaxDeviceStageGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError()
             << "ReduceMaxDeviceStageGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_max_device_stage_grad", ReduceMaxDeviceStageGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMaxGlobalStageOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError()
             << "ReduceMaxGlobalStage op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_max_global_stage", ReduceMaxGlobalStageOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMaxGlobalStageGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError()
             << "ReduceMaxGlobalStageGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_max_global_stage_grad", ReduceMaxGlobalStageGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMinOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError() << "ReduceMin op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_min", ReduceMinOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMinDeviceStageOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError()
             << "ReduceMinDeviceStage op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_min_device_stage", ReduceMinDeviceStageOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMinDeviceStageGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError()
             << "ReduceMinDeviceStageGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_min_device_stage_grad", ReduceMinDeviceStageGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMinGlobalStageOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError()
             << "ReduceMinGlobalStage op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_min_global_stage", ReduceMinGlobalStageOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceMinGlobalStageGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError()
             << "ReduceMinGlobalStageGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_min_global_stage_grad", ReduceMinGlobalStageGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceProdOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError() << "ReduceProd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_prod", ReduceProdOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceSumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "keepdims") {
      return CastAttr(&keepdims);
    } else {
      return Error::RuntimeError() << "ReduceSum op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "keepdims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_sum", ReduceSumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReduceSumLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError() << "ReduceSumLike op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reduce_sum_like", ReduceSumLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReflectionPad2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else {
      return Error::RuntimeError() << "ReflectionPad2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reflection_pad2d", ReflectionPad2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReflectionPad2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else {
      return Error::RuntimeError() << "ReflectionPad2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reflection_pad2d_grad", ReflectionPad2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Relu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.relu", ReluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ReluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.relu_grad", ReluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RepeatOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "repeat_num") {
      return CastAttr(&repeat_num);
    } else {
      return Error::RuntimeError() << "Repeat op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"repeat_num"};
    return attr_names;
  }

 public:
  int32_t repeat_num;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.repeat", RepeatOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReplicationPad2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else {
      return Error::RuntimeError() << "ReplicationPad2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.replication_pad2d", ReplicationPad2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReplicationPad2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else {
      return Error::RuntimeError()
             << "ReplicationPad2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding"};
    return attr_names;
  }

 public:
  std::vector<int64_t> padding;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.replication_pad2d_grad", ReplicationPad2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReshapeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "Reshape op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"shape"};
    return attr_names;
  }

 public:
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reshape", ReshapeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ReshapeLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ReshapeLike op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.reshape_like", ReshapeLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RintOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Rint op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.rint", RintOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RintGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "RintGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.rint_grad", RintGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RmspropUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "learning_rate_val") {
      return CastAttr(&learning_rate_val);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "centered") {
      return CastAttr(&centered);
    } else if (attr_name == "epsilon") {
      return CastAttr(&epsilon);
    } else if (attr_name == "decay_rate") {
      return CastAttr(&decay_rate);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError() << "RmspropUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "learning_rate_val", "scale",   "l1",         "l2",
        "centered",          "epsilon", "decay_rate", "weight_decay"};
    return attr_names;
  }

 public:
  float learning_rate_val;
  double scale;
  float l1;
  float l2;
  bool centered;
  float epsilon;
  float decay_rate;
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.rmsprop_update", RmspropUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RollOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "shifts") {
      return CastAttr(&shifts);
    } else if (attr_name == "dims") {
      return CastAttr(&dims);
    } else {
      return Error::RuntimeError() << "Roll op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"shifts", "dims"};
    return attr_names;
  }

 public:
  std::vector<int32_t> shifts;
  std::vector<int32_t> dims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.roll", RollOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RoundOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Round op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.round", RoundOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RoundGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "RoundGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.round_grad", RoundGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RsqrtOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Rsqrt op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.rsqrt", RsqrtOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class RsqrtGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "RsqrtGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.rsqrt_grad", RsqrtGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SamePaddingOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else {
      return Error::RuntimeError() << "SamePadding op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "data_format", "kernel_size", "strides",
                                           "dilation_rate"};
    return attr_names;
  }

 public:
  std::string padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.same_padding", SamePaddingOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SamePaddingGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else {
      return Error::RuntimeError() << "SamePaddingGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding", "data_format", "kernel_size", "strides",
                                           "dilation_rate"};
    return attr_names;
  }

 public:
  std::string padding;
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.same_padding_grad", SamePaddingGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarAddOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarAdd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_add", ScalarAddOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarAddByTensorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ScalarAddByTensor op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_add_by_tensor", ScalarAddByTensorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarDivByTensorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ScalarDivByTensor op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_div_by_tensor", ScalarDivByTensorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarFloordivOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarFloordiv op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_floordiv", ScalarFloordivOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarFmodOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarFmod op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_fmod", ScalarFmodOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalAndOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarLogicalAnd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_and", ScalarLogicalAndOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarLogicalEqual op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_equal", ScalarLogicalEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalGreaterOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError()
             << "ScalarLogicalGreater op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_greater", ScalarLogicalGreaterOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalGreaterEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError()
             << "ScalarLogicalGreaterEqual op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_greater_equal", ScalarLogicalGreaterEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalLessOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarLogicalLess op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_less", ScalarLogicalLessOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalLessEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError()
             << "ScalarLogicalLessEqual op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_less_equal", ScalarLogicalLessEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalNotEqualOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError()
             << "ScalarLogicalNotEqual op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_not_equal", ScalarLogicalNotEqualOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalOrOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarLogicalOr op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_or", ScalarLogicalOrOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarLogicalXorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarLogicalXor op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_logical_xor", ScalarLogicalXorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarMulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarMul op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_mul", ScalarMulOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarMulByTensorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ScalarMulByTensor op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_mul_by_tensor", ScalarMulByTensorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarPowOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarPow op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_pow", ScalarPowOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarPowGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "ScalarPowGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_pow_grad", ScalarPowGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScalarSubByTensorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ScalarSubByTensor op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scalar_sub_by_tensor", ScalarSubByTensorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScatterNdOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "shape") {
      return CastAttr(&shape);
    } else {
      return Error::RuntimeError() << "ScatterNd op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"shape"};
    return attr_names;
  }

 public:
  Shape shape;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scatter_nd", ScatterNdOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ScatterNdLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ScatterNdLike op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.scatter_nd_like", ScatterNdLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SeluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Selu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.selu", SeluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SeluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SeluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.selu_grad", SeluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SendOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dst_process_id") {
      return CastAttr(&dst_process_id);
    } else {
      return Error::RuntimeError() << "Send op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dst_process_id"};
    return attr_names;
  }

 public:
  int64_t dst_process_id;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.send", SendOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SgdUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "learning_rate_val") {
      return CastAttr(&learning_rate_val);
    } else if (attr_name == "scale") {
      return CastAttr(&scale);
    } else if (attr_name == "l1") {
      return CastAttr(&l1);
    } else if (attr_name == "l2") {
      return CastAttr(&l2);
    } else if (attr_name == "weight_decay") {
      return CastAttr(&weight_decay);
    } else {
      return Error::RuntimeError() << "SgdUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"learning_rate_val", "scale", "l1", "l2",
                                           "weight_decay"};
    return attr_names;
  }

 public:
  float learning_rate_val;
  double scale;
  float l1;
  float l2;
  float weight_decay;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sgd_update", SgdUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SigmoidOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Sigmoid op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sigmoid", SigmoidOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SigmoidCrossEntropyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SigmoidCrossEntropy op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sigmoid_cross_entropy", SigmoidCrossEntropyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SigmoidCrossEntropyGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SigmoidCrossEntropyGrad op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sigmoid_cross_entropy_grad", SigmoidCrossEntropyGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SigmoidGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SigmoidGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sigmoid_grad", SigmoidGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SigmoidV2OpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SigmoidV2 op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sigmoid_v2", SigmoidV2OpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SigmoidV2GradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SigmoidV2Grad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sigmoid_v2_grad", SigmoidV2GradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SignOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Sign op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sign", SignOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SignGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SignGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sign_grad", SignGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SiluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Silu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.silu", SiluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SiluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SiluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.silu_grad", SiluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SinOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Sin op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sin", SinOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SinGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SinGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sin_grad", SinGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SinhOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Sinh op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sinh", SinhOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SinhGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SinhGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sinh_grad", SinhGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SliceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "stop") {
      return CastAttr(&stop);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "Slice op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"start", "stop", "step"};
    return attr_names;
  }

 public:
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.slice", SliceOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SliceGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "stop") {
      return CastAttr(&stop);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "SliceGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"start", "stop", "step"};
    return attr_names;
  }

 public:
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.slice_grad", SliceGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SliceUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "start") {
      return CastAttr(&start);
    } else if (attr_name == "stop") {
      return CastAttr(&stop);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "SliceUpdate op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"start", "stop", "step"};
    return attr_names;
  }

 public:
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.slice_update", SliceUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SmoothL1LossOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else if (attr_name == "beta") {
      return CastAttr(&beta);
    } else {
      return Error::RuntimeError() << "SmoothL1Loss op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"reduction", "beta"};
    return attr_names;
  }

 public:
  std::string reduction;
  float beta;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.smooth_l1_loss", SmoothL1LossOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SmoothL1LossGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "reduction") {
      return CastAttr(&reduction);
    } else if (attr_name == "beta") {
      return CastAttr(&beta);
    } else {
      return Error::RuntimeError() << "SmoothL1LossGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"reduction", "beta"};
    return attr_names;
  }

 public:
  std::string reduction;
  float beta;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.smooth_l1_loss_grad", SmoothL1LossGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftmaxOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Softmax op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softmax", SoftmaxOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftmaxCrossEntropyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SoftmaxCrossEntropy op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softmax_cross_entropy", SoftmaxCrossEntropyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftmaxCrossEntropyGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SoftmaxCrossEntropyGrad op has no attribute named "
                                 << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softmax_cross_entropy_grad", SoftmaxCrossEntropyGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftmaxGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SoftmaxGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softmax_grad", SoftmaxGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftplusOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Softplus op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softplus", SoftplusOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftplusGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SoftplusGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softplus_grad", SoftplusGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftsignOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Softsign op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softsign", SoftsignOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SoftsignGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SoftsignGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.softsign_grad", SoftsignGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SortOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "direction") {
      return CastAttr(&direction);
    } else {
      return Error::RuntimeError() << "Sort op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"direction"};
    return attr_names;
  }

 public:
  std::string direction;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sort", SortOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseCrossEntropyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError() << "SparseCrossEntropy op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy", SparseCrossEntropyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseCrossEntropyGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseCrossEntropyGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy_grad", SparseCrossEntropyGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseCrossEntropyMsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseCrossEntropyMs op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy_ms", SparseCrossEntropyMsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseCrossEntropyMsGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseCrossEntropyMsGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy_ms_grad", SparseCrossEntropyMsGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseSoftmaxCrossEntropyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseSoftmaxCrossEntropy op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy", SparseSoftmaxCrossEntropyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseSoftmaxCrossEntropyGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseSoftmaxCrossEntropyGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy_grad",
                       SparseSoftmaxCrossEntropyGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseSoftmaxCrossEntropyMsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseSoftmaxCrossEntropyMs op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy_ms",
                       SparseSoftmaxCrossEntropyMsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SparseSoftmaxCrossEntropyMsGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth") {
      return CastAttr(&depth);
    } else {
      return Error::RuntimeError()
             << "SparseSoftmaxCrossEntropyMsGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth"};
    return attr_names;
  }

 public:
  int64_t depth;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy_ms_grad",
                       SparseSoftmaxCrossEntropyMsGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SplitLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError() << "SplitLike op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int64_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.split_like", SplitLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SqrtOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Sqrt op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sqrt", SqrtOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SqrtGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SqrtGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.sqrt_grad", SqrtGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SquareOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Square op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.square", SquareOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SquareGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SquareGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.square_grad", SquareGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SquareSumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SquareSum op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.square_sum", SquareSumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SqueezeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axes") {
      return CastAttr(&axes);
    } else {
      return Error::RuntimeError() << "Squeeze op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axes"};
    return attr_names;
  }

 public:
  std::vector<int32_t> axes;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.squeeze", SqueezeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SspVariableProxyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "buffer_size") {
      return CastAttr(&buffer_size);
    } else {
      return Error::RuntimeError() << "SspVariableProxy op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"buffer_size"};
    return attr_names;
  }

 public:
  int64_t buffer_size;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.ssp_variable_proxy", SspVariableProxyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SummaryWriteHistogramOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SummaryWriteHistogram op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.summary_write_histogram", SummaryWriteHistogramOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SummaryWriteImageOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SummaryWriteImage op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.summary_write_image", SummaryWriteImageOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SummaryWritePbOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SummaryWritePb op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.summary_write_pb", SummaryWritePbOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class SummaryWriteScalarOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "SummaryWriteScalar op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.summary_write_scalar", SummaryWriteScalarOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TanOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Tan op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tan", TanOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TanGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TanGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tan_grad", TanGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TanhOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Tanh op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tanh", TanhOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TanhGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TanhGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tanh_grad", TanhGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TensorBufferToListOfTensorsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_shape") {
      return CastAttr(&out_shape);
    } else if (attr_name == "out_dtype") {
      return CastAttr(&out_dtype);
    } else if (attr_name == "dynamic_out") {
      return CastAttr(&dynamic_out);
    } else {
      return Error::RuntimeError()
             << "TensorBufferToListOfTensors op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_shape", "out_dtype", "dynamic_out"};
    return attr_names;
  }

 public:
  Shape out_shape;
  DataType out_dtype;
  bool dynamic_out;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tensor_buffer_to_list_of_tensors",
                       TensorBufferToListOfTensorsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TensorBufferToListOfTensorsV2OpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_shapes") {
      return CastAttr(&out_shapes);
    } else if (attr_name == "out_dtypes") {
      return CastAttr(&out_dtypes);
    } else if (attr_name == "dynamic_out") {
      return CastAttr(&dynamic_out);
    } else {
      return Error::RuntimeError()
             << "TensorBufferToListOfTensorsV2 op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_shapes", "out_dtypes", "dynamic_out"};
    return attr_names;
  }

 public:
  std::vector<Shape> out_shapes;
  std::vector<DataType> out_dtypes;
  bool dynamic_out;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tensor_buffer_to_list_of_tensors_v2",
                       TensorBufferToListOfTensorsV2OpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TensorBufferToTensorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "instance_shape") {
      return CastAttr(&instance_shape);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else {
      return Error::RuntimeError()
             << "TensorBufferToTensor op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"instance_shape", "dtype"};
    return attr_names;
  }

 public:
  Shape instance_shape;
  DataType dtype;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tensor_buffer_to_tensor", TensorBufferToTensorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TensorScatterNdAddOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TensorScatterNdAdd op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tensor_scatter_nd_add", TensorScatterNdAddOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TensorScatterNdUpdateOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TensorScatterNdUpdate op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tensor_scatter_nd_update", TensorScatterNdUpdateOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TensorToTensorBufferOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "instance_dims") {
      return CastAttr(&instance_dims);
    } else {
      return Error::RuntimeError()
             << "TensorToTensorBuffer op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"instance_dims"};
    return attr_names;
  }

 public:
  int32_t instance_dims;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tensor_to_tensor_buffer", TensorToTensorBufferOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TestUserOpAttrAutoTypeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "int1") {
      return CastAttr(&int1);
    } else if (attr_name == "int2") {
      return CastAttr(&int2);
    } else {
      return Error::RuntimeError()
             << "TestUserOpAttrAutoType op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"int1", "int2"};
    return attr_names;
  }

 public:
  int32_t int1;
  int32_t int2;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.test_user_op_attr_auto_type", TestUserOpAttrAutoTypeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfAvgPool1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfAvgPool1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_avg_pool_1d", TfAvgPool1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfAvgPool1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfAvgPool1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_avg_pool_1d_grad", TfAvgPool1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfAvgPool2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfAvgPool2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_avg_pool_2d", TfAvgPool2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfAvgPool2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfAvgPool2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_avg_pool_2d_grad", TfAvgPool2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfAvgPool3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfAvgPool3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_avg_pool_3d", TfAvgPool3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfAvgPool3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfAvgPool3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_avg_pool_3d_grad", TfAvgPool3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfMaxPool1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfMaxPool1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_max_pool_1d", TfMaxPool1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfMaxPool1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfMaxPool1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_max_pool_1d_grad", TfMaxPool1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfMaxPool2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfMaxPool2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_max_pool_2d", TfMaxPool2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfMaxPool2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfMaxPool2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_max_pool_2d_grad", TfMaxPool2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfMaxPool3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfMaxPool3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_max_pool_3d", TfMaxPool3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfMaxPool3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "padding_before") {
      return CastAttr(&padding_before);
    } else if (attr_name == "padding_after") {
      return CastAttr(&padding_after);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "pool_size") {
      return CastAttr(&pool_size);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "ceil_mode") {
      return CastAttr(&ceil_mode);
    } else {
      return Error::RuntimeError() << "TfMaxPool3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"padding",     "padding_before", "padding_after",
                                           "data_format", "pool_size",      "strides",
                                           "ceil_mode"};
    return attr_names;
  }

 public:
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::string data_format;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_max_pool_3d_grad", TfMaxPool3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfPreluOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TfPrelu op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_prelu", TfPreluOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TfPreluGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TfPreluGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tf_prelu_grad", TfPreluGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TopKOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "k") {
      return CastAttr(&k);
    } else if (attr_name == "sorted") {
      return CastAttr(&sorted);
    } else {
      return Error::RuntimeError() << "TopK op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"k", "sorted"};
    return attr_names;
  }

 public:
  int32_t k;
  bool sorted;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.top_k", TopKOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TransposeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "perm") {
      return CastAttr(&perm);
    } else {
      return Error::RuntimeError() << "Transpose op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"perm"};
    return attr_names;
  }

 public:
  std::vector<int32_t> perm;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.transpose", TransposeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TrilOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else if (attr_name == "floating_fill_value") {
      return CastAttr(&floating_fill_value);
    } else if (attr_name == "integer_fill_value") {
      return CastAttr(&integer_fill_value);
    } else if (attr_name == "is_floating_fill_value") {
      return CastAttr(&is_floating_fill_value);
    } else {
      return Error::RuntimeError() << "Tril op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal", "floating_fill_value", "integer_fill_value",
                                           "is_floating_fill_value"};
    return attr_names;
  }

 public:
  int64_t diagonal;
  double floating_fill_value;
  int64_t integer_fill_value;
  bool is_floating_fill_value;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tril", TrilOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TriuOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "diagonal") {
      return CastAttr(&diagonal);
    } else {
      return Error::RuntimeError() << "Triu op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"diagonal"};
    return attr_names;
  }

 public:
  int64_t diagonal;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.triu", TriuOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class TupleIdentityOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "TupleIdentity op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.tuple_identity", TupleIdentityOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnfoldOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "kernel_size") {
      return CastAttr(&kernel_size);
    } else if (attr_name == "padding") {
      return CastAttr(&padding);
    } else if (attr_name == "strides") {
      return CastAttr(&strides);
    } else if (attr_name == "dilation_rate") {
      return CastAttr(&dilation_rate);
    } else {
      return Error::RuntimeError() << "Unfold op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"data_format", "kernel_size", "padding", "strides",
                                           "dilation_rate"};
    return attr_names;
  }

 public:
  std::string data_format;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> padding;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unfold", UnfoldOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnfoldTensorOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dimension") {
      return CastAttr(&dimension);
    } else if (attr_name == "size") {
      return CastAttr(&size);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "UnfoldTensor op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dimension", "size", "step"};
    return attr_names;
  }

 public:
  int32_t dimension;
  int32_t size;
  int32_t step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unfold_tensor", UnfoldTensorOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnfoldTensorGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "dimension") {
      return CastAttr(&dimension);
    } else if (attr_name == "size") {
      return CastAttr(&size);
    } else if (attr_name == "step") {
      return CastAttr(&step);
    } else {
      return Error::RuntimeError() << "UnfoldTensorGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"dimension", "size", "step"};
    return attr_names;
  }

 public:
  int32_t dimension;
  int32_t size;
  int32_t step;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unfold_tensor_grad", UnfoldTensorGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UniformOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "from") {
      return CastAttr(&from);
    } else if (attr_name == "to") {
      return CastAttr(&to);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "Uniform op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"from", "to", "seed", "dtype", "shape", "nd_sbp"};
    return attr_names;
  }

 public:
  double from;
  double to;
  int64_t seed;
  DataType dtype;
  Shape shape;
  std::string nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.uniform", UniformOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UniformIntOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "from") {
      return CastAttr(&from);
    } else if (attr_name == "to") {
      return CastAttr(&to);
    } else if (attr_name == "seed") {
      return CastAttr(&seed);
    } else if (attr_name == "dtype") {
      return CastAttr(&dtype);
    } else if (attr_name == "shape") {
      return CastAttr(&shape);
    } else if (attr_name == "nd_sbp") {
      return CastAttr(&nd_sbp);
    } else {
      return Error::RuntimeError() << "UniformInt op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"from", "to", "seed", "dtype", "shape", "nd_sbp"};
    return attr_names;
  }

 public:
  int64_t from;
  int64_t to;
  int64_t seed;
  DataType dtype;
  Shape shape;
  std::string nd_sbp;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.uniform_int", UniformIntOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UniqueWithCountsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "out_idx") {
      return CastAttr(&out_idx);
    } else {
      return Error::RuntimeError() << "UniqueWithCounts op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"out_idx"};
    return attr_names;
  }

 public:
  DataType out_idx;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unique_with_counts", UniqueWithCountsOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnpackOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "unpack_num") {
      return CastAttr(&unpack_num);
    } else {
      return Error::RuntimeError() << "Unpack op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"unpack_num"};
    return attr_names;
  }

 public:
  int32_t unpack_num;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unpack", UnpackOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnsortedBatchSegmentSumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "num_segments") {
      return CastAttr(&num_segments);
    } else {
      return Error::RuntimeError()
             << "UnsortedBatchSegmentSum op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"num_segments"};
    return attr_names;
  }

 public:
  int64_t num_segments;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unsorted_batch_segment_sum", UnsortedBatchSegmentSumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnsortedSegmentSumOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else if (attr_name == "num_segments") {
      return CastAttr(&num_segments);
    } else {
      return Error::RuntimeError() << "UnsortedSegmentSum op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis", "num_segments"};
    return attr_names;
  }

 public:
  int64_t axis;
  int64_t num_segments;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unsorted_segment_sum", UnsortedSegmentSumOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UnsortedSegmentSumLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "axis") {
      return CastAttr(&axis);
    } else {
      return Error::RuntimeError()
             << "UnsortedSegmentSumLike op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"axis"};
    return attr_names;
  }

 public:
  int64_t axis;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.unsorted_segment_sum_like", UnsortedSegmentSumLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "interpolation") {
      return CastAttr(&interpolation);
    } else {
      return Error::RuntimeError() << "Upsample op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "align_corners",
                                           "data_format", "interpolation"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
  std::string interpolation;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample", UpsampleOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleBicubic2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleBicubic2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "align_corners",
                                           "data_format"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_bicubic_2d", UpsampleBicubic2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleBicubic2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleBicubic2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "align_corners",
                                           "data_format"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_bicubic_2d_grad", UpsampleBicubic2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleBilinear2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleBilinear2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "align_corners",
                                           "data_format"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_bilinear_2d", UpsampleBilinear2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleBilinear2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleBilinear2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "align_corners",
                                           "data_format"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_bilinear_2d_grad", UpsampleBilinear2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else if (attr_name == "interpolation") {
      return CastAttr(&interpolation);
    } else {
      return Error::RuntimeError() << "UpsampleGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "align_corners",
                                           "data_format", "interpolation"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
  std::string interpolation;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_grad", UpsampleGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleLinear1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_factor") {
      return CastAttr(&scale_factor);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleLinear1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_factor", "align_corners", "data_format"};
    return attr_names;
  }

 public:
  float scale_factor;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_linear_1d", UpsampleLinear1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleLinear1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_factor") {
      return CastAttr(&scale_factor);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleLinear1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_factor", "align_corners", "data_format"};
    return attr_names;
  }

 public:
  float scale_factor;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_linear_1d_grad", UpsampleLinear1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleNearest1DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_factor") {
      return CastAttr(&scale_factor);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleNearest1D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_factor", "data_format"};
    return attr_names;
  }

 public:
  float scale_factor;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_nearest_1d", UpsampleNearest1DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleNearest1DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "scale_factor") {
      return CastAttr(&scale_factor);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleNearest1DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"scale_factor", "data_format"};
    return attr_names;
  }

 public:
  float scale_factor;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_nearest_1d_grad", UpsampleNearest1DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleNearest2DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleNearest2D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "data_format"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_nearest_2d", UpsampleNearest2DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleNearest2DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleNearest2DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"height_scale", "width_scale", "data_format"};
    return attr_names;
  }

 public:
  float height_scale;
  float width_scale;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_nearest_2d_grad", UpsampleNearest2DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleNearest3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth_scale") {
      return CastAttr(&depth_scale);
    } else if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleNearest3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth_scale", "height_scale", "width_scale",
                                           "data_format"};
    return attr_names;
  }

 public:
  float depth_scale;
  float height_scale;
  float width_scale;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_nearest_3d", UpsampleNearest3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleNearest3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth_scale") {
      return CastAttr(&depth_scale);
    } else if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleNearest3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth_scale", "height_scale", "width_scale",
                                           "data_format"};
    return attr_names;
  }

 public:
  float depth_scale;
  float height_scale;
  float width_scale;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_nearest_3d_grad", UpsampleNearest3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleTrilinear3DOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth_scale") {
      return CastAttr(&depth_scale);
    } else if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError() << "UpsampleTrilinear3D op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth_scale", "height_scale", "width_scale",
                                           "align_corners", "data_format"};
    return attr_names;
  }

 public:
  float depth_scale;
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_trilinear_3d", UpsampleTrilinear3DOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class UpsampleTrilinear3DGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "depth_scale") {
      return CastAttr(&depth_scale);
    } else if (attr_name == "height_scale") {
      return CastAttr(&height_scale);
    } else if (attr_name == "width_scale") {
      return CastAttr(&width_scale);
    } else if (attr_name == "align_corners") {
      return CastAttr(&align_corners);
    } else if (attr_name == "data_format") {
      return CastAttr(&data_format);
    } else {
      return Error::RuntimeError()
             << "UpsampleTrilinear3DGrad op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"depth_scale", "height_scale", "width_scale",
                                           "align_corners", "data_format"};
    return attr_names;
  }

 public:
  float depth_scale;
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.upsample_trilinear_3d_grad", UpsampleTrilinear3DGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class WhereOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Where op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.where", WhereOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class WhereScalarXOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "WhereScalarX op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.where_scalar_x", WhereScalarXOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class WhereScalarXyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_x_int_operand") {
      return CastAttr(&has_x_int_operand);
    } else if (attr_name == "has_x_float_operand") {
      return CastAttr(&has_x_float_operand);
    } else if (attr_name == "has_y_int_operand") {
      return CastAttr(&has_y_int_operand);
    } else if (attr_name == "has_y_float_operand") {
      return CastAttr(&has_y_float_operand);
    } else if (attr_name == "x_int_operand") {
      return CastAttr(&x_int_operand);
    } else if (attr_name == "x_float_operand") {
      return CastAttr(&x_float_operand);
    } else if (attr_name == "y_int_operand") {
      return CastAttr(&y_int_operand);
    } else if (attr_name == "y_float_operand") {
      return CastAttr(&y_float_operand);
    } else {
      return Error::RuntimeError() << "WhereScalarXy op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{
        "has_x_int_operand", "has_x_float_operand", "has_y_int_operand", "has_y_float_operand",
        "x_int_operand",     "x_float_operand",     "y_int_operand",     "y_float_operand"};
    return attr_names;
  }

 public:
  bool has_x_int_operand;
  bool has_x_float_operand;
  bool has_y_int_operand;
  bool has_y_float_operand;
  int64_t x_int_operand;
  double x_float_operand;
  int64_t y_int_operand;
  double y_float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.where_scalar_xy", WhereScalarXyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class WhereScalarYOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    if (attr_name == "has_int_operand") {
      return CastAttr(&has_int_operand);
    } else if (attr_name == "has_float_operand") {
      return CastAttr(&has_float_operand);
    } else if (attr_name == "int_operand") {
      return CastAttr(&int_operand);
    } else if (attr_name == "float_operand") {
      return CastAttr(&float_operand);
    } else {
      return Error::RuntimeError() << "WhereScalarY op has no attribute named " << attr_name;
    }
  }
  const HashSet<std::string>& AttrNamesSet() const override {
    static HashSet<std::string> attr_names{"has_int_operand", "has_float_operand", "int_operand",
                                           "float_operand"};
    return attr_names;
  }

 public:
  bool has_int_operand;
  bool has_float_operand;
  int64_t int_operand;
  double float_operand;
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.where_scalar_y", WhereScalarYOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class XdivyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Xdivy op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.xdivy", XdivyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class XdivyXGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "XdivyXGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.xdivy_x_grad", XdivyXGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class XdivyYGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "XdivyYGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.xdivy_y_grad", XdivyYGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class XlogyOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "Xlogy op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.xlogy", XlogyOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class XlogyXGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "XlogyXGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.xlogy_x_grad", XlogyXGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class XlogyYGradOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "XlogyYGrad op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.xlogy_y_grad", XlogyYGradOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

class ZeroLikeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return Error::RuntimeError() << "ZeroLike op has no attribute named " << attr_name;
  }
};
#ifdef NEED_REGISTER_OP_INTERP_CTX
REGISTER_OP_INTERP_CTX("user.zero_like", ZeroLikeOpInterpCtx);
#endif  // NEED_REGISTER_OP_INTERP_CTX

#endif  // DEFINE_OP_INTERP_CTX_CLASS
