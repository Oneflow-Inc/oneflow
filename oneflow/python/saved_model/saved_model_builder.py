"""
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
"""

import oneflow as flow
import oneflow.core.common.data_type_pb2 as dtype_util
import oneflow.core.common.shape_pb2 as shape_pb2
import oneflow.core.job.dlnet_conf_pb2 as op_list_pb
import oneflow.core.job.saved_model_pb2 as model_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_pb2
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export

import os
from typing import Callable, List


def GetBlobConf(job_func_name: str, logical_blob_name: str):
    blob_conf = op_conf_pb2.InterfaceBlobConf()

    # get shape
    shape = c_api_util.JobBuildAndInferCtx_GetStaticShape(
        job_func_name, logical_blob_name
    )
    shape_proto = shape_pb2.ShapeProto()
    for dim in shape:
        shape_proto.dim.append(dim)
    blob_conf.shape.CopyFrom(shape_proto)

    # get data type
    blob_conf.data_type = c_api_util.JobBuildAndInferCtx_GetDataType(
        job_func_name, logical_blob_name
    )

    # get split axis
    split_axis = c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
        job_func_name, logical_blob_name
    )
    if split_axis != None:
        split_axis_proto = dtype_util.OptInt64()
        split_axis_proto.value = split_axis
        blob_conf.split_axis.CopyFrom(split_axis_proto)

    # get batch axis
    batch_axis = c_api_util.JobBuildAndInferCtx_GetBatchAxis(
        job_func_name, logical_blob_name
    )
    if batch_axis != None:
        batch_axis_proto = dtype_util.OptInt64()
        batch_axis_proto.value = batch_axis
        blob_conf.batch_axis.CopyFrom(batch_axis_proto)

    # is dynamic
    blob_conf.is_dynamic = c_api_util.JobBuildAndInferCtx_IsDynamic(
        job_func_name, logical_blob_name
    )

    # is tensor list
    blob_conf.is_tensor_list = c_api_util.JobBuildAndInferCtx_IsTensorList(
        job_func_name, logical_blob_name
    )

    return blob_conf


# example usage
# saved_model_builder = SavedModelBuilder("./models", 1)
# saved_model_builder
#     .ModelName("alexnet")
#     .AddJob(alexnet_train_job, input_blob_names, output_blob_names) # get from xxx.logical_blob_name
#     .Save()


@oneflow_export("saved_model.SavedModelBuilder")
class SavedModelBuilder(object):
    def __init__(self, path: str, version: int = 1):
        assert path != None

        self.model_path_ = path
        self.version = str(version)
        self.saved_model_proto_ = model_pb.SavedModel()
        self.saved_model_proto_.version = version
        self.saved_model_proto_.checkpoint_dir.append("variables")

        if not os.path.exists(path):
            os.makedirs(path)
        assert not os.path.exists(os.path.join(path, self.version))
        os.makedirs(os.path.join(path, self.version))

    def ModelName(self, name: str):
        self.saved_model_proto_.name = name
        return self

    def AddJobFunction(
        self,
        job_func: Callable,
        input_blob_names: List[str],
        output_blob_names: List[str],
        method_name: str = "serving",
    ):
        assert job_func.__name__ in compiler.job_map

        self.saved_model_proto_.graphs[job_func.__name__].CopyFrom(
            compiler.job_map[job_func.__name__].net
        )

        op_graph = compiler.job_map[job_func.__name__].net.op
        # fill the sinagures field
        singature_def = model_pb.SignatureDef()
        for ibn in input_blob_names:
            input_def = model_pb.InputDef()

            # find consumer op names
            for op in op_graph:
                if op.HasField("user_conf"):
                    for key in op.user_conf.input:
                        if op.user_conf.input[key].s[0] == ibn:
                            input_def.consumer_op_names.append(op.name)
                            input_def.consumer_op_input_bns.append(key)
                            break

            # fill blob_conf
            input_def.blob_conf.CopyFrom(GetBlobConf(job_func.__name__, ibn))

            # fill consumers and consumer_bns
            singature_def.inputs[ibn].CopyFrom(input_def)

        for obn in output_blob_names:
            output_def = model_pb.OutputDef()
            found_producer = False
            # find producer op name
            for op in op_graph:
                if op.HasField("user_conf"):
                    for key in op.user_conf.output:
                        if op.user_conf.output[key].s[0] == obn:
                            output_def.producer_op_name = op.name
                            found_producer = True
                            break
                if found_producer == True:
                    break

            singature_def.outputs[obn].CopyFrom(output_def)

        singature_def.method_name = method_name
        self.saved_model_proto_.singatures[job_func.__name__].CopyFrom(singature_def)
        return self

    def Save(self):
        session_ctx.GetDefaultSession().TryInit()

        # checkpoint save
        checkpoint = flow.train.CheckPoint()
        checkpoint.save(
            os.path.join(
                self.model_path_,
                self.version,
                self.saved_model_proto_.checkpoint_dir[0],
            )
        )

        # save op_list
        with open(
            os.path.join(self.model_path_, self.version, "saved_model.pb"), "wb"
        ) as f:
            f.write(self.saved_model_proto_.SerializeToString())
        with open(
            os.path.join(self.model_path_, self.version, "saved_model.prototxt"), "w"
        ) as f:
            f.write("%s\n" % self.saved_model_proto_)
