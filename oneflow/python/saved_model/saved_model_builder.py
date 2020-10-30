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
import oneflow.core.job.saved_model_pb2 as model_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_pb2
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.session_context as session_ctx
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export

import os
import itertools
import google.protobuf.text_format as pb_text_format
from typing import Callable, Dict, List, Tuple


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


def Lbn2Lbi(lbn):
    assert isinstance(lbn, str)
    assert "/" in lbn, 'invalid lbn "{}"'.format(lbn)
    [op_name, blob_name] = lbn.split("/")
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_name
    lbi.blob_name = blob_name
    return lbi


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
        input_blob_name_4_lbn: Dict[str, str],
        output_blob_name_4_lbn: Dict[str, str],
        method_name: str = "serving",
    ):
        assert job_func.__name__ in compiler.job_map

        self.saved_model_proto_.graphs[job_func.__name__].CopyFrom(
            compiler.job_map[job_func.__name__].net
        )

        op_graph = compiler.job_map[job_func.__name__].net.op
        # fill the sinagures field
        singature_def = model_pb.SignatureDef()
        for ibn, lbn in input_blob_name_4_lbn.items():
            input_def = model_pb.InputDef()

            # find consumer op names
            for op in op_graph:
                if op.HasField("user_conf"):
                    for key in op.user_conf.input:
                        if op.user_conf.input[key].s[0] == lbn:
                            input_def.consumer_op_names.append(op.name)
                            input_def.consumer_op_input_bns.append(key)
                            break

            # fill blob_conf
            input_def.blob_conf.CopyFrom(GetBlobConf(job_func.__name__, lbn))

            # fill consumers and consumer_bns
            singature_def.inputs[ibn].CopyFrom(input_def)

        for obn, lbn in output_blob_name_4_lbn.items():
            output_def = model_pb.OutputDef()
            found_producer = False
            # find producer op name
            for op in op_graph:
                if op.HasField("user_conf"):
                    for key in op.user_conf.output:
                        if op.user_conf.output[key].s[0] == lbn:
                            output_def.producer_op_name = op.name
                            found_producer = True
                            break
                if found_producer == True:
                    break

            singature_def.outputs[obn].CopyFrom(output_def)

        singature_def.method_name = method_name
        self.saved_model_proto_.signatures[job_func.__name__].CopyFrom(singature_def)
        return self

    def AddJobGraph(
        self,
        job_graph_name: str,
        input_jobs: List[Callable],
        output_jobs: List[Callable],
        dependency_between_jobs: List[
            Tuple[Tuple[Callable, Callable], Tuple[str, str]]
        ],
    ):
        job_graph = model_pb.JobGraph()
        job_graph.job_graph_name = job_graph_name

        for in_job in input_jobs:
            job_graph.input_job_names.append(in_job.__name__)

        for out_job in output_jobs:
            job_graph.output_job_names.append(out_job.__name__)

        for dep in dependency_between_jobs:
            producer_info = model_pb.JobGraph.ProducerInfo()
            job_tuple, blob_name_tuple = dep[0], dep[1]
            curr_job_name, producer_job_name = (
                job_tuple[0].__name__,
                job_tuple[1].__name__ if job_tuple[1] != None else None,
            )
            curr_job_in_bn, producer_job_out_bn = blob_name_tuple[0], blob_name_tuple[1]

            if producer_job_name != None:
                producer_info.producer_job_name = producer_job_name
            if producer_job_out_bn != None:
                producer_info.producer_job_output_blob_name = producer_job_out_bn
            producer_info.current_job_input_blob_name = curr_job_in_bn

            job_graph.dependency_between_jobs[curr_job_name].infos.append(producer_info)

        self.saved_model_proto_.job_graphs.append(job_graph)

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


@oneflow_export("SavedModelBuilder")
class SavedModelBuilderV2(object):
    """
    """

    class JobBuilder(object):
        def __init__(self, job_name, model_builder):
            self.job_name_ = job_name
            self.model_builder_ = model_builder
            self.input_name2lbn_ = {}
            self.output_name2lbn_ = {}

        def Input(self, input_name, lbn):
            if input_name in self.input_name2lbn_:
                raise ValueError('input_name "{}" already exist'.format(input_name))

            self.input_name2lbn_[input_name] = lbn
            return self

        def Output(self, output_name, lbn):
            if output_name in self.output_name2lbn_:
                raise ValueError('output_name "{}" already exist'.format(output_name))

            self.output_name2lbn_[output_name] = lbn
            return self

        def Complete(self):
            return self.model_builder_

        def GetSignature(self):
            for lbn in itertools.chain(
                self.input_name2lbn_.values(), self.output_name2lbn_.values()
            ):
                c_api_util.JobBuildAndInferCtx_CheckLbnValidAndExist(
                    self.job_name_, lbn
                )

            signature = job_conf_pb.JobSignatureDef()
            for name, lbn in self.input_name2lbn_.items():
                input_def = signature.inputs[name]
                input_def.lbi.CopyFrom(Lbn2Lbi(lbn))
                input_def.blob_conf.CopyFrom(GetBlobConf(self.job_name_, lbn))
            for name, lbn in self.output_name2lbn_.items():
                output_def = signature.outputs[name]
                output_def.lbi.CopyFrom(Lbn2Lbi(lbn))

            return signature

    def __init__(self, save_path: str):
        assert isinstance(save_path, str)

        self.saved_model_dir_ = save_path
        self.version_ = None
        self.checkpoint_dir_ = "variables"
        self.saved_model_pb_filename_ = "saved_model.pb"
        self.saved_model_pbtxt_filename_ = "saved_model.prototxt"
        self.saved_model_proto_ = model_pb.SavedModel()
        self.job_name2job_builder_ = {}

    def ModelName(self, model_name: str):
        assert isinstance(model_name, str)

        self.saved_model_proto_.name = model_name
        return self

    def Version(self, version: int):
        self.version_ = version
        return self

    def Job(self, job_func: Callable):
        job_name = job_func.__name__
        if job_name not in self.job_name2job_builder_:
            self.job_name2job_builder_[job_name] = self.JobBuilder(job_name, self)

        return self.job_name2job_builder_[job_name]

    # TODO: JobGraph
    # TODO: check multi user job input/output name conflict
    @session_ctx.try_init_default_session
    def Save(self):
        sess = session_ctx.GetDefaultSession()
        for job_name, job_builder in self.job_name2job_builder_.items():
            job = sess.Job(job_name)
            self.saved_model_proto_.graphs[job_name].CopyFrom(job.net)
            self.saved_model_proto_.signatures_v2[job_name].CopyFrom(
                job_builder.GetSignature()
            )

        if not os.path.exists(self.saved_model_dir_):
            os.makedirs(self.saved_model_dir_)

        if self.version_ is None:
            raise ValueError("model version not set")

        version_dir = os.path.join(self.saved_model_dir_, str(self.version_))
        if os.path.exists(version_dir):
            raise ValueError(
                'Directory of model "{}" version "{}" already exist.'.format(
                    self.saved_model_dir_, self.version_
                )
            )
        os.makedirs(version_dir)
        self.saved_model_proto_.version = self.version_

        checkpoint_dir = os.path.join(version_dir, self.checkpoint_dir_)
        checkpoint = flow.train.CheckPoint()
        checkpoint.save(checkpoint_dir)
        self.saved_model_proto_.checkpoint_dir.append(self.checkpoint_dir_)

        saved_model_pb_path = os.path.join(version_dir, self.saved_model_pb_filename_)
        with open(saved_model_pb_path, "wb") as writer:
            writer.write(self.saved_model_proto_.SerializeToString())

        saved_model_pbtxt_path = os.path.join(
            version_dir, self.saved_model_pbtxt_filename_
        )
        with open(saved_model_pbtxt_path, "wt") as writer:
            writer.write(pb_text_format.MessageToString(self.saved_model_proto_))
