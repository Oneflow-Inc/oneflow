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

import os
import typing
from google.protobuf import text_format

import oneflow as flow
import oneflow_api
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.core.serving.saved_model_pb2 as saved_model_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_pb
import oneflow.core.operator.interface_blob_conf_pb2 as interface_blob_conf_pb
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("saved_model.ModelBuilder")
class ModelBuilder(object):
    DEFAULT_CHECKPOINT_DIR = "variables"
    DEFAULT_SAVED_MODEL_FILE_BASENAME = "saved_model"

    def __init__(self, save_path: str):
        if not isinstance(save_path, str):
            raise ValueError(
                "param 'save_path' must be str, but got {}".format(save_path)
            )

        self.version_ = None
        self.checkpoint_dir_ = self.DEFAULT_CHECKPOINT_DIR
        self.saved_model_dir_ = save_path
        self.saved_model_pb_filename_ = "{}.pb".format(
            self.DEFAULT_SAVED_MODEL_FILE_BASENAME
        )
        self.saved_model_pbtxt_filename_ = "{}.prototxt".format(
            self.DEFAULT_SAVED_MODEL_FILE_BASENAME
        )
        self.saved_model_proto_ = saved_model_pb.SavedModel()
        self.graph_builders_ = {}

    @property
    def proto(self):
        return self.saved_model_proto_

    def ModelName(self, model_name: str):
        assert isinstance(model_name, str)
        self.proto.name = model_name
        return self

    def Version(self, version: int):
        assert isinstance(version, int)
        self.version_ = version
        return self

    def AddFunction(self, func):
        func_name = func.__name__
        if func_name in self.graph_builders_:
            raise ValueError("function with name {} already exists".format(func_name))
        graph_builder = GraphBuilder(func_name, self)
        self.graph_builders_[func_name] = graph_builder
        if not self.proto.HasField("default_graph_name"):
            self.proto.default_graph_name = func_name
        return graph_builder

    def _check_input_output_name_conflict(self):
        name_set = set()
        lbn_set = set()

        def check_name_conflict(name, interface_def):
            if name in name_set:
                raise ValueError("input conflict, {} already exist".format(name))
            name_set.add(name)
            lbn = Lbi2Lbn(interface_def.lbi)
            if lbn in lbn_set:
                raise ValueError(
                    "input conflict, {} already bind to other input".format(lbn)
                )
            lbn_set.add(lbn)

        for _, graph_def in self.proto.graphs.items():
            for _, signature_def in graph_def.signatures.items():
                for input_name, input_def in signature_def.inputs.items():
                    check_name_conflict(input_name, input_def)
                for output_name, output_def in signature_def.outputs.items():
                    check_name_conflict(output_name, output_def)

    @session_ctx.try_init_default_session
    def Save(self):
        self._check_input_output_name_conflict()
        for _, graph_builder in self.graph_builders_.items():
            if not graph_builder.finished:
                graph_builder.Finish()

        sess = session_ctx.GetDefaultSession()
        for graph_name, graph_def in self.proto.graphs.items():
            job = sess.Job(graph_name)
            graph_def.op_list.extend(list(job.net.op))

        if not os.path.exists(self.saved_model_dir_):
            os.makedirs(self.saved_model_dir_)

        if self.version_ is None:
            raise ValueError("model version is not set")

        version_dir = os.path.join(self.saved_model_dir_, str(self.version_))
        if os.path.exists(version_dir):
            raise ValueError(
                'Directory of model "{}" version "{}" already exist.'.format(
                    self.saved_model_dir_, self.version_
                )
            )

        os.makedirs(version_dir)
        self.proto.version = self.version_

        checkpoint_path = os.path.join(version_dir, self.checkpoint_dir_)
        flow.checkpoint.save(checkpoint_path)
        self.proto.checkpoint_dir = self.checkpoint_dir_

        saved_model_pb_path = os.path.join(version_dir, self.saved_model_pb_filename_)
        with open(saved_model_pb_path, "wb") as writer:
            writer.write(self.saved_model_proto_.SerializeToString())

        saved_model_pbtxt_path = os.path.join(
            version_dir, self.saved_model_pbtxt_filename_
        )
        with open(saved_model_pbtxt_path, "wt") as writer:
            writer.write(text_format.MessageToString(self.saved_model_proto_))


@oneflow_export("saved_model.GraphBuilder")
class GraphBuilder(object):
    def __init__(self, name: str, model_builder: typing.Optional[ModelBuilder] = None):
        if not isinstance(name, str):
            raise ValueError("param 'name' must be str, but got {}".format(name))

        if not isinstance(model_builder, ModelBuilder) and model_builder is not None:
            raise ValueError(
                "param 'model_builder' must be a type of ModelBuilder or None"
            )

        if model_builder is not None:
            if name in model_builder.proto.graphs:
                raise ValueError(
                    "graph function ({}) is already added to model ({})".format(
                        name, model_builder.proto.name
                    )
                )

            self.proto_ = model_builder.proto.graphs[name]
            self.owner_ = model_builder
        else:
            self.proto_ = saved_model_pb.GraphDef()
            self.owner_ = None

        self.name_ = name
        self.finished_ = False
        self.signature_builders_ = {}

    @property
    def name(self):
        return self.name_

    @property
    def proto(self):
        return self.proto_

    @property
    def finished(self):
        return self.finished_

    def AddSignature(self, signature_name: str):
        assert isinstance(signature_name, str)
        if signature_name in self.signature_builders_:
            raise ValueError("signature name {} already exists".format(signature_name))
        signature_builder = SignatureBuilder(signature_name, self)
        self.signature_builders_[signature_name] = signature_builder
        if not self.proto.HasField("default_signature_name"):
            self.proto.default_signature_name = signature_name
        return signature_builder

    def Finish(self):
        assert self.finished is False
        for _, signature_def in self.proto.signatures.items():
            for _, input_def in signature_def.inputs.items():
                input_lbn = Lbi2Lbn(input_def.lbi)
                oneflow_api.JobBuildAndInferCtx_CheckLbnValidAndExist(
                    self.name, input_lbn
                )
                GetInterfaceBlobConf(self.name, input_lbn, input_def.blob_conf)

            for _, output_def in signature_def.outputs.items():
                oneflow_api.JobBuildAndInferCtx_CheckLbnValidAndExist(
                    self.name, Lbi2Lbn(output_def.lbi)
                )

        self.finished_ = True

    def OwnerModelBuilder(self):
        return self.owner_

    def AsDefault(self):
        if self.owner_ is not None:
            self.owner_.proto.default_graph_name = self.name

        return self


@oneflow_export("saved_model.SignatureBuilder")
class SignatureBuilder(object):
    def __init__(self, name: str, graph_builder: typing.Optional[GraphBuilder] = None):
        if not isinstance(name, str):
            raise ValueError("param 'name' must be str, but got {}".format(name))

        if not isinstance(graph_builder, GraphBuilder) and graph_builder is not None:
            raise ValueError(
                "param 'graph_builder' must be a type of GraphBuilder or None"
            )

        if graph_builder is not None:
            if name in graph_builder.proto.signatures:
                raise ValueError(
                    "signature ({}) already exist in graph ({})".format(
                        name, graph_builder.name,
                    )
                )

            self.proto_ = graph_builder.proto.signatures[name]
            self.owner_ = graph_builder
        else:
            self.proto_ = job_conf_pb.JobSignatureDef()
            self.owner_ = None

        self.name_ = name

    @property
    def name(self):
        return self.name_

    @property
    def proto(self):
        return self.proto_

    def Input(self, input_name: str, lbn: str):
        assert isinstance(input_name, str)
        assert isinstance(lbn, str)
        assert "/" in lbn

        if input_name in self.proto.inputs:
            raise ValueError(
                "input_name ({}) already exist in signature ({}) of graph ({})".format(
                    input_name, self.name, self.graph_builder_.name
                )
            )

        input_def = self.proto.inputs[input_name]
        Lbn2Lbi(lbn, input_def.lbi)
        return self

    def Output(self, output_name: str, lbn: str):
        assert isinstance(output_name, str)
        assert isinstance(lbn, str)
        assert "/" in lbn

        if output_name in self.proto.outputs:
            raise ValueError(
                "output_name ({}) already exist in signature ({}) of graph ({})".format(
                    output_name, self.name, self.graph_builder_.name
                )
            )

        output_def = self.proto.outputs[output_name]
        Lbn2Lbi(lbn, output_def.lbi)
        return self

    def OwnerGraphBuilder(self):
        return self.owner_

    def AsDefault(self):
        if self.owner_ is not None:
            self.owner_.proto.default_signature_name = self.name

        return self


def GetInterfaceBlobConf(job_name, lbn, blob_conf=None):
    assert isinstance(job_name, str)
    assert isinstance(lbn, str)
    if blob_conf is None:
        blob_conf = interface_blob_conf_pb.InterfaceBlobConf()
    else:
        assert isinstance(blob_conf, interface_blob_conf_pb.InterfaceBlobConf)

    shape = c_api_util.JobBuildAndInferCtx_GetStaticShape(job_name, lbn)
    dtype = c_api_util.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    split_axis = c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
        job_name, lbn
    )
    is_dynamic = c_api_util.JobBuildAndInferCtx_IsDynamic(job_name, lbn)
    is_tensor_list = c_api_util.JobBuildAndInferCtx_IsTensorList(job_name, lbn)

    blob_conf.shape.dim.extend(shape)
    blob_conf.data_type = dtype
    if split_axis is not None:
        blob_conf.split_axis.value = split_axis
    blob_conf.is_dynamic = is_dynamic
    blob_conf.is_tensor_list = is_tensor_list
    return blob_conf


def Lbn2Lbi(lbn, lbi=None):
    assert isinstance(lbn, str)
    assert "/" in lbn, 'invalid lbn "{}"'.format(lbn)

    [op_name, blob_name] = lbn.split("/")
    if lbi is None:
        lbi = logical_blob_id_pb.LogicalBlobId()

    lbi.op_name = op_name
    lbi.blob_name = blob_name
    return lbi


def Lbi2Lbn(lbi):
    assert isinstance(lbi, logical_blob_id_pb.LogicalBlobId)
    return "{}/{}".format(lbi.op_name, lbi.blob_name)
