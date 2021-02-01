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
from __future__ import absolute_import

import re
from contextlib import contextmanager

import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.job.scope_pb2 as scope_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.operator.op_node_signature_pb2 as op_node_signature_pb
import oneflow.core.register.blob_desc_pb2 as blob_desc_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.eager.object_storage as object_storage
import oneflow.python.eager.symbol as symbol_util
import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow.python.framework.parallel_conf_util as parallel_conf_util
import oneflow_api.oneflow.core.job.scope as scope_cfg
import oneflow.python.framework.balanced_splitter as balanced_splitter
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.python_interpreter_util as python_interpreter_util
from oneflow.python.eager.opkernel_object import OpKernelObject
import oneflow
import oneflow_api.oneflow.core.vm.instruction as instr_cfg
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api.oneflow.core.job.job_conf as job_conf_cfg
import oneflow_api.oneflow.core.operator.op_node_signature as op_node_signature_cfg
import oneflow_api.oneflow.core.eager.eager_symbol as eager_symbol_cfg
from google.protobuf import text_format
import oneflow_api


def PhysicalRun(build):
    return _Run(
        build,
        oneflow_api.vm.PhysicalIdGenerator(),
        oneflow_api.vm.RunPhysicalInstruction,
        _ReleasePhysicalObject,
    )


def LogicalRun(build):
    return _Run(
        build,
        oneflow_api.vm.LogicalIdGenerator(),
        oneflow_api.vm.RunLogicalInstruction,
        _ReleaseLogicalObject,
    )


def _Run(build, id_generator, run_api, release_object):
    instruction_list = session_ctx.GetDefaultSession().instruction_list
    eager_symbol_list = session_ctx.GetDefaultSession().eager_symbol_list
    assert isinstance(instruction_list, instr_cfg.InstructionListProto)
    assert isinstance(eager_symbol_list, eager_symbol_cfg.EagerSymbolList)
    build(
        oneflow_api.deprecated.InstructionsBuilder(
            id_generator, instruction_list, eager_symbol_list, release_object
        )
    )
    run_api(instruction_list, eager_symbol_list)
    instruction_list.clear_instruction()
    eager_symbol_list.clear_eager_symbol()


def _DefaultBlobObject4Ibn(ibn):
    raise NotImplementedError


def StatelessCall(self, op_attribute, parallel_conf, bn_in_op2blob_object={}):
    op_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
    self._CheckRefInBlobObjectParallelDesc(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )

    def FetchDelegateBlobObject(x_blob_object, op_arg_parallel_attr):
        return boxing_util.BoxingTo(self, x_blob_object, op_arg_parallel_attr)

    def GetDelegateBlobObject(blob_object, op_arg_parallel_attr):
        return _FindOrCreateDelegateBlobObject(
            self, FetchDelegateBlobObject, blob_object, op_arg_parallel_attr
        )

    self._StatelessCall(
        "compute",
        op_attribute,
        op_parallel_desc_sym=op_parallel_desc_sym,
        blob_parallel_desc_sym=op_parallel_desc_sym,
        bn_in_op2blob_object=bn_in_op2blob_object,
        get_delegate_blob_object=GetDelegateBlobObject,
    )


def NoBoxingStatelessCall(self, op_attribute, parallel_conf, bn_in_op2blob_object={}):
    op_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
    self._CheckRefInBlobObjectParallelDesc(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )

    def FetchDelegateBlobObject(blob_object, op_arg_parallel_attr):
        from_pd = blob_object.parallel_desc_symbol
        to_pd = op_arg_parallel_attr.parallel_desc_symbol
        if from_pd == to_pd:
            return blob_object
        assert from_pd.device_tag == "cpu"
        assert to_pd.device_tag == "cpu"
        assert from_pd.parallel_num == to_pd.parallel_num
        from_machine_ids = dict(from_pd.machine_id2device_id_list).keys()
        to_machine_ids = dict(to_pd.machine_id2device_id_list).keys()
        if (
            len(from_pd.machine_id2device_id_list) == from_pd.parallel_num
            and from_machine_ids == to_machine_ids
        ):
            return self.BroadcastBlobReference(blob_object, to_pd)
        return self.Build121To(blob_object, to_pd)

    def GetDirectOr121BlobObject(blob_object, op_arg_parallel_attr):
        return _FindOrCreateDelegateBlobObject(
            self, FetchDelegateBlobObject, blob_object, op_arg_parallel_attr
        )

    self._StatelessCall(
        "compute",
        op_attribute,
        op_parallel_desc_sym=op_parallel_desc_sym,
        blob_parallel_desc_sym=op_parallel_desc_sym,
        bn_in_op2blob_object=bn_in_op2blob_object,
        get_delegate_blob_object=GetDirectOr121BlobObject,
    )


def NoBoxingCudaD2HStatelessCall(
    self, op_attribute, in_parallel_conf, bn_in_op2blob_object={}
):
    op_parallel_desc_sym = self.GetParallelDescSymbol(in_parallel_conf)
    blob_parallel_desc_sym = boxing_util.TryReplaceDeviceTag(
        self, op_parallel_desc_sym, "cpu"
    )
    self._CheckRefInBlobObjectParallelDesc(
        op_attribute, blob_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )

    def GetDirectBlobObject(blob_object, op_arg_parallel_attr):
        return blob_object

    self._StatelessCall(
        "copy_d2h",
        op_attribute,
        op_parallel_desc_sym=op_parallel_desc_sym,
        blob_parallel_desc_sym=blob_parallel_desc_sym,
        bn_in_op2blob_object=bn_in_op2blob_object,
        get_delegate_blob_object=GetDirectBlobObject,
    )


def NoBoxingCudaH2DStatelessCall(
    self, op_attribute, out_parallel_conf, bn_in_op2blob_object={}
):
    op_parallel_desc_sym = self.GetParallelDescSymbol(out_parallel_conf)
    self._CheckRefInBlobObjectParallelDesc(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )

    def GetDirectBlobObject(blob_object, op_arg_parallel_attr):
        return blob_object

    self._StatelessCall(
        "copy_h2d",
        op_attribute,
        op_parallel_desc_sym=op_parallel_desc_sym,
        blob_parallel_desc_sym=op_parallel_desc_sym,
        bn_in_op2blob_object=bn_in_op2blob_object,
        get_delegate_blob_object=GetDirectBlobObject,
    )


def RawStatelessCall(self, op_attribute, parallel_conf, bn_in_op2blob_object={}):
    op_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
    self._CheckRefInBlobObjectParallelDesc(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )

    def GetDirectBlobObject(blob_object, op_arg_parallel_attr):
        return blob_object

    self._StatelessCall(
        "compute",
        op_attribute,
        op_parallel_desc_sym=op_parallel_desc_sym,
        blob_parallel_desc_sym=op_parallel_desc_sym,
        bn_in_op2blob_object=bn_in_op2blob_object,
        get_delegate_blob_object=GetDirectBlobObject,
    )


def StatefulCall(self, op_attribute, opkernel_object, bn_in_op2blob_object={}):
    op_parallel_desc_sym = opkernel_object.parallel_desc_symbol
    parallel_sig = op_attribute.parallel_signature
    assert parallel_sig.HasField("op_parallel_desc_symbol_id")
    assert op_parallel_desc_sym.symbol_id == parallel_sig.op_parallel_desc_symbol_id
    self._CheckRefInBlobObjectParallelDesc(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )

    def FetchDelegateBlobObject(x_blob_object, op_arg_parallel_attr):
        return boxing_util.BoxingTo(self, x_blob_object, op_arg_parallel_attr)

    def GetDelegateBlobObject(blob_object, op_arg_parallel_attr):
        return _FindOrCreateDelegateBlobObject(
            self, FetchDelegateBlobObject, blob_object, op_arg_parallel_attr
        )

    self._StatefulCall(
        op_attribute,
        opkernel_object=opkernel_object,
        bn_in_op2blob_object=bn_in_op2blob_object,
        get_delegate_blob_object=GetDelegateBlobObject,
    )


def InsertRemoveForeignCallbackInstruction(self, object_id, callback):
    unique_callback_id = python_callback.GetIdForRegisteredCallback(callback)
    instruction = instr_cfg.InstructionProto()
    instruction.set_instr_type_name("RemoveForeignCallback")
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.DelObjectOperand(object_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.Int64Operand(unique_callback_id)
    )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)


def FetchBlobHeader(self, blob_object, callback):
    return self._FetchBlob("FetchBlobHeader", blob_object, callback)


def FetchBlobBody(self, blob_object, callback):
    return self._FetchBlob("FetchBlobBody", blob_object, callback)


def MakeLazyRefBlobObject(self, interface_op_name):
    sess = session_ctx.GetDefaultSession()
    op_attribute = sess.OpAttribute4InterfaceOpName(interface_op_name)
    assert len(op_attribute.output_bns) == 1
    obn = op_attribute.output_bns[0]

    parallel_conf = sess.ParallelConf4LazyInterfaceOpName(interface_op_name)
    if not isinstance(
        parallel_conf, oneflow_api.oneflow.core.job.placement.ParallelConf
    ):
        parallel_conf_cfg = placement_cfg.ParallelConf()
        parallel_conf_cfg.set_device_tag(parallel_conf.device_tag)
        for device_name in parallel_conf.device_name:
            parallel_conf_cfg.add_device_name(device_name)
        parallel_conf = parallel_conf_cfg
    blob_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)

    op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
        blob_parallel_desc_sym, str(op_attribute), obn
    )
    op_arg_blob_attr = oneflow_api.GetOpArgBlobAttribute(str(op_attribute), obn)

    blob_object = self.NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr)
    self.LazyReference(blob_object, interface_op_name)
    return blob_object


def BuildInitialScope(
    self, session_id, job_conf, device_tag, machine_device_ids, is_mirrored,
):
    scope_proto = scope_cfg.ScopeProto()
    scope_proto.set_session_id(session_id)
    job_conf_sym = self.GetJobConfSymbol(job_conf)
    scope_proto.set_job_desc_symbol_id(job_conf_sym.symbol_id)
    parallel_conf = parallel_conf_util.MakeParallelConf(device_tag, machine_device_ids)
    device_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
    scope_proto.set_device_parallel_desc_symbol_id(device_parallel_desc_sym.symbol_id)
    parallel_conf = parallel_conf_util.MakeParallelConf("cpu", machine_device_ids)
    host_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
    scope_proto.set_host_parallel_desc_symbol_id(host_parallel_desc_sym.symbol_id)
    if is_mirrored:
        scope_proto.mutable_opt_mirrored_parallel_conf().mutable_mirrored_parallel()
    else:
        scope_proto.mutable_opt_mirrored_parallel_conf().clear_mirrored_parallel()
    return self.GetScopeSymbol(scope_proto)


def BuildScopeWithNewParallelDesc(self, scope, device_tag, machine_device_ids):
    if isinstance(machine_device_ids, str):
        machine_device_ids = [machine_device_ids]

    def SetScopeProto(scope_proto):
        parallel_conf = parallel_conf_util.MakeParallelConf(
            device_tag, machine_device_ids
        )
        device_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
        parallel_conf = parallel_conf_util.MakeParallelConf("cpu", machine_device_ids)
        host_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
        scope_proto.set_device_parallel_desc_symbol_id(
            device_parallel_desc_sym.symbol_id
        )
        scope_proto.set_host_parallel_desc_symbol_id(host_parallel_desc_sym.symbol_id)

    return self.BuildScopeByProtoSetter(scope, SetScopeProto)


def BuildScopeWithNewParallelConf(self, scope, parallel_conf):
    tag_and_dev_ids = parallel_conf_util.GetDeviceTagAndMachineDeviceIds(parallel_conf)
    return self.BuildScopeWithNewParallelDesc(scope, *tag_and_dev_ids)


def GetSharedOpKernelObject4ParallelConfSymbol(self, parallel_desc_sym):
    if object_storage.HasSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym):
        return object_storage.GetSharedOpKernelObject4ParallelConfSymbol(
            parallel_desc_sym
        )
    object_id = self.NewSharedOpKernelObjectId4ParallelConfSymbolId(parallel_desc_sym)
    obj = oneflow_api.Object(object_id, parallel_desc_sym)
    object_storage.SetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym, obj)
    return obj


@contextmanager
def CudaHostPinBlob(self, blob_object):
    self.CudaHostRegisterBlob(blob_object)
    try:
        yield
    finally:
        self.CudaHostUnregisterBlob(blob_object)


def NewOpKernelObject(self, op_conf):
    assert op_conf.HasField("scope_symbol_id")
    scope_symbol = oneflow_api.GetScopeSymbol(op_conf.scope_symbol_id)
    op_conf_sym = self._GetOpConfSymbol(op_conf)
    parallel_desc_sym_id = c_api_util.GetOpParallelSymbolId(op_conf)
    parallel_desc_symbol = oneflow_api.GetPlacementSymbol(parallel_desc_sym_id)
    object_id = self._NewOpKernelObject(
        parallel_desc_symbol, scope_symbol.job_desc_symbol, op_conf_sym
    )
    return OpKernelObject(object_id, op_conf, self.object_releaser())


def Build121To(self, blob_object, parallel_desc_symbol):
    ref_blob_object = _MakeNewBlobObjectLike(self, blob_object, parallel_desc_symbol)
    self.Build121AssignInstruction(ref_blob_object, blob_object)
    return ref_blob_object


def _NewOpKernelObject(self, parallel_desc_symbol, job_desc_sym, op_conf_sym):
    object_id = self.NewObjectId(parallel_desc_symbol)
    instruction = instr_cfg.InstructionProto()
    instruction.set_instr_type_name("InitOpKernelObject")
    instruction.set_parallel_desc_symbol_id(parallel_desc_symbol.symbol_id)
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.SymbolOperand(job_desc_sym.symbol_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.SymbolOperand(op_conf_sym.symbol_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.MutOperand(object_id)
    )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)
    return object_id


def _StatelessCall(
    self,
    stream_tag,
    op_attribute,
    op_parallel_desc_sym=None,
    blob_parallel_desc_sym=None,
    bn_in_op2blob_object={},
    get_delegate_blob_object=None,
):
    assert callable(get_delegate_blob_object)
    if op_attribute.parallel_signature.HasField("op_parallel_desc_symbol_id"):
        symbol_id = op_attribute.parallel_signature.op_parallel_desc_symbol_id
        op_parallel_desc_sym = oneflow_api.GetPlacementSymbol(symbol_id)
    assert op_parallel_desc_sym is not None

    def DelegateBlobObject4Ibn(ibn):
        op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
            op_parallel_desc_sym, str(op_attribute), ibn
        )
        return get_delegate_blob_object(bn_in_op2blob_object[ibn], op_arg_parallel_attr)

    op_conf = op_attribute.op_conf
    assert op_conf.HasField("scope_symbol_id"), op_conf
    scope_symbol = oneflow_api.GetScopeSymbol(op_conf.scope_symbol_id)
    job_desc_sym = scope_symbol.job_desc_symbol
    op_conf_sym = self._GetOpConfSymbol(op_conf)
    op_node_signature_sym = self._GetOpNodeSignatureSymbol(op_attribute)
    opkernel_obj = self.GetSharedOpKernelObject4ParallelConfSymbol(op_parallel_desc_sym)
    assert opkernel_obj.parallel_desc_symbol == op_parallel_desc_sym, (
        str(opkernel_obj.parallel_desc_symbol.parallel_conf),
        str(op_parallel_desc_sym.parallel_conf),
    )
    const_input_operand_blob_objects = self._GetConstInputOperandBlobObjects(
        op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
    )
    mutable_input_operand_blob_objects = self._GetMutableInputOperandBlobObjects(
        op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
    )
    mut1_operand_blob_objects = self._GetMut1OperandBlobObjects(
        op_attribute, blob_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )
    mut2_operand_blob_objects = self._GetMut2OperandBlobObjects(
        op_attribute, blob_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )
    is_user_op = op_attribute.op_conf.HasField("user_conf")
    instruction_prefix = "User" if is_user_op else "System"
    self._StatelessCallOpKernel(
        "%s.%sStatelessCallOpKernel" % (stream_tag, instruction_prefix),
        op_parallel_desc_sym,
        job_desc_sym,
        op_conf_sym,
        op_node_signature_sym,
        opkernel_obj,
        const_input_operand_blob_objects,
        mutable_input_operand_blob_objects,
        mut1_operand_blob_objects,
        mut2_operand_blob_objects,
    )


def _StatefulCall(
    self, op_attribute, opkernel_object, bn_in_op2blob_object, get_delegate_blob_object,
):
    op_parallel_desc_sym = opkernel_object.parallel_desc_symbol

    def DelegateBlobObject4Ibn(ibn):
        op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
            op_parallel_desc_sym, str(op_attribute), ibn
        )
        return get_delegate_blob_object(bn_in_op2blob_object[ibn], op_arg_parallel_attr)

    op_node_signature_sym = self._GetOpNodeSignatureSymbol(op_attribute)
    const_input_operand_blob_objects = self._GetConstInputOperandBlobObjects(
        op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
    )
    mutable_input_operand_blob_objects = self._GetMutableInputOperandBlobObjects(
        op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
    )
    mut1_operand_blob_objects = self._GetMut1OperandBlobObjects(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )
    mut2_operand_blob_objects = self._GetMut2OperandBlobObjects(
        op_attribute, op_parallel_desc_sym, bn_in_op2blob_object=bn_in_op2blob_object,
    )
    is_user_op = op_attribute.op_conf.HasField("user_conf")
    assert is_user_op
    instruction_prefix = "" if is_user_op else "System"
    self._StatefulCallOpKernel(
        "%sCallOpKernel" % instruction_prefix,
        op_parallel_desc_sym,
        opkernel_object,
        op_node_signature_sym,
        const_input_operand_blob_objects,
        mutable_input_operand_blob_objects,
        mut1_operand_blob_objects,
        mut2_operand_blob_objects,
    )


def _GetOpConfSymbol(self, op_conf):
    serialized_op_conf = op_conf.SerializeToString()
    if symbol_storage.HasSymbol4SerializedOpConf(serialized_op_conf):
        return symbol_storage.GetSymbol4SerializedOpConf(serialized_op_conf)
    symbol_id = self._NewSymbolId4OpConf(op_conf)
    symbol = symbol_util.Symbol(symbol_id, op_conf)
    symbol_storage.SetSymbol4Id(symbol_id, symbol)
    symbol_storage.SetSymbol4SerializedOpConf(serialized_op_conf, symbol)
    return symbol


def _GetOpNodeSignatureSymbol(self, op_attribute):
    new_op_node_signature = oneflow_api.deprecated.MakeOpNodeSignatureFromSerializedOpAttribute(
        str(op_attribute)
    )
    if oneflow_api.HasOpNodeSignatureSymbol(new_op_node_signature):
        return oneflow_api.GetOpNodeSignatureSymbol(new_op_node_signature)
    symbol_id = self.NewSymbolId4OpNodeSignature(new_op_node_signature)
    oneflow_api.AddOpNodeSignatureSymbol(symbol_id, new_op_node_signature)
    return oneflow_api.GetOpNodeSignatureSymbol(symbol_id)


def _GetConstInputOperandBlobObjects(self, op_attribute, blob_object4ibn=None):
    assert callable(blob_object4ibn)
    const_input_operand_blob_objects = []
    for ibn in op_attribute.input_bns:
        ibn2modifier = op_attribute.arg_modifier_signature.ibn2input_blob_modifier
        if ibn2modifier[ibn].is_mutable:
            continue
        ibn_sym = self.GetSymbol4String(ibn)
        in_object = blob_object4ibn(ibn)
        const_input_operand_blob_objects.append((ibn_sym, in_object))
    return const_input_operand_blob_objects


def _GetMutableInputOperandBlobObjects(self, op_attribute, blob_object4ibn=None):
    mutable_input_operand_blob_objects = []
    for ibn in op_attribute.input_bns:
        ibn2modifier = op_attribute.arg_modifier_signature.ibn2input_blob_modifier
        if not ibn2modifier[ibn].is_mutable:
            continue
        ibn_sym = self.GetSymbol4String(ibn)
        in_object = blob_object4ibn(ibn)
        mutable_input_operand_blob_objects.append((ibn_sym, in_object))
    return mutable_input_operand_blob_objects


def _GetMut1OperandBlobObjects(
    self, op_attribute, parallel_desc_sym, bn_in_op2blob_object={}
):
    mut1_operand_blob_objects = []

    def GetOutBlobParallelDescSymbol(obn):
        parallel_signature = op_attribute.parallel_signature
        bn2symbol_id = parallel_signature.bn_in_op2parallel_desc_symbol_id
        if obn in bn2symbol_id:
            return oneflow_api.GetPlacementSymbol(bn2symbol_id[obn])
        else:
            return parallel_desc_sym

    def OutputBns():
        obn2modifier = op_attribute.arg_modifier_signature.obn2output_blob_modifier
        for obn in op_attribute.output_bns:
            if obn2modifier[obn].header_infered_before_compute:
                yield obn

        for tmp_bn in op_attribute.tmp_bns:
            yield tmp_bn

    for obn in OutputBns():
        obn_sym = self.GetSymbol4String(obn)
        op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
            GetOutBlobParallelDescSymbol(obn), str(op_attribute), obn
        )
        op_arg_blob_attr = oneflow_api.GetOpArgBlobAttribute(str(op_attribute), obn)
        out_blob_object = self.NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr)
        lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
        bn_in_op2blob_object[obn] = out_blob_object
        mut1_operand_blob_objects.append((obn_sym, out_blob_object))
    return mut1_operand_blob_objects


def _CheckRefInBlobObjectParallelDesc(
    self, op_attribute, op_parallel_desc_sym, bn_in_op2blob_object={}
):
    op_conf = op_attribute.op_conf
    for ibn in op_attribute.input_bns:
        ibn2modifier = op_attribute.arg_modifier_signature.ibn2input_blob_modifier
        if not ibn2modifier[ibn].is_mutable:
            continue
        ref_blob_object = bn_in_op2blob_object[ibn]
        assert op_parallel_desc_sym == ref_blob_object.parallel_desc_symbol, (
            "op_conf: %s\n%s\nv.s.\n%s"
            % (op_conf, op_parallel_desc_sym, ref_blob_object.parallel_desc_symbol)
        )


def _GetMut2OperandBlobObjects(
    self, op_attribute, parallel_desc_sym, bn_in_op2blob_object={}
):
    mut2_operand_blob_objects = []

    def GetOutBlobParallelDescSymbol(obn):
        parallel_signature = op_attribute.parallel_signature
        bn2symbol_id = parallel_signature.bn_in_op2parallel_desc_symbol_id
        if obn in bn2symbol_id:
            return oneflow_api.GetPlacementSymbol(bn2symbol_id[obn])
        else:
            return parallel_desc_sym

    for obn in op_attribute.output_bns:
        obn2modifier = op_attribute.arg_modifier_signature.obn2output_blob_modifier
        if obn2modifier[obn].header_infered_before_compute:
            continue
        obn_sym = self.GetSymbol4String(obn)
        op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
            GetOutBlobParallelDescSymbol(obn), str(op_attribute), obn
        )
        op_arg_blob_attr = oneflow_api.GetOpArgBlobAttribute(str(op_attribute), obn)
        out_blob_object = self.NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr)
        bn_in_op2blob_object[obn] = out_blob_object
        mut2_operand_blob_objects.append((obn_sym, out_blob_object))
    return mut2_operand_blob_objects


def _NewSymbolId4OpConf(self, op_conf):
    symbol_id = self.NewSymbolId()
    self._InitOpConfSymbol(symbol_id, op_conf)
    return symbol_id


def _StatelessCallOpKernel(
    self,
    instr_name,
    parallel_desc_sym,
    job_desc_sym,
    op_conf_sym,
    op_node_signature_sym,
    shared_opkernel_obj,
    const_input_operand_blob_objects,
    mutable_input_operand_blob_objects,
    mut1_operand_blob_objects,
    mut2_operand_blob_objects,
):
    instruction = instr_cfg.InstructionProto()
    instruction.set_instr_type_name(
        "%s.%s" % (parallel_desc_sym.device_tag, instr_name)
    )
    instruction.set_parallel_desc_symbol_id(parallel_desc_sym.symbol_id)
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.SymbolOperand(job_desc_sym.symbol_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.SymbolOperand(op_conf_sym.symbol_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.SymbolOperand(op_node_signature_sym.symbol_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.MutOperand(shared_opkernel_obj.object_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for ibn_sym, _ in const_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(ibn_sym.symbol_id)
        )
    for _, blob_object in const_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.ConstOperand(blob_object.object_id)
        )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for ibn_sym, _ in mutable_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(ibn_sym.symbol_id)
        )
    for _, blob_object in mutable_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.MutOperand(blob_object.object_id)
        )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for obn_sym, _ in mut1_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(obn_sym.symbol_id)
        )
    for _, blob_object in mut1_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.MutOperand(blob_object.object_id)
        )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for obn_sym, _ in mut2_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(obn_sym.symbol_id)
        )
    for _, blob_object in mut2_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.Mut2Operand(blob_object.object_id)
        )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)


def _StatefulCallOpKernel(
    self,
    instr_name,
    parallel_desc_sym,
    opkernel_object,
    op_node_signature_sym,
    const_input_operand_blob_objects,
    mutable_input_operand_blob_objects,
    mut1_operand_blob_objects,
    mut2_operand_blob_objects,
):
    instruction = instr_cfg.InstructionProto()
    instruction.set_instr_type_name(
        "%s.%s" % (parallel_desc_sym.device_tag, instr_name,)
    )
    instruction.set_parallel_desc_symbol_id(parallel_desc_sym.symbol_id)
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.MutOperand(opkernel_object.object_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.SymbolOperand(op_node_signature_sym.symbol_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for ibn_sym, _ in const_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(ibn_sym.symbol_id)
        )
    for _, blob_object in const_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.ConstOperand(blob_object.object_id)
        )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for ibn_sym, _ in mutable_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(ibn_sym.symbol_id)
        )
    for _, blob_object in mutable_input_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.MutOperand(blob_object.object_id)
        )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for obn_sym, _ in mut1_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(obn_sym.symbol_id)
        )
    for _, blob_object in mut1_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.MutOperand(blob_object.object_id)
        )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.OperandSeparator()
    )
    for obn_sym, _ in mut2_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.SymbolOperand(obn_sym.symbol_id)
        )
    for _, blob_object in mut2_operand_blob_objects:
        instruction.mutable_operand().Add().CopyFrom(
            oneflow_api.deprecated.vm.Mut2Operand(blob_object.object_id)
        )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)


def _InitOpConfSymbol(self, symbol_id, op_conf):
    instruction = instr_cfg.InstructionProto()
    instruction.set_instr_type_name("InitOperatorConfSymbol")
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.InitSymbolOperand(symbol_id)
    )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)
    eager_symbol = eager_symbol_pb.EagerSymbol()
    eager_symbol.symbol_id = symbol_id
    eager_symbol.op_conf_symbol.CopyFrom(op_conf)
    eager_symbol = oneflow_api.deprecated.MakeEagerSymbolByString(str(eager_symbol))
    self.eager_symbol_list().mutable_eager_symbol().Add().CopyFrom(eager_symbol)


def _FetchBlob(self, instruction_name, blob_object, fetcher):
    unique_callback_id = python_callback.GetIdForRegisteredCallback(fetcher)
    instruction = instr_cfg.InstructionProto()
    device_tag = blob_object.parallel_desc_symbol.device_tag
    instruction.set_instr_type_name("%s.%s" % (device_tag, instruction_name))
    instruction.set_parallel_desc_symbol_id(blob_object.parallel_desc_symbol.symbol_id)
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.ConstOperand(blob_object.object_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.Int64Operand(unique_callback_id)
    )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)


def FeedBlob(self, blob_object, feeder):
    unique_callback_id = python_callback.GetIdForRegisteredCallback(feeder)
    instruction = instr_cfg.InstructionProto()
    device_tag = blob_object.parallel_desc_symbol.device_tag
    instruction.set_instr_type_name("%s.%s" % (device_tag, "FeedBlob"))
    instruction.set_parallel_desc_symbol_id(blob_object.parallel_desc_symbol.symbol_id)
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.Mut2Operand(blob_object.object_id)
    )
    instruction.mutable_operand().Add().CopyFrom(
        oneflow_api.deprecated.vm.Int64Operand(unique_callback_id)
    )
    self.instruction_list().mutable_instruction().Add().CopyFrom(instruction)


def RegisterMethod4InstructionsBuilder():
    oneflow_api.deprecated.InstructionsBuilder.StatelessCall = StatelessCall
    oneflow_api.deprecated.InstructionsBuilder.NoBoxingStatelessCall = (
        NoBoxingStatelessCall
    )
    oneflow_api.deprecated.InstructionsBuilder.NoBoxingCudaD2HStatelessCall = (
        NoBoxingCudaD2HStatelessCall
    )
    oneflow_api.deprecated.InstructionsBuilder.NoBoxingCudaH2DStatelessCall = (
        NoBoxingCudaH2DStatelessCall
    )
    oneflow_api.deprecated.InstructionsBuilder.RawStatelessCall = RawStatelessCall
    oneflow_api.deprecated.InstructionsBuilder.StatefulCall = StatefulCall
    oneflow_api.deprecated.InstructionsBuilder.InsertRemoveForeignCallbackInstruction = (
        InsertRemoveForeignCallbackInstruction
    )
    oneflow_api.deprecated.InstructionsBuilder.FetchBlobHeader = FetchBlobHeader
    oneflow_api.deprecated.InstructionsBuilder.FetchBlobBody = FetchBlobBody
    oneflow_api.deprecated.InstructionsBuilder.MakeLazyRefBlobObject = (
        MakeLazyRefBlobObject
    )
    oneflow_api.deprecated.InstructionsBuilder.BuildInitialScope = BuildInitialScope
    oneflow_api.deprecated.InstructionsBuilder.BuildScopeWithNewParallelDesc = (
        BuildScopeWithNewParallelDesc
    )
    oneflow_api.deprecated.InstructionsBuilder.BuildScopeWithNewParallelConf = (
        BuildScopeWithNewParallelConf
    )
    oneflow_api.deprecated.InstructionsBuilder.GetSharedOpKernelObject4ParallelConfSymbol = (
        GetSharedOpKernelObject4ParallelConfSymbol
    )
    oneflow_api.deprecated.InstructionsBuilder.CudaHostPinBlob = CudaHostPinBlob
    oneflow_api.deprecated.InstructionsBuilder.NewOpKernelObject = NewOpKernelObject
    oneflow_api.deprecated.InstructionsBuilder.Build121To = Build121To
    oneflow_api.deprecated.InstructionsBuilder._NewOpKernelObject = _NewOpKernelObject
    oneflow_api.deprecated.InstructionsBuilder._StatelessCall = _StatelessCall
    oneflow_api.deprecated.InstructionsBuilder._StatefulCall = _StatefulCall
    oneflow_api.deprecated.InstructionsBuilder._GetOpConfSymbol = _GetOpConfSymbol
    oneflow_api.deprecated.InstructionsBuilder._GetOpNodeSignatureSymbol = (
        _GetOpNodeSignatureSymbol
    )
    oneflow_api.deprecated.InstructionsBuilder._GetConstInputOperandBlobObjects = (
        _GetConstInputOperandBlobObjects
    )
    oneflow_api.deprecated.InstructionsBuilder._GetMutableInputOperandBlobObjects = (
        _GetMutableInputOperandBlobObjects
    )
    oneflow_api.deprecated.InstructionsBuilder._GetMut1OperandBlobObjects = (
        _GetMut1OperandBlobObjects
    )
    oneflow_api.deprecated.InstructionsBuilder._CheckRefInBlobObjectParallelDesc = (
        _CheckRefInBlobObjectParallelDesc
    )
    oneflow_api.deprecated.InstructionsBuilder._GetMut2OperandBlobObjects = (
        _GetMut2OperandBlobObjects
    )
    oneflow_api.deprecated.InstructionsBuilder._NewSymbolId4OpConf = _NewSymbolId4OpConf
    oneflow_api.deprecated.InstructionsBuilder._StatelessCallOpKernel = (
        _StatelessCallOpKernel
    )
    oneflow_api.deprecated.InstructionsBuilder._StatefulCallOpKernel = (
        _StatefulCallOpKernel
    )
    oneflow_api.deprecated.InstructionsBuilder._InitOpConfSymbol = _InitOpConfSymbol
    oneflow_api.deprecated.InstructionsBuilder._FetchBlob = _FetchBlob
    oneflow_api.deprecated.InstructionsBuilder.FeedBlob = FeedBlob


def _MakeNewBlobObjectLike(builder, blob_object, new_parallel_desc_symbol):
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = id_util.UniqueStr("Input")
    op_conf.device_tag = new_parallel_desc_symbol.device_tag
    op_conf.input_conf.out = "out"
    cfg_interface_blob_conf = (
        oneflow_api.oneflow.core.operator.interface_blob_conf.InterfaceBlobConf()
    )
    blob_object.op_arg_parallel_attr.DumpToInterfaceBlobConf(cfg_interface_blob_conf)
    blob_object.op_arg_blob_attr.DumpToInterfaceBlobConf(cfg_interface_blob_conf)
    text_format.Parse(str(cfg_interface_blob_conf), op_conf.input_conf.blob_conf)
    op_conf.scope_symbol_id = oneflow.current_scope().symbol_id
    upstream_signature = op_node_signature_pb.OpNodeSignature()
    op_attribute = c_api_util.InferOpConf(op_conf, upstream_signature)
    parallel_conf = new_parallel_desc_symbol.parallel_conf
    bn_in_op2blob_object = {}
    builder.RawStatelessCall(
        op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
    )
    return bn_in_op2blob_object["out"]


def _FindOrCreateDelegateBlobObject(
    builder, Fetch, x_blob_object, op_arg_parallel_attr
):
    if x_blob_object.op_arg_parallel_attr == op_arg_parallel_attr:
        return x_blob_object
    blob_cache = blob_cache_util.FindOrCreateBlobCache(x_blob_object)
    return blob_cache.GetCachedDelegateBlobObject(op_arg_parallel_attr, Fetch)


def _GetOpConfBlobNameAttr(pb_message, field):
    if hasattr(pb_message, field):
        return getattr(pb_message, field)
    m = re.search("_(\d+)$", field)
    assert m is not None
    blob_name = field[0 : -len(m.group(0))]
    index = int(m.group(0)[1:])
    assert hasattr(pb_message, blob_name), (pb_message, blob_name)
    repeated_field = getattr(pb_message, blob_name)
    assert index >= 0
    assert index < len(repeated_field)
    return repeated_field[index]


def _ReleaseLogicalObject(obj, is_shutting_down=python_interpreter_util.IsShuttingDown):
    if is_shutting_down():
        return
    LogicalRun(lambda builder: builder.DeleteObject(obj))


def _ReleasePhysicalObject(
    obj, is_shutting_down=python_interpreter_util.IsShuttingDown
):
    if is_shutting_down():
        return
    PhysicalRun(lambda builder: builder.DeleteObject(obj))
