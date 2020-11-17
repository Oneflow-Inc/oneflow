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
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.register.blob_desc_pb2 as blob_desc_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.eager.object as object_util
import oneflow.python.eager.object_storage as object_storage
import oneflow.python.eager.symbol as symbol_util
import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow.python.framework.balanced_splitter as balanced_splitter
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.scope_symbol as scope_symbol
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.eager.opkernel_object import OpKernelObject
import oneflow.python.vm.id_util as vm_id_util
import oneflow
import oneflow_api.oneflow.core.vm.instruction as instr_cfg
from google.protobuf import text_format


oneflow_api = oneflow.oneflow_api


def PhysicalRun(build):
    return _Run(
        build,
        vm_id_util.PhysicalIdGenerator(),
        c_api_util.RunPhysicalInstruction,
        _ReleasePhysicalObject,
    )


def LogicalRun(build):
    return _Run(
        build,
        vm_id_util.LogicalIdGenerator(),
        c_api_util.RunLogicalInstruction,
        _ReleaseLogicalObject,
    )


def _Run(build, id_generator, run_api, release_object):
    instruction_list = session_ctx.GetDefaultSession().instruction_list
    eager_symbol_list = session_ctx.GetDefaultSession().eager_symbol_list
    build(
        InstructionsBuilder(
            id_generator, release_object, instruction_list, eager_symbol_list
        )
    )
    run_api(instruction_list, eager_symbol_list)
    instruction_list.clear_instruction()
    eager_symbol_list.ClearField("eager_symbol")


def _DefaultBlobObject4Ibn(ibn):
    raise NotImplementedError


class InstructionsBuilder(object):
    def __init__(
        self, id_generator, release_object, instruction_list, eager_symbol_list
    ):
        self.id_generator_ = id_generator
        self.release_object_ = release_object
        assert isinstance(instruction_list, instr_cfg.InstructionListProto)
        assert isinstance(eager_symbol_list, eager_symbol_pb.EagerSymbolList)
        self.instruction_list_ = instruction_list
        self.eager_symbol_list_ = eager_symbol_list

    def StatelessCall(self, op_attribute, parallel_conf, bn_in_op2blob_object={}):
        op_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
        self._CheckRefInBlobObjectParallelDesc(
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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

    def NoBoxingStatelessCall(
        self, op_attribute, parallel_conf, bn_in_op2blob_object={}
    ):
        op_parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf)
        self._CheckRefInBlobObjectParallelDesc(
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
        )

        def FetchDelegateBlobObject(blob_object, op_arg_parallel_attr):
            from_pd = blob_object.parallel_desc_symbol
            to_pd = op_arg_parallel_attr.parallel_desc_symbol
            if from_pd == to_pd:
                return blob_object
            assert from_pd.device_tag == "cpu"
            assert to_pd.device_tag == "cpu"
            assert from_pd.parallel_num == to_pd.parallel_num
            from_machine_ids = from_pd.machine_id2device_id_list.keys()
            to_machine_ids = to_pd.machine_id2device_id_list.keys()
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
            op_attribute,
            blob_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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

    def DeleteObject(self, obj):
        self._TryClearObject(obj)
        self._DeleteObject(obj)

    def InsertRemoveForeignCallbackInstruction(self, object_id, callback):
        unique_callback_id = python_callback.GetIdForRegisteredCallback(callback)
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("RemoveForeignCallback")
        instruction.mutable_operand().Add().CopyFrom(_DelObjectOperand(object_id))
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(unique_callback_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def FetchBlobHeader(self, blob_object, callback):
        return self._FetchBlob("FetchBlobHeader", blob_object, callback)

    def FetchBlobBody(self, blob_object, callback):
        return self._FetchBlob("FetchBlobBody", blob_object, callback)

    def PackPhysicalBlobsToLogicalBlob(
        self, physical_blob_objects, op_arg_parallel_attr, op_arg_blob_attr
    ):
        parallel_desc_symbol = op_arg_parallel_attr.parallel_desc_symbol
        machine_id2device_ids = parallel_desc_symbol.machine_id2device_id_list
        device_tag = parallel_desc_symbol.parallel_conf.device_tag
        machine_device_ids = set()
        for physical_blob_object in physical_blob_objects:
            phy_paralle_desc_sym = physical_blob_object.parallel_desc_symbol
            assert (
                phy_paralle_desc_sym.parallel_num == 1
            ), phy_paralle_desc_sym.parallel_num
            assert phy_paralle_desc_sym.device_tag == device_tag, "%s v.s. %s" % (
                phy_paralle_desc_sym.device_tag,
                device_tag,
            )
            phy_machine_id2device_ids = phy_paralle_desc_sym.machine_id2device_id_list
            machine_id = list(phy_machine_id2device_ids.keys())[0]
            pair = (machine_id, phy_machine_id2device_ids[machine_id][0])
            machine_device_ids.add(pair)

        for machine_id, device_ids in machine_id2device_ids.items():
            for device_id in device_ids:
                assert (machine_id, device_id) in machine_device_ids, "%s not in %s" % (
                    (machine_id, device_id),
                    machine_device_ids,
                )
        logical_blob_object = self._NewBlobObject(
            op_arg_parallel_attr, op_arg_blob_attr
        )
        self._ReplaceMirrored(
            op_arg_parallel_attr.parallel_desc_symbol,
            [logical_blob_object],
            physical_blob_objects,
        )
        return logical_blob_object

    def GetPhysicalParallelDescSymbols(self, parallel_desc_symbol):
        machine_id2device_ids = parallel_desc_symbol.machine_id2device_id_list
        device_tag = parallel_desc_symbol.parallel_conf.device_tag
        phy_parallel_desc_symbols = []

        def AppendPhyParallelDescSymbol(machine_id, device_id):
            parallel_conf = placement_pb.ParallelConf()
            parallel_conf.device_tag = device_tag
            parallel_conf.device_name.append("%d:%d" % (machine_id, device_id))
            phy_parallel_desc_symbols.append(self.GetParallelDescSymbol(parallel_conf))

        for machine_id, device_ids in machine_id2device_ids.items():
            for device_id in device_ids:
                AppendPhyParallelDescSymbol(machine_id, device_id)
        return phy_parallel_desc_symbols

    def _GetPhysicalOpArgBlobAttrs(self, logical_blob_object):
        parallel_num = logical_blob_object.parallel_desc_symbol.parallel_num
        logical_blob_attr = logical_blob_object.op_arg_blob_attr
        sbp_parallel = logical_blob_object.op_arg_parallel_attr.sbp_parallel

        def GetSplittedBlobAttr(
            logical_blob_attr, split_axis, parallel_num, parallel_id
        ):
            blob_desc = blob_desc_pb.BlobDescProto()
            blob_desc.CopyFrom(logical_blob_attr.blob_desc)
            physical_len = balanced_splitter.BalancedPartNums(
                logical_blob_attr.shape[split_axis], parallel_num
            )[parallel_id]
            blob_desc.body.shape.dim[split_axis] = physical_len
            physical_blob_attr = op_arg_util.OpArgBlobAttribute(
                logical_blob_attr.batch_axis,
                blob_desc,
                logical_blob_attr.logical_blob_name,
            )
            return physical_blob_attr

        if sbp_parallel.HasField("split_parallel"):
            split_axis = sbp_parallel.split_parallel.axis

            return [
                GetSplittedBlobAttr(logical_blob_attr, split_axis, parallel_num, i)
                for i in range(parallel_num)
            ]
        else:
            return [logical_blob_attr] * parallel_num

    def UnpackLogicalBlobToPhysicalBlobs(self, blob_object):
        phy_parallel_desc_symbols = self.GetPhysicalParallelDescSymbols(
            blob_object.parallel_desc_symbol
        )
        phy_op_arg_blob_attrs = self._GetPhysicalOpArgBlobAttrs(blob_object)

        def GetPhysicalBlob(parallel_desc_sym, blob_attr):
            op_arg_parallel_attr = op_arg_util.MakeMirroredOpArgParallelAttribute(
                parallel_desc_sym
            )
            pyhsical_blob_object = self._NewBlobObject(op_arg_parallel_attr, blob_attr)
            return pyhsical_blob_object

        physical_blob_objects = [
            GetPhysicalBlob(phy_parallel_desc_symbols[i], phy_op_arg_blob_attrs[i])
            for i in range(len(phy_parallel_desc_symbols))
        ]
        self._ReplaceMirrored(
            blob_object.parallel_desc_symbol, physical_blob_objects, [blob_object]
        )
        return physical_blob_objects

    def MakeReferenceBlobObject(self, blob_object, op_arg_parallel_attr):
        parallel_desc_symbol = blob_object.parallel_desc_symbol
        assert parallel_desc_symbol == op_arg_parallel_attr.parallel_desc_symbol
        ref_blob_object = self._NewBlobObject(
            op_arg_parallel_attr, blob_object.op_arg_blob_attr
        )
        self._ReplaceMirrored(parallel_desc_symbol, [ref_blob_object], [blob_object])
        return ref_blob_object

    def MakeLazyRefBlobObject(self, interface_op_name):
        sess = session_ctx.GetDefaultSession()
        op_attribute = sess.OpAttribute4InterfaceOpName(interface_op_name)
        assert len(op_attribute.output_bns) == 1
        obn = op_attribute.output_bns[0]

        blob_parallel_desc_sym_id = op_attribute.parallel_signature.bn_in_op2parallel_desc_symbol_id[
            obn
        ]
        blob_parallel_desc_sym = symbol_storage.GetSymbol4Id(blob_parallel_desc_sym_id)
        op_arg_parallel_attr = op_arg_util.GetOpArgParallelAttribute(
            blob_parallel_desc_sym, op_attribute, obn
        )
        op_arg_blob_attr = op_arg_util.GetOpArgBlobAttribute(op_attribute, obn)

        blob_object = self._NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr)
        self._LazyReference(blob_object, interface_op_name)
        return blob_object

    def GetSymbol4String(self, string):
        if symbol_storage.HasSymbol4String(string):
            return symbol_storage.GetSymbol4String(string)
        symbol_id = self._NewSymbolId4String(string)
        symbol = symbol_util.Symbol(symbol_id, string)
        symbol_storage.SetSymbol4Id(symbol_id, symbol)
        symbol_storage.SetSymbol4String(string, symbol)
        return symbol

    def GetJobConfSymbol(self, job_conf):
        if symbol_storage.HasSymbol4JobConf(job_conf):
            return symbol_storage.GetSymbol4JobConf(job_conf)
        symbol_id = self._NewSymbolId4JobConf(job_conf)
        symbol = symbol_util.Symbol(symbol_id, job_conf)
        symbol_storage.SetSymbol4Id(symbol_id, symbol)
        symbol_storage.SetSymbol4JobConf(job_conf, symbol)
        return symbol

    def GetParallelDescSymbol(self, parallel_conf):
        serialized_parallel_conf = parallel_conf.SerializeToString()
        if symbol_storage.HasSymbol4SerializedParallelConf(serialized_parallel_conf):
            return symbol_storage.GetSymbol4SerializedParallelConf(
                serialized_parallel_conf
            )
        symbol_id = self._NewSymbolId4ParallelConf(parallel_conf)
        symbol = symbol_util.ParallelDescSymbol(symbol_id, parallel_conf)
        symbol_storage.SetSymbol4Id(symbol_id, symbol)
        symbol_storage.SetSymbol4SerializedParallelConf(
            serialized_parallel_conf, symbol
        )
        return symbol

    def GetScopeSymbol(self, scope_proto, parent_scope_symbol=None):
        symbol_id = self._NewSymbolId4Scope(scope_proto)
        serialized_scope_proto = scope_proto.SerializeToString()
        if symbol_storage.HasSymbol4SerializedScopeProto(serialized_scope_proto):
            return symbol_storage.GetSymbol4SerializedScopeProto(serialized_scope_proto)
        symbol = scope_symbol.ScopeSymbol(symbol_id, scope_proto, parent_scope_symbol)
        symbol_storage.SetSymbol4Id(symbol_id, symbol)
        symbol_storage.SetSymbol4SerializedScopeProto(serialized_scope_proto, symbol)
        return symbol

    def GetSharedOpKernelObject4ParallelConfSymbol(self, parallel_desc_sym):
        if object_storage.HasSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym):
            return object_storage.GetSharedOpKernelObject4ParallelConfSymbol(
                parallel_desc_sym
            )
        object_id = self._NewSharedOpKernelObjectId4ParallelConfSymbolId(
            parallel_desc_sym
        )
        obj = object_util.Object(object_id, parallel_desc_sym)
        object_storage.SetSharedOpKernelObject4ParallelConfSymbol(
            parallel_desc_sym, obj
        )
        return obj

    @contextmanager
    def CudaHostPinBlob(self, blob_object):
        self._CudaHostRegisterBlob(blob_object)
        try:
            yield
        finally:
            self._CudaHostUnregisterBlob(blob_object)

    def BroadcastBlobReference(self, sole_mirrored_blob_object, parallel_desc_sym):
        device_ids = (
            sole_mirrored_blob_object.parallel_desc_symbol.machine_id2device_id_list
        )
        for _, dev_ids in device_ids.items():
            assert len(dev_ids) == 1, "dev_ids: %s" % dev_ids
        object_id = self._BroadcastObjectReference(
            sole_mirrored_blob_object, parallel_desc_sym
        )
        op_arg_parallel_attr = op_arg_util.MakeBroadcastOpArgParallelAttribute(
            parallel_desc_sym
        )
        return object_util.BlobObject(
            object_id=object_id,
            op_arg_parallel_attr=op_arg_parallel_attr,
            op_arg_blob_attr=sole_mirrored_blob_object.op_arg_blob_attr,
            release=self.release_object_,
        )

    def NewOpKernelObject(self, op_conf):
        assert op_conf.HasField("scope_symbol_id")
        scope_symbol = symbol_storage.GetSymbol4Id(op_conf.scope_symbol_id)
        op_conf_sym = self._GetOpConfSymbol(op_conf)
        parallel_desc_sym_id = c_api_util.GetOpParallelSymbolId(op_conf)
        parallel_desc_symbol = symbol_storage.GetSymbol4Id(parallel_desc_sym_id)
        object_id = self._NewOpKernelObject(
            parallel_desc_symbol, scope_symbol.job_desc_symbol, op_conf_sym
        )
        return OpKernelObject(object_id, op_conf, self.release_object_)

    def Build121To(self, blob_object, parallel_desc_symbol):
        ref_blob_object = _MakeNewBlobObjectLike(
            self, blob_object, parallel_desc_symbol
        )
        self.Build121AssignInstruction(ref_blob_object, blob_object)
        return ref_blob_object

    def Build121AssignInstruction(self, ref_blob_object, value_blob_object):
        parallel_num = ref_blob_object.parallel_desc_symbol.parallel_num
        assert parallel_num == value_blob_object.parallel_desc_symbol.parallel_num
        token_ids = (
            [oneflow_api.NewTokenId() for _ in range(parallel_num)],
            [oneflow_api.NewTokenId() for _ in range(parallel_num)],
        )
        self._BuildSendInstruction(
            ref_blob_object.parallel_desc_symbol, value_blob_object, token_ids
        )
        self._BuildRecvInstruction(
            value_blob_object.parallel_desc_symbol, ref_blob_object, token_ids
        )

    def _BuildSendInstruction(
        self, dst_parallel_desc_symbol, src_blob_object, token_ids
    ):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("SendBlob")
        instruction.set_parallel_desc_symbol_id(
            src_blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(dst_parallel_desc_symbol.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _ConstOperand(src_blob_object.object_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for token_id in token_ids[0]:
            instruction.mutable_operand().Add().CopyFrom(_Uint64Operand(token_id))
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for token_id in token_ids[1]:
            instruction.mutable_operand().Add().CopyFrom(_Uint64Operand(token_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _BuildRecvInstruction(
        self, src_parallel_desc_symbol, dst_blob_object, token_ids
    ):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("ReceiveBlob")
        instruction.set_parallel_desc_symbol_id(
            dst_blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(src_parallel_desc_symbol.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _Mut2Operand(dst_blob_object.object_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for token_id in token_ids[0]:
            instruction.mutable_operand().Add().CopyFrom(_Uint64Operand(token_id))
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for token_id in token_ids[1]:
            instruction.mutable_operand().Add().CopyFrom(_Uint64Operand(token_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _NewOpKernelObject(self, parallel_desc_symbol, job_desc_sym, op_conf_sym):
        object_id = self._NewObjectId(parallel_desc_symbol)
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("InitOpKernelObject")
        instruction.set_parallel_desc_symbol_id(parallel_desc_symbol.symbol_id)
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(job_desc_sym.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(op_conf_sym.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_MutOperand(object_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
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
            op_parallel_desc_sym = symbol_storage.GetSymbol4Id(symbol_id)
        assert op_parallel_desc_sym is not None

        def DelegateBlobObject4Ibn(ibn):
            op_arg_parallel_attr = op_arg_util.GetOpArgParallelAttribute(
                op_parallel_desc_sym, op_attribute, ibn
            )
            return get_delegate_blob_object(
                bn_in_op2blob_object[ibn], op_arg_parallel_attr
            )

        op_conf = op_attribute.op_conf
        assert op_conf.HasField("scope_symbol_id"), op_conf
        scope_symbol = symbol_storage.GetSymbol4Id(op_conf.scope_symbol_id)
        job_desc_sym = scope_symbol.job_desc_symbol
        op_conf_sym = self._GetOpConfSymbol(op_conf)
        op_node_signature_sym = self._GetOpNodeSignatureSymbol(op_attribute)
        opkernel_obj = self.GetSharedOpKernelObject4ParallelConfSymbol(
            op_parallel_desc_sym
        )
        const_input_operand_blob_objects = self._GetConstInputOperandBlobObjects(
            op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
        )
        mutable_input_operand_blob_objects = self._GetMutableInputOperandBlobObjects(
            op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
        )
        mut1_operand_blob_objects = self._GetMut1OperandBlobObjects(
            op_attribute,
            blob_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
        )
        mut2_operand_blob_objects = self._GetMut2OperandBlobObjects(
            op_attribute,
            blob_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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
        self,
        op_attribute,
        opkernel_object,
        bn_in_op2blob_object,
        get_delegate_blob_object,
    ):
        op_parallel_desc_sym = opkernel_object.parallel_desc_symbol

        def DelegateBlobObject4Ibn(ibn):
            op_arg_parallel_attr = op_arg_util.GetOpArgParallelAttribute(
                op_parallel_desc_sym, op_attribute, ibn
            )
            return get_delegate_blob_object(
                bn_in_op2blob_object[ibn], op_arg_parallel_attr
            )

        op_node_signature_sym = self._GetOpNodeSignatureSymbol(op_attribute)
        const_input_operand_blob_objects = self._GetConstInputOperandBlobObjects(
            op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
        )
        mutable_input_operand_blob_objects = self._GetMutableInputOperandBlobObjects(
            op_attribute, blob_object4ibn=DelegateBlobObject4Ibn
        )
        mut1_operand_blob_objects = self._GetMut1OperandBlobObjects(
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
        )
        mut2_operand_blob_objects = self._GetMut2OperandBlobObjects(
            op_attribute,
            op_parallel_desc_sym,
            bn_in_op2blob_object=bn_in_op2blob_object,
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

    def _CudaHostRegisterBlob(self, blob_object):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("CudaHostRegisterBlob")
        instruction.set_parallel_desc_symbol_id(
            blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(_MutOperand(blob_object.object_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _CudaHostUnregisterBlob(self, blob_object):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("CudaHostUnregisterBlob")
        instruction.set_parallel_desc_symbol_id(
            blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(_MutOperand(blob_object.object_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

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
        new_op_node_signature = op_attribute_pb.OpNodeSignature()
        new_op_node_signature.sbp_signature.CopyFrom(op_attribute.sbp_signature)
        new_op_node_signature.mirrored_signature.CopyFrom(
            op_attribute.mirrored_signature
        )
        new_op_node_signature.logical_blob_desc_signature.CopyFrom(
            op_attribute.logical_blob_desc_signature
        )
        new_op_node_signature.batch_axis_signature.CopyFrom(
            op_attribute.batch_axis_signature
        )
        new_op_node_signature.parallel_signature.CopyFrom(
            op_attribute.parallel_signature
        )
        serialized_op_node_signature = new_op_node_signature.SerializeToString()
        if symbol_storage.HasSymbol4SerializedOpNodeSignature(
            serialized_op_node_signature
        ):
            return symbol_storage.GetSymbol4SerializedOpNodeSignature(
                serialized_op_node_signature
            )
        symbol_id = self._NewSymbolId4OpNodeSignature(new_op_node_signature)
        symbol = symbol_util.Symbol(symbol_id, new_op_node_signature)
        symbol_storage.SetSymbol4Id(symbol_id, symbol)
        symbol_storage.SetSymbol4SerializedOpNodeSignature(
            serialized_op_node_signature, symbol
        )
        return symbol

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
                return symbol_storage.GetSymbol4Id(bn2symbol_id[obn])
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
            op_arg_parallel_attr = op_arg_util.GetOpArgParallelAttribute(
                GetOutBlobParallelDescSymbol(obn), op_attribute, obn
            )
            op_arg_blob_attr = op_arg_util.GetOpArgBlobAttribute(op_attribute, obn)
            out_blob_object = self._NewBlobObject(
                op_arg_parallel_attr, op_arg_blob_attr
            )
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
                return symbol_storage.GetSymbol4Id(bn2symbol_id[obn])
            else:
                return parallel_desc_sym

        for obn in op_attribute.output_bns:
            obn2modifier = op_attribute.arg_modifier_signature.obn2output_blob_modifier
            if obn2modifier[obn].header_infered_before_compute:
                continue
            obn_sym = self.GetSymbol4String(obn)
            op_arg_parallel_attr = op_arg_util.GetOpArgParallelAttribute(
                GetOutBlobParallelDescSymbol(obn), op_attribute, obn
            )
            op_arg_blob_attr = op_arg_util.GetOpArgBlobAttribute(op_attribute, obn)
            out_blob_object = self._NewBlobObject(
                op_arg_parallel_attr, op_arg_blob_attr
            )
            bn_in_op2blob_object[obn] = out_blob_object
            mut2_operand_blob_objects.append((obn_sym, out_blob_object))
        return mut2_operand_blob_objects

    def _NewBlobObject(self, op_arg_parallel_attr, op_arg_blob_attr):
        object_id = self._NewObjectId(op_arg_parallel_attr.parallel_desc_symbol)
        return object_util.BlobObject(
            object_id=object_id,
            op_arg_parallel_attr=op_arg_parallel_attr,
            op_arg_blob_attr=op_arg_blob_attr,
            release=self.release_object_,
        )

    def _NewSymbolId4String(self, string):
        symbol_id = self._NewSymbolId()
        self._InitStringSymbol(symbol_id, string)
        return symbol_id

    def _NewSymbolId4ParallelConf(self, parallel_conf):
        symbol_id = self.id_generator_.NewSymbolId()
        self._NewParallelConfSymbol(symbol_id, parallel_conf)
        return symbol_id

    def _NewSymbolId4Scope(self, scope_proto):
        symbol_id = self._NewSymbolId()
        scope_proto.symbol_id = symbol_id
        self._NewScopeSymbol(scope_proto)
        return symbol_id

    def _NewSymbolId4JobConf(self, job_conf):
        symbol_id = self._NewSymbolId()
        self._InitJobConfSymbol(symbol_id, job_conf)
        return symbol_id

    def _NewSymbolId4OpConf(self, op_conf):
        symbol_id = self._NewSymbolId()
        self._InitOpConfSymbol(symbol_id, op_conf)
        return symbol_id

    def _NewSymbolId4OpNodeSignature(self, op_node_signature):
        symbol_id = self._NewSymbolId()
        self._InitOpNodeSignatureDescSymbol(symbol_id, op_node_signature)
        return symbol_id

    def _NewSharedOpKernelObjectId4ParallelConfSymbolId(self, parallel_desc_sym):
        return self._NewObjectId(parallel_desc_sym)

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
            _SymbolOperand(job_desc_sym.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(op_conf_sym.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(op_node_signature_sym.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _MutOperand(shared_opkernel_obj.object_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for ibn_sym, _ in const_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(ibn_sym.symbol_id)
            )
        for _, blob_object in const_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _ConstOperand(blob_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for ibn_sym, _ in mutable_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(ibn_sym.symbol_id)
            )
        for _, blob_object in mutable_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _MutOperand(blob_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for obn_sym, _ in mut1_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(obn_sym.symbol_id)
            )
        for _, blob_object in mut1_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _MutOperand(blob_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for obn_sym, _ in mut2_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(obn_sym.symbol_id)
            )
        for _, blob_object in mut2_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _Mut2Operand(blob_object.object_id)
            )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

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
            _MutOperand(opkernel_object.object_id)
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(op_node_signature_sym.symbol_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for ibn_sym, _ in const_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(ibn_sym.symbol_id)
            )
        for _, blob_object in const_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _ConstOperand(blob_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for ibn_sym, _ in mutable_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(ibn_sym.symbol_id)
            )
        for _, blob_object in mutable_input_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _MutOperand(blob_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for obn_sym, _ in mut1_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(obn_sym.symbol_id)
            )
        for _, blob_object in mut1_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _MutOperand(blob_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for obn_sym, _ in mut2_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _SymbolOperand(obn_sym.symbol_id)
            )
        for _, blob_object in mut2_operand_blob_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _Mut2Operand(blob_object.object_id)
            )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _NewSymbolId(self):
        symbol_id = self.id_generator_.NewSymbolId()
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("NewSymbol")
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(symbol_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        return symbol_id

    def _NewObjectId(self, parallel_desc_sym):
        object_id = self.id_generator_.NewObjectId()
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("NewObject")
        instruction.set_parallel_desc_symbol_id(parallel_desc_sym.symbol_id)
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(object_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        return object_id

    def _LazyReference(self, blob_object, interface_op_name):
        instruction = instr_cfg.InstructionProto()
        device_tag = blob_object.parallel_desc_symbol.device_tag
        instruction.set_instr_type_name("{}.LazyReference".format(device_tag))
        instruction.set_parallel_desc_symbol_id(
            blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(_MutOperand(blob_object.object_id))
        interface_op_name_sym = self.GetSymbol4String(
            blob_object.op_arg_blob_attr.logical_blob_name
        )
        instruction.mutable_operand().Add().CopyFrom(
            _SymbolOperand(interface_op_name_sym.symbol_id)
        )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _BroadcastObjectReference(self, sole_mirrored_object, parallel_desc_sym):
        object_id = self.id_generator_.NewObjectId()
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("BroadcastObjectReference")
        instruction.set_parallel_desc_symbol_id(parallel_desc_sym.symbol_id)
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(object_id))
        instruction.mutable_operand().Add().CopyFrom(
            _Int64Operand(sole_mirrored_object.object_id)
        )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        return object_id

    def _InitStringSymbol(self, symbol_id, string):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("InitStringSymbol")
        instruction.mutable_operand().Add().CopyFrom(_InitSymbolOperand(symbol_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        eager_symbol = eager_symbol_pb.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.string_symbol = string
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _NewParallelConfSymbol(self, symbol_id, parallel_conf):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("NewParallelDescSymbol")
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(symbol_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        eager_symbol = eager_symbol_pb.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.parallel_conf_symbol.CopyFrom(parallel_conf)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _NewScopeSymbol(self, scope_proto):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("InitScopeSymbol")
        instruction.mutable_operand().Add().CopyFrom(
            _InitSymbolOperand(scope_proto.symbol_id)
        )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        eager_symbol = eager_symbol_pb.EagerSymbol()
        eager_symbol.symbol_id = scope_proto.symbol_id
        eager_symbol.scope_symbol.CopyFrom(scope_proto)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _InitJobConfSymbol(self, symbol_id, job_conf):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("InitJobDescSymbol")
        instruction.mutable_operand().Add().CopyFrom(_InitSymbolOperand(symbol_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        eager_symbol = eager_symbol_pb.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.job_conf_symbol.CopyFrom(
            text_format.Parse(str(job_conf), job_conf_pb.JobConfigProto())
        )
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _InitOpConfSymbol(self, symbol_id, op_conf):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("InitOperatorConfSymbol")
        instruction.mutable_operand().Add().CopyFrom(_InitSymbolOperand(symbol_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        eager_symbol = eager_symbol_pb.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.op_conf_symbol.CopyFrom(op_conf)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _InitOpNodeSignatureDescSymbol(self, symbol_id, op_node_signature):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("InitOpNodeSignatureDescSymbol")
        instruction.mutable_operand().Add().CopyFrom(_InitSymbolOperand(symbol_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)
        eager_symbol = eager_symbol_pb.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.op_node_signature_symbol.CopyFrom(op_node_signature)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _FetchBlob(self, instruction_name, blob_object, fetcher):
        unique_callback_id = python_callback.GetIdForRegisteredCallback(fetcher)
        instruction = instr_cfg.InstructionProto()
        device_tag = blob_object.parallel_desc_symbol.device_tag
        instruction.set_instr_type_name("%s.%s" % (device_tag, instruction_name))
        instruction.set_parallel_desc_symbol_id(
            blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(
            _ConstOperand(blob_object.object_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(unique_callback_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def FeedBlob(self, blob_object, feeder):
        unique_callback_id = python_callback.GetIdForRegisteredCallback(feeder)
        instruction = instr_cfg.InstructionProto()
        device_tag = blob_object.parallel_desc_symbol.device_tag
        instruction.set_instr_type_name("%s.%s" % (device_tag, "FeedBlob"))
        instruction.set_parallel_desc_symbol_id(
            blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(
            _Mut2Operand(blob_object.object_id)
        )
        instruction.mutable_operand().Add().CopyFrom(_Int64Operand(unique_callback_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _TryClearObject(self, obj):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("TryClearObject")
        instruction.set_parallel_desc_symbol_id(obj.parallel_desc_symbol.symbol_id)
        instruction.mutable_operand().Add().CopyFrom(_MutOperand(obj.object_id))
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _DeleteObject(self, blob_object):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("DeleteObject")
        instruction.set_parallel_desc_symbol_id(
            blob_object.parallel_desc_symbol.symbol_id
        )
        instruction.mutable_operand().Add().CopyFrom(
            _DelObjectOperand(blob_object.object_id)
        )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)

    def _ReplaceMirrored(self, parallel_desc_sym, lhs_objects, rhs_objects):
        instruction = instr_cfg.InstructionProto()
        instruction.set_instr_type_name("ReplaceMirrored")
        instruction.set_parallel_desc_symbol_id(parallel_desc_sym.symbol_id)
        for lhs_object in lhs_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _Int64Operand(lhs_object.object_id)
            )
        instruction.mutable_operand().Add().CopyFrom(_OperandSeparator())
        for rhs_object in rhs_objects:
            instruction.mutable_operand().Add().CopyFrom(
                _Int64Operand(rhs_object.object_id)
            )
        self.instruction_list_.mutable_instruction().Add().CopyFrom(instruction)


def _MakeNewBlobObjectLike(builder, blob_object, new_parallel_desc_symbol):
    op_conf = op_conf_pb.OperatorConf()
    op_conf.name = id_util.UniqueStr("Input")
    op_conf.device_tag = new_parallel_desc_symbol.device_tag
    op_conf.input_conf.out = "out"
    blob_object.op_arg_parallel_attr.DumpToToInterfaceBlobConf(
        op_conf.input_conf.blob_conf
    )
    blob_object.op_arg_blob_attr.DumpToToInterfaceBlobConf(op_conf.input_conf.blob_conf)
    op_conf.scope_symbol_id = oneflow.current_scope().symbol_id
    upstream_signature = op_attribute_pb.OpNodeSignature()
    op_attribute = c_api_util.InferOpConf(op_conf, upstream_signature)
    parallel_conf = new_parallel_desc_symbol.parallel_conf
    bn_in_op2blob_object = {}
    builder.RawStatelessCall(
        op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
    )
    return bn_in_op2blob_object["out"]


def _SymbolOperand(val):
    operand = instr_cfg.InstructionOperandProto()
    _SetSoleMirroredOperand(operand.mutable_symbol_operand(), val)
    return operand


def _InitSymbolOperand(val):
    operand = instr_cfg.InstructionOperandProto()
    _SetSoleMirroredOperand(operand.mutable_init_symbol_operand(), val)
    return operand


def _ConstOperand(val):
    operand = instr_cfg.InstructionOperandProto()
    _SetMirroredOperand(operand.mutable_const_operand(), val)
    return operand


def _MutOperand(val):
    operand = instr_cfg.InstructionOperandProto()
    _SetMirroredOperand(operand.mutable_mut_operand(), val)
    return operand


def _Mut2Operand(val):
    operand = instr_cfg.InstructionOperandProto()
    _SetMirroredOperand(operand.mutable_mut2_operand(), val)
    return operand


def _DelObjectOperand(val):
    operand = instr_cfg.InstructionOperandProto()
    _SetAllMirroredOperand(operand.mutable_mut_operand(), val)
    return operand


def _Int64Operand(val):
    operand = instr_cfg.InstructionOperandProto()
    operand.set_int64_operand(val)
    return operand


def _Uint64Operand(val):
    operand = instr_cfg.InstructionOperandProto()
    operand.set_uint64_operand(val)
    return operand


def _OperandSeparator():
    operand = instr_cfg.InstructionOperandProto()
    operand.mutable_separator()
    return operand


def _SetMirroredOperand(operand, val):
    operand.set_logical_object_id(val)
    operand.mutable_current_global_device_id()


def _SetSoleMirroredOperand(operand, val):
    operand.set_logical_object_id(val)
    operand.mutable_sole_mirrored_object()


def _SetAllMirroredOperand(operand, val):
    operand.set_logical_object_id(val)
    operand.mutable_all_mirrored_object()


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


def _ReleaseLogicalObject(obj):
    LogicalRun(lambda builder: builder.DeleteObject(obj))


def _ReleasePhysicalObject(obj):
    PhysicalRun(lambda builder: builder.DeleteObject(obj))
