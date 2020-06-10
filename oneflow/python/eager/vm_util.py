from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
import oneflow.python.vm.id_util as vm_id_util
import oneflow.core.vm.instruction_pb2 as instr_util
import oneflow.core.job.placement_pb2 as placement_pb_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.eager.job_conf_ctx as job_conf_ctx
import oneflow.python.eager.symbol as symbol_util
import oneflow.python.eager.symbol_cache as symbol_cache
import oneflow.python.eager.object as object_util
import oneflow.python.eager.object_cache as object_cache
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.physical_blob_callback as physical_blob_callback
import oneflow.python.eager.boxing_util as boxing_util
import re
import oneflow

from contextlib import contextmanager

def PhysicalRun(build):
    return _Run(build, vm_id_util.PhysicalIdGenerator(), c_api_util.RunPhysicalInstruction,
                _ReleasePhysicalBlobObject)

def LogicalRun(build):
    return _Run(build, vm_id_util.LogicalIdGenerator(), c_api_util.RunLogicalInstruction,
                _ReleaseLogicalBlobObject)

def _Run(build, id_generator, run_api, release_blob_object):
    instruction_list = session_ctx.GetDefaultSession().instruction_list
    eager_symbol_list = session_ctx.GetDefaultSession().eager_symbol_list
    build(InstructionsBuilder(id_generator, release_blob_object,
                              instruction_list, eager_symbol_list))
    run_api(instruction_list, eager_symbol_list)
    instruction_list.ClearField("instruction")
    eager_symbol_list.ClearField("eager_symbol")


def _DefaultBlobObject4Ibn(ibn):
    raise NotImplementedError


class InstructionsBuilder(object):
    def __init__(self, id_generator, release_blob_object, instruction_list, eager_symbol_list):
        self.id_generator_ = id_generator
        self.release_blob_object_ = release_blob_object
        assert isinstance(instruction_list, instr_util.InstructionListProto)
        assert isinstance(eager_symbol_list, eager_symbol_util.EagerSymbolList)
        self.instruction_list_ = instruction_list
        self.eager_symbol_list_ = eager_symbol_list

    def StatelessCall(self, op_conf):
        return self._StatelessCall("compute", op_conf)

    def _StatelessCall(self, stream_tag, op_conf):
        assert op_conf.HasField('user_conf')
        placement_scope = oneflow.placement.current_scope()
        parallel_conf = placement_scope.default_parallel_conf
        device_tag = placement_scope.default_device_tag
        parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf, device_tag)
        job_conf_sym = self.GetJobConfSymbol(job_conf_ctx.CurrentJobConf())
        op_conf_sym = self._GetOpConfSymbol(op_conf)
        opkernel_obj = self.GetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym)
        const_operand_blob_objects = self._GetConstOperandBlobObjects(op_conf, parallel_desc_sym)
        mut1_operand_blob_objects = self._GetMut1OperandBlobObjects(op_conf, parallel_desc_sym)
        mut2_operand_blob_objects = self._GetMut2OperandBlobObjects(op_conf, parallel_desc_sym)
        return self._StatelessCallOpKernel("%s.StatelessCallOpKernel" % stream_tag,
                parallel_desc_sym, job_conf_sym, op_conf_sym, opkernel_obj,
                const_operand_blob_objects, mut1_operand_blob_objects, mut2_operand_blob_objects)

    def SystemStatelessCall(self, op_conf, parallel_conf = None, device_tag = None,
                                const_arg_bns=[], mut_arg_bns=[], mut2_arg_bns=[],
                                bn_in_op2blob_object={}):
        system_op_conf = getattr(op_conf, op_conf.WhichOneof('op_type'))
        placement_scope = oneflow.placement.current_scope()
        if parallel_conf is None: parallel_conf = placement_scope.default_parallel_conf
        if device_tag is None: device_tag = placement_scope.default_device_tag
        def GetDelegateBlobObject(blob_object, op_parallel_desc_symbol):
            return _FindOrCreateDelegateBlobObject(self, blob_object, op_parallel_desc_symbol)
        self._SystemStatelessCall("compute", op_conf, parallel_conf, device_tag,
                const_arg_bns=const_arg_bns, mut_arg_bns=mut_arg_bns, mut2_arg_bns=mut2_arg_bns,
                bn_in_op2blob_object=bn_in_op2blob_object,
                get_delegate_blob_object=GetDelegateBlobObject)

    def SystemCudaD2HStatelessCall(self, op_conf, in_parallel_conf,
                                       const_arg_bns=[], mut_arg_bns=[], mut2_arg_bns=[],
                                       bn_in_op2blob_object={}):
        def GetDirectBlobObject(blob_object, op_parallel_desc_symbol): return blob_object
        self._SystemStatelessCall("copy_d2h", op_conf, in_parallel_conf, "gpu",
                const_arg_bns=const_arg_bns, mut_arg_bns=mut_arg_bns, mut2_arg_bns=mut2_arg_bns,
                bn_in_op2blob_object=bn_in_op2blob_object,
                get_delegate_blob_object=GetDirectBlobObject)

    def SystemCudaH2DStatelessCall(self, op_conf, out_parallel_conf,
                                       const_arg_bns=[], mut_arg_bns=[], mut2_arg_bns=[],
                                       bn_in_op2blob_object={}):
        def GetDirectBlobObject(blob_object, op_parallel_desc_symbol): return blob_object
        self._SystemStatelessCall("copy_h2d", op_conf, out_parallel_conf, "gpu",
                const_arg_bns=const_arg_bns, mut_arg_bns=mut_arg_bns, mut2_arg_bns=mut2_arg_bns,
                bn_in_op2blob_object=bn_in_op2blob_object,
                get_delegate_blob_object=GetDirectBlobObject)

    def _SystemStatelessCall(self, stream_tag, op_conf, op_parallel_conf, op_device_tag,
                                 const_arg_bns=[], mut_arg_bns=[], mut2_arg_bns=[],
                                 bn_in_op2blob_object={}, get_delegate_blob_object=None):
        def BlobObject4Ibn(ibn): return bn_in_op2blob_object[ibn]
        assert callable(get_delegate_blob_object)
        assert isinstance(const_arg_bns, (list, tuple))
        assert isinstance(mut_arg_bns, (list, tuple))
        assert isinstance(mut2_arg_bns, (list, tuple))
        assert len(const_arg_bns) + len(mut_arg_bns) + len(mut2_arg_bns) > 0
        opkernel_parallel_desc_sym = self.GetParallelDescSymbol(op_parallel_conf, op_device_tag)
        placement_scope = oneflow.placement.current_scope()
        blob_parallel_desc_sym = self.GetParallelDescSymbol(
                placement_scope.default_parallel_conf, placement_scope.default_device_tag)
        job_conf_sym = self.GetJobConfSymbol(job_conf_ctx.CurrentJobConf())
        op_conf_sym = self._SystemGetOpConfSymbol(op_conf)
        opkernel_obj = self.GetSharedOpKernelObject4ParallelConfSymbol(opkernel_parallel_desc_sym)
        def DelegateBlobObject4Ibn(ibn):
            return get_delegate_blob_object(BlobObject4Ibn(ibn), opkernel_parallel_desc_sym)
        const_operand_blob_objects = self._SystemGetConstOperandBlobObjects(op_conf, const_arg_bns,
                blob_object4ibn=DelegateBlobObject4Ibn)
        mut1_operand_blob_objects = self._SystemGetMut1OperandBlobObjects(
                op_conf, mut_arg_bns, blob_parallel_desc_sym, blob_object4ibn=BlobObject4Ibn)
        mut2_operand_blob_objects = self._SystemGetMut2OperandBlobObjects(
                op_conf, mut2_arg_bns, blob_parallel_desc_sym, blob_object4ibn=BlobObject4Ibn)
        self._StatelessCallOpKernel("%s.SystemStatelessCallOpKernel" % stream_tag,
                opkernel_parallel_desc_sym, job_conf_sym, op_conf_sym, opkernel_obj,
                const_operand_blob_objects, mut1_operand_blob_objects, mut2_operand_blob_objects)
        bn_in_op2blob_object.update(
                self._SystemGetObn2BlobObject(op_conf, mut1_operand_blob_objects))
        bn_in_op2blob_object.update(
                self._SystemGetObn2BlobObject(op_conf, mut2_operand_blob_objects))

    def DeleteBlob(self, blob_object):
        self._TryClearObject(blob_object)
        self._DeleteObject(blob_object)

    def WatchBlobHeader(self, blob_object, callback):
        return self._WatchBlob("WatchBlobHeader", blob_object, callback)

    def WatchBlobBody(self, blob_object, callback):
        return self._WatchBlob("WatchBlobBody", blob_object, callback)

    def UnpackLogicalBlobToPhysicalBlobs(self, blob_object):
        parallel_desc_symbol = blob_object.parallel_desc_symbol
        parallel_conf = parallel_desc_symbol.parallel_conf
        device_tag = parallel_desc_symbol.device_tag
        machine_id2device_ids = placement_ctx.MakeMachineId2DeviceIdList(parallel_conf)
        def GetPhysicalBlob(machine_id, device_id):
            parallel_conf = placement_pb_util.ParallelConf()
            parallel_conf.device_name.append("%d:%s:%d" % (machine_id, device_tag, device_id))
            parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf, device_tag)
            pyhsical_blob_object = self._NewBlobObject(parallel_desc_sym)
            return pyhsical_blob_object
        physical_blob_objects = []
        for machine_id, device_ids in machine_id2device_ids.items():
            for device_id in device_ids: 
                physical_blob_objects.append(GetPhysicalBlob(machine_id, device_id))
        self._ReplaceMirrored(parallel_desc_symbol, physical_blob_objects, [blob_object])
        return physical_blob_objects

    def GetSymbol4String(self, string):
        if symbol_cache.HasSymbol4String(string): return symbol_cache.GetSymbol4String(string)
        symbol_id = self._NewSymbolId4String(string)
        symbol = symbol_util.Symbol(symbol_id, string)
        symbol_cache.SetSymbol4String(string, symbol)
        return symbol

    def GetJobConfSymbol(self, job_conf):
        if symbol_cache.HasSymbol4JobConf(job_conf):
            return symbol_cache.GetSymbol4JobConf(job_conf)
        symbol_id = self._NewSymbolId4JobConf(job_conf)
        symbol = symbol_util.Symbol(symbol_id, job_conf)
        symbol_cache.SetSymbol4JobConf(job_conf, symbol)
        return symbol

    def GetParallelDescSymbol(self, parallel_conf, device_tag):
        serialized_parallel_conf = parallel_conf.SerializeToString()
        if symbol_cache.HasSymbol4SerializedParallelConf(serialized_parallel_conf):
            return symbol_cache.GetSymbol4SerializedParallelConf(serialized_parallel_conf)
        symbol_id = self._NewSymbolId4ParallelConf(parallel_conf)
        symbol = symbol_util.ParallelDescSymbol(symbol_id, parallel_conf, device_tag)
        symbol_cache.SetSymbol4SerializedParallelConf(serialized_parallel_conf, symbol)
        return symbol

    def GetSharedOpKernelObject4ParallelConfSymbol(self, parallel_desc_sym):
        if object_cache.HasSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym):
            return object_cache.GetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym)
        object_id = self._NewSharedOpKernelObjectId4ParallelConfSymbolId(parallel_desc_sym)
        obj = object_util.Object(object_id, parallel_desc_sym)
        object_cache.SetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym, obj)
        return obj

    @contextmanager
    def CudaHostPinBlob(self, blob_object):
        self._CudaHostRegisterBlob(blob_object)
        try:
            yield
        finally:
            self._CudaHostUnregisterBlob(blob_object)

    def BroadcastBlobReference(self, sole_mirrored_blob_object, parallel_desc_sym):
        device_ids = sole_mirrored_blob_object.parallel_desc_symbol.machine_id2device_id_list
        for _, dev_ids in device_ids.items(): assert len(dev_ids) == 1, "dev_ids: %s" % dev_ids
        object_id = self._BroadcastObjectReference(sole_mirrored_blob_object, parallel_desc_sym)
        return object_util.BlobObject(object_id, parallel_desc_sym, self.release_blob_object_)

    def _CudaHostRegisterBlob(self, blob_object):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "CudaHostRegisterBlob"
        instruction.parallel_desc_symbol_id = blob_object.parallel_desc_symbol.symbol_id
        instruction.operand.append(_MutOperand(blob_object.object_id))
        self.instruction_list_.instruction.append(instruction)

    def _CudaHostUnregisterBlob(self, blob_object):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "CudaHostUnregisterBlob"
        instruction.parallel_desc_symbol_id = blob_object.parallel_desc_symbol.symbol_id
        instruction.operand.append(_MutOperand(blob_object.object_id))
        self.instruction_list_.instruction.append(instruction)

    def _GetOpConfSymbol(self, op_conf):
        new_op_conf = op_conf_util.OperatorConf()
        new_op_conf.CopyFrom(op_conf)
        # drop unique name to achieve a higher cache hit rate
        new_op_conf.name = new_op_conf.user_conf.op_type_name
        for ibn, lbns in new_op_conf.user_conf.input.items():
            for i in range(len(lbns.s)): lbns.s[i] = "%s/%s_%d"%(new_op_conf.name, ibn, i)
        for obn, lbns in new_op_conf.user_conf.output.items():
            for i in range(len(lbns.s)): lbns.s[i] = "%s/%s_%d"%(new_op_conf.name, obn, i)
        serialized_op_conf = new_op_conf.SerializeToString()
        if symbol_cache.HasSymbol4SerializedOpConf(serialized_op_conf):
            return symbol_cache.GetSymbol4SerializedOpConf(serialized_op_conf)
        symbol_id = self._NewSymbolId4OpConf(new_op_conf)
        symbol = symbol_util.Symbol(symbol_id, new_op_conf)
        symbol_cache.SetSymbol4SerializedOpConf(serialized_op_conf, symbol)
        return symbol

    def _SystemGetOpConfSymbol(self, op_conf):
        new_op_conf = op_conf_util.OperatorConf()
        new_op_conf.CopyFrom(op_conf)
        serialized_op_conf = new_op_conf.SerializeToString()
        if symbol_cache.HasSymbol4SerializedOpConf(serialized_op_conf):
            return symbol_cache.GetSymbol4SerializedOpConf(serialized_op_conf)
        symbol_id = self._NewSymbolId4OpConf(new_op_conf)
        symbol = symbol_util.Symbol(symbol_id, new_op_conf)
        symbol_cache.SetSymbol4SerializedOpConf(serialized_op_conf, symbol)
        return symbol

    def _GetConstOperandBlobObjects(self, op_conf, parallel_desc_sym):
        const_operand_blob_objects = []
        for ibn_prefix, lbns in op_conf.user_conf.input.items():
            for i in range(len(lbns.s)):
                ibn = "%s_%d"%(ibn_prefix, i)
                ibn_sym = self.GetSymbol4String(ibn)
                x_blob_object = object_cache.GetObject4BlobName(lbns.s[i])
                in_object = _FindOrCreateDelegateBlobObject(self, x_blob_object, parallel_desc_sym)
                const_operand_blob_objects.append((ibn_sym, in_object))
        return const_operand_blob_objects

    def _SystemGetConstOperandBlobObjects(self, op_conf, ibns, blob_object4ibn=None):
        assert callable(blob_object4ibn)
        system_op_conf = getattr(op_conf, op_conf.WhichOneof('op_type'))
        const_operand_blob_objects = []
        for ibn in ibns:
            ibn_sym = self.GetSymbol4String(ibn)
            in_object = blob_object4ibn(ibn)
            const_operand_blob_objects.append((ibn_sym, in_object))
        return const_operand_blob_objects

    def _GetMut1OperandBlobObjects(self, op_conf, parallel_desc_sym):
        mut1_operand_blob_objects = []
        for obn_prefix, lbns in op_conf.user_conf.output.items():
            for i in range(len(lbns.s)):
                obn = "%s_%d" %(obn_prefix, i)
                obn_sym = self.GetSymbol4String(obn)
                out_object = self._NewBlobObject(parallel_desc_sym)
                object_cache.SetObject4BlobName(lbns.s[i], out_object)
                mut1_operand_blob_objects.append((obn_sym, out_object))
        return mut1_operand_blob_objects

    def _SystemGetMut1OperandBlobObjects(self, op_conf, bns_in_op, parallel_desc_sym,
            blob_object4ibn=None):
        assert callable(blob_object4ibn)
        field = op_conf.WhichOneof('op_type')
        assert field is not None
        system_op_conf = getattr(op_conf, field)
        mut1_operand_blob_objects = []
        for bn_in_op in bns_in_op:
            obn_sym = self.GetSymbol4String(bn_in_op)
            if _GetOpConfBlobNameAttr(system_op_conf, bn_in_op).find("/") >= 0:
                out_object = blob_object4ibn(bn_in_op)
            else:
                out_object = self._NewBlobObject(parallel_desc_sym)
            mut1_operand_blob_objects.append((obn_sym, out_object))
        return mut1_operand_blob_objects

    def _SystemGetObn2BlobObject(self, op_conf, mut1_operand_blob_objects):
        system_op_conf = getattr(op_conf, op_conf.WhichOneof('op_type'))
        obn2blob_object = {}
        for obn_symbol, blob_object in mut1_operand_blob_objects:
            if obn_symbol.data in obn2blob_object: continue
            obn2blob_object[obn_symbol.data] = blob_object
        return obn2blob_object

    def _GetMut2OperandBlobObjects(self, op_conf, parallel_desc_sym):
        mut2_operand_blob_objects = []
        # TODO(lixinqi)
        return mut2_operand_blob_objects

    def _SystemGetMut2OperandBlobObjects(self, op_conf, bns_in_op, parallel_desc_sym,
                                        blob_object4ibn=None):
        field = op_conf.WhichOneof('op_type')
        assert field is not None
        system_op_conf = getattr(op_conf, field)
        mut2_operand_blob_objects = []
        if len(mut2_operand_blob_objects) != 0:
            raise NotImplementedError("mut2 output not supported")
        # TODO(lixinqi)
        return mut2_operand_blob_objects

    def _NewBlobObject(self, parallel_desc_sym):
        object_id = self._NewObjectId(parallel_desc_sym)
        return object_util.BlobObject(object_id, parallel_desc_sym, self.release_blob_object_)

    def _NewSymbolId4String(self, string):
        symbol_id = self._NewSymbolId()
        self._InitStringSymbol(symbol_id, string)
        return symbol_id

    def _NewSymbolId4ParallelConf(self, parallel_conf):
        symbol_id = self.id_generator_.NewSymbolId()
        self._NewParallelConfSymbol(symbol_id, parallel_conf)
        return symbol_id

    def _NewSymbolId4JobConf(self, job_conf):
        symbol_id = self._NewSymbolId()
        self._InitJobConfSymbol(symbol_id, job_conf)
        return symbol_id

    def _NewSymbolId4OpConf(self, op_conf):
        symbol_id = self._NewSymbolId()
        self._InitOpConfSymbol(symbol_id, op_conf)
        return symbol_id

    def _NewSharedOpKernelObjectId4ParallelConfSymbolId(self, parallel_desc_sym):
        return self._NewObjectId(parallel_desc_sym)

    def _StatelessCallOpKernel(self, instr_name, parallel_desc_sym, job_conf_sym, op_conf_sym,
                               shared_opkernel_obj, const_operand_blob_objects,
                               mut1_operand_blob_objects, mut2_operand_blob_objects):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "%s.%s" % (parallel_desc_sym.device_tag, instr_name)
        instruction.parallel_desc_symbol_id = parallel_desc_sym.symbol_id
        instruction.operand.append(_SymbolOperand(job_conf_sym.symbol_id))
        instruction.operand.append(_SymbolOperand(op_conf_sym.symbol_id))
        instruction.operand.append(_MutOperand(shared_opkernel_obj.object_id))
        instruction.operand.append(_OperandSeparator())
        for ibn_sym, _ in const_operand_blob_objects:
            instruction.operand.append(_SymbolOperand(ibn_sym.symbol_id))
        for _, blob_object in const_operand_blob_objects:
            instruction.operand.append(_ConstOperand(blob_object.object_id))
        instruction.operand.append(_OperandSeparator())
        for obn_sym, _ in mut1_operand_blob_objects:
            instruction.operand.append(_SymbolOperand(obn_sym.symbol_id))
        for _, blob_object in mut1_operand_blob_objects:
            instruction.operand.append(_MutOperand(blob_object.object_id))
        instruction.operand.append(_OperandSeparator())
        for obn_sym, _ in mut2_operand_blob_objects:
            instruction.operand.append(_SymbolOperand(obn_sym.symbol_id))
        for _, blob_object in mut2_operand_blob_objects:
            instruction.operand.append(_Mut2Operand(blob_object.object_id))
        self.instruction_list_.instruction.append(instruction)

    def _NewSymbolId(self):
        symbol_id = self.id_generator_.NewSymbolId()
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "NewSymbol"
        instruction.operand.append(_Int64Operand(symbol_id))
        self.instruction_list_.instruction.append(instruction)
        return symbol_id

    def _NewObjectId(self, parallel_desc_sym):
        object_id = self.id_generator_.NewObjectId()
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "NewObject"
        instruction.parallel_desc_symbol_id = parallel_desc_sym.symbol_id
        instruction.operand.append(_Int64Operand(object_id))
        self.instruction_list_.instruction.append(instruction)
        return object_id

    def _BroadcastObjectReference(self, sole_mirrored_object, parallel_desc_sym):
        object_id = self.id_generator_.NewObjectId()
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "BroadcastObjectReference"
        instruction.parallel_desc_symbol_id = parallel_desc_sym.symbol_id
        instruction.operand.append(_Int64Operand(object_id))
        instruction.operand.append(_Int64Operand(sole_mirrored_object.object_id))
        self.instruction_list_.instruction.append(instruction)
        return object_id

    def _InitStringSymbol(self, symbol_id, string):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "InitStringSymbol"
        instruction.operand.append(_InitSymbolOperand(symbol_id))
        self.instruction_list_.instruction.append(instruction)
        eager_symbol = eager_symbol_util.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.string_symbol = string
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _NewParallelConfSymbol(self, symbol_id, parallel_conf):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "NewParallelDescSymbol"
        instruction.operand.append(_Int64Operand(symbol_id))
        self.instruction_list_.instruction.append(instruction)
        eager_symbol = eager_symbol_util.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.parallel_conf_symbol.CopyFrom(parallel_conf)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _InitJobConfSymbol(self, symbol_id, job_conf):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "InitJobDescSymbol"
        instruction.operand.append(_InitSymbolOperand(symbol_id))
        self.instruction_list_.instruction.append(instruction)
        eager_symbol = eager_symbol_util.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.job_conf_symbol.CopyFrom(job_conf)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _InitOpConfSymbol(self, symbol_id, op_conf):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "InitOperatorConfSymbol"
        instruction.operand.append(_InitSymbolOperand(symbol_id))
        self.instruction_list_.instruction.append(instruction)
        eager_symbol = eager_symbol_util.EagerSymbol()
        eager_symbol.symbol_id = symbol_id
        eager_symbol.op_conf_symbol.CopyFrom(op_conf)
        self.eager_symbol_list_.eager_symbol.append(eager_symbol)

    def _WatchBlob(self, instruction_name, blob_object, fetcher):
        unique_callback_id = physical_blob_callback.GetIdForRegisteredCallback(fetcher)
        instruction = instr_util.InstructionProto()
        device_tag = blob_object.parallel_desc_symbol.device_tag
        instruction.instr_type_name = "%s.%s"%(device_tag, instruction_name)
        instruction.parallel_desc_symbol_id = blob_object.parallel_desc_symbol.symbol_id
        instruction.operand.append(_ConstOperand(blob_object.object_id))
        instruction.operand.append(_Int64Operand(unique_callback_id))
        self.instruction_list_.instruction.append(instruction)

    def _TryClearObject(self, blob_object):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "TryClearObject"
        instruction.parallel_desc_symbol_id = blob_object.parallel_desc_symbol.symbol_id
        instruction.operand.append(_MutOperand(blob_object.object_id))
        self.instruction_list_.instruction.append(instruction)

    def _DeleteObject(self, blob_object):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "DeleteObject"
        instruction.operand.append(_DelObjectOperand(blob_object.object_id))
        self.instruction_list_.instruction.append(instruction)

    def _ReplaceMirrored(self, parallel_desc_sym, lhs_objects, rhs_objects):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "ReplaceMirrored"
        instruction.parallel_desc_symbol_id = parallel_desc_sym.symbol_id
        for lhs_object in lhs_objects:
            instruction.operand.append(_Int64Operand(lhs_object.object_id))
        instruction.operand.append(_OperandSeparator())
        for rhs_object in rhs_objects:
            instruction.operand.append(_Int64Operand(rhs_object.object_id))
        self.instruction_list_.instruction.append(instruction)

def _SymbolOperand(val):
    operand = instr_util.InstructionOperandProto()
    _SetSoleMirroredOperand(operand.symbol_operand, val)
    return operand

def _InitSymbolOperand(val):
    operand = instr_util.InstructionOperandProto()
    _SetSoleMirroredOperand(operand.init_symbol_operand, val)
    return operand

def _ConstOperand(val):
    operand = instr_util.InstructionOperandProto()
    _SetMirroredOperand(operand.const_operand, val)
    return operand

def _MutOperand(val):
    operand = instr_util.InstructionOperandProto()
    _SetMirroredOperand(operand.mut_operand, val)
    return operand

def _Mut2Operand(val):
    operand = instr_util.InstructionOperandProto()
    _SetMirroredOperand(operand.mut2_operand, val)
    return operand

def _DelObjectOperand(val):
    operand = instr_util.InstructionOperandProto()
    _SetAllMirroredOperand(operand.mut_operand, val)
    return operand

def _Int64Operand(val):
    operand = instr_util.InstructionOperandProto()
    operand.int64_operand = val
    return operand

def _OperandSeparator():
    operand = instr_util.InstructionOperandProto()
    operand.separator.SetInParent()
    return operand

def _SetMirroredOperand(operand, val):
    operand.logical_object_id = val
    operand.current_global_device_id.SetInParent()

def _SetSoleMirroredOperand(operand, val):
    operand.logical_object_id = val
    operand.sole_mirrored_object.SetInParent()

def _SetAllMirroredOperand(operand, val):
    operand.logical_object_id = val
    operand.all_mirrored_object.SetInParent()


def _FindOrCreateDelegateBlobObject(builder, x_blob_object, op_parallel_desc_symbol):
    if x_blob_object.parallel_desc_symbol == op_parallel_desc_symbol: return x_blob_object
    blob_cache = blob_cache_util.FindOrCreateBlobCache(x_blob_object)
    def Fetch(x_blob_object, op_parallel_desc_symbol):
        return _FetchDelegateBlobObject(builder, x_blob_object, op_parallel_desc_symbol)
    return blob_cache.GetCachedDelegateBlobObject(op_parallel_desc_symbol, Fetch)

def _FetchDelegateBlobObject(builder, x_blob_object, op_parallel_desc_symbol):
    blob_device_ids = x_blob_object.parallel_desc_symbol.machine_id2device_id_list
    op_device_ids = op_parallel_desc_symbol.machine_id2device_id_list
    prompt = "\nboxing is not supported yet."
    assert blob_device_ids == op_device_ids, "%s blob_device_ids: %s\nop_device_ids: %s"%(
            prompt, blob_device_ids, op_device_ids)
    blob_device_tag = x_blob_object.parallel_desc_symbol.device_tag
    op_device_tag = op_parallel_desc_symbol.device_tag
    assert blob_device_tag != op_device_tag, "\nblob_device_tag: %s\nop_device_tag: %s"%(
            blob_device_tag, op_device_tag)
    return boxing_util.BuildCopyHdInstruction(builder, x_blob_object)

def _GetOpConfBlobNameAttr(pb_message, field):
    if hasattr(pb_message, field): return getattr(pb_message, field);
    m = re.search("_(\d+)$", field)
    assert m is not None
    blob_name = field[0:-len(m.group(0))]
    index = int(m.group(0)[1:])
    assert hasattr(pb_message, blob_name), (pb_message, blob_name)
    repeated_field = getattr(pb_message, blob_name)
    assert index >= 0
    assert index < len(repeated_field)
    return repeated_field[index]

def _ReleaseLogicalBlobObject(blob_object):
    LogicalRun(lambda builder: builder.DeleteBlob(blob_object))

def _ReleasePhysicalBlobObject(blob_object):
    PhysicalRun(lambda builder: builder.DeleteBlob(blob_object))

