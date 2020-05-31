from __future__ import absolute_import

import oneflow.python.vm.id_util as id_util
import oneflow.core.vm.instruction_pb2 as instr_util
import oneflow.core.job.placement_pb2 as placement_pb_util
import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.eager.job_conf_ctx as job_conf_ctx
import oneflow.python.eager.symbol as symbol_util
import oneflow.python.eager.symbol_cache as symbol_cache
import oneflow.python.eager.object as object_util
import oneflow.python.eager.object_cache as object_cache
import oneflow.python.eager.physical_blob_callback as physical_blob_callback
import oneflow

def PhysicalRun(build):
    return _Run(build, id_util.PhysicalIdGenerator(), c_api_util.RunPhysicalInstruction)

def LogicalRun(build):
    return _Run(build, id_util.LogicalIdGenerator(), c_api_util.RunLogicalInstruction)

def _Run(build, id_generator, run_api):
    instruction_list = instr_util.InstructionListProto()
    eager_symbol_list = eager_symbol_util.EagerSymbolList()
    build(InstructionsBuilder(id_generator, instruction_list, eager_symbol_list))
    run_api(instruction_list, eager_symbol_list)

class InstructionsBuilder(object):
    def __init__(self, id_generator, instruction_list, eager_symbol_list):
        assert isinstance(instruction_list, instr_util.InstructionListProto)
        assert isinstance(eager_symbol_list, eager_symbol_util.EagerSymbolList)
        self.instruction_list_ = instruction_list
        self.eager_symbol_list_ = eager_symbol_list
        self.id_generator_ = id_generator

    def StatelessCall(self, op_conf):
        assert op_conf.HasField('user_conf')
        placement_scope = oneflow.placement.current_scope()
        parallel_conf = placement_scope.default_parallel_conf
        device_tag = placement_scope.default_device_tag
        parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf, device_tag)
        job_conf_sym = self.GetJobConfSymbol(job_conf_ctx.CurrentJobConf())
        op_conf_sym = self._GetOpConfSymbol(op_conf)
        opkernel_obj = self.GetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym)
        input_triples = self._GetInputTriples(op_conf)
        output_triples = self._GetOutputTriples(op_conf, parallel_desc_sym)
        mut2_output_triples = self._GetMut2OutputTriples(op_conf, parallel_desc_sym)
        return self._StatelessCall("StatelessCallOpKernel",
                parallel_desc_sym, job_conf_sym, op_conf_sym, opkernel_obj,
                input_triples, output_triples, mut2_output_triples)

    def DeprecatedStatelessCall(self, op_conf, const_arg_bns=[], mut_arg_bns=[], mut2_arg_bns=[]):
        assert isinstance(const_arg_bns, (list, tuple))
        assert isinstance(mut_arg_bns, (list, tuple))
        assert isinstance(mut2_arg_bns, (list, tuple))
        assert len(const_arg_bns) + len(mut_arg_bns) + len(mut2_arg_bns) > 0
        placement_scope = oneflow.placement.current_scope()
        parallel_conf = placement_scope.default_parallel_conf
        device_tag = placement_scope.default_device_tag
        parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf, device_tag)
        job_conf_sym = self.GetJobConfSymbol(job_conf_ctx.CurrentJobConf())
        op_conf_sym = self._DeprecatedGetOpConfSymbol(op_conf)
        opkernel_obj = self.GetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym)
        input_triples = self._DeprecatedGetInputTriples(op_conf, const_arg_bns)
        output_triples = self._DeprecatedGetOutputTriples(op_conf, mut_arg_bns, parallel_desc_sym)
        mut2_output_triples = self._DeprecatedGetMut2OutputTriples(
                op_conf, mut2_arg_bns, parallel_desc_sym)
        return self._StatelessCall("DeprecatedStatelessCallOpKernel",
                parallel_desc_sym, job_conf_sym, op_conf_sym, opkernel_obj,
                input_triples, output_triples, mut2_output_triples)

    def DeleteBlob(self, blob_object):
        self._TryClearObject(blob_object)
        self._DeleteObject(blob_object)

    def WatchBlobHeader(self, blob_object, callback):
        return self._WatchBlob("WatchBlobHeader", blob_object, callback)

    def WatchBlobBody(self, blob_object, callback):
        return self._WatchBlob("WatchBlobBody", blob_object, callback)

    def UnpackLogicalBlobNameToPhysicalBlobNames(self, blob_name):
        blob_object = object_cache.GetObject4BlobName(blob_name)
        parallel_desc_symbol = blob_object.parallel_desc_symbol
        parallel_conf = parallel_desc_symbol.parallel_conf
        device_tag = parallel_desc_symbol.device_tag
        machine_id2device_ids = placement_ctx.MakeMachineId2DeviceIdList(parallel_conf)
        def GetPhysicalBlob(machine_id, device_id):
            parallel_conf = placement_pb_util.ParallelConf()
            parallel_conf.device_name.append("%d:%s:%d" % (machine_id, device_tag, device_id))
            parallel_desc_sym = self.GetParallelDescSymbol(parallel_conf, device_tag)
            physical_blob_name = "%s/%d/%d" % (blob_name, machine_id, device_id)
            pyhsical_blob_object = self._NewBlobObject(physical_blob_name, parallel_desc_sym)
            return physical_blob_name, pyhsical_blob_object
        physical_blob_names = []
        physical_blob_objects = []
        for machine_id, device_ids in machine_id2device_ids.items():
            for device_id in device_ids: 
                physical_blob_name, pyhsical_blob_object = GetPhysicalBlob(machine_id, device_id)
                physical_blob_names.append(physical_blob_name)
                physical_blob_objects.append(pyhsical_blob_object)
        self._ReplaceMirrored(parallel_desc_symbol, physical_blob_objects, [blob_object])
        return physical_blob_names

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

    def _GetOpConfSymbol(self, op_conf):
        new_op_conf = op_conf_util.OperatorConf()
        new_op_conf.CopyFrom(op_conf)
        # drop unique name to achieve a higher cache hit rate
        new_op_conf.name = new_op_conf.user_conf.op_type_name
        new_op_conf.user_conf.ClearField("input")
        new_op_conf.user_conf.ClearField("output")
        serialized_op_conf = new_op_conf.SerializeToString()
        if symbol_cache.HasSymbol4SerializedOpConf(serialized_op_conf):
            return symbol_cache.GetSymbol4SerializedOpConf(serialized_op_conf)
        symbol_id = self._NewSymbolId4OpConf(new_op_conf)
        symbol = symbol_util.Symbol(symbol_id, new_op_conf)
        symbol_cache.SetSymbol4SerializedOpConf(serialized_op_conf, symbol)
        return symbol

    def _DeprecatedGetOpConfSymbol(self, op_conf):
        new_op_conf = op_conf_util.OperatorConf()
        new_op_conf.CopyFrom(op_conf)
        serialized_op_conf = new_op_conf.SerializeToString()
        if symbol_cache.HasSymbol4SerializedOpConf(serialized_op_conf):
            return symbol_cache.GetSymbol4SerializedOpConf(serialized_op_conf)
        symbol_id = self._NewSymbolId4OpConf(new_op_conf)
        symbol = symbol_util.Symbol(symbol_id, new_op_conf)
        symbol_cache.SetSymbol4SerializedOpConf(serialized_op_conf, symbol)
        return symbol

    def _GetInputTriples(self, op_conf):
        input_triples = []
        for ibn, lbns in op_conf.user_conf.input.items():
            ibn_sym = self.GetSymbol4String(ibn)
            for i in range(len(lbns.s)):
                in_object = object_cache.GetObject4BlobName(lbns.s[i])
                input_triples.append((ibn_sym, i, in_object))
        return input_triples

    def _DeprecatedGetInputTriples(self, op_conf, ibns):
        field = op_conf.WhichOneof('op_type')
        assert field is not None
        deprecated_op_conf = getattr(op_conf, field)
        input_triples = []
        for ibn in ibns:
            ibn_sym = self.GetSymbol4String(ibn)
            in_object = object_cache.GetObject4BlobName(getattr(deprecated_op_conf, ibn))
            input_triples.append((ibn_sym, 0, in_object))
        return input_triples

    def _GetOutputTriples(self, op_conf, parallel_desc_sym):
        output_triples = []
        for obn, lbns in op_conf.user_conf.output.items():
            obn_sym = self.GetSymbol4String(obn)
            for i in range(len(lbns.s)):
                out_object = self._NewBlobObject(lbns.s[i], parallel_desc_sym)
                output_triples.append((obn_sym, i, out_object))
        return output_triples

    def _DeprecatedGetOutputTriples(self, op_conf, bns_in_op, parallel_desc_sym):
        field = op_conf.WhichOneof('op_type')
        assert field is not None
        deprecated_op_conf = getattr(op_conf, field)
        def GetLogicalBlobName(bn_in_op):
            blob_name = getattr(deprecated_op_conf, bn_in_op)
            if blob_name.find("/") > 0: return blob_name
            return "%s/%s"%(op_conf.name, blob_name)
        output_triples = []
        for bn_in_op in bns_in_op:
            obn_sym = self.GetSymbol4String(bn_in_op)
            out_object = self._NewBlobObject(GetLogicalBlobName(bn_in_op), parallel_desc_sym)
            output_triples.append((obn_sym, 0, out_object))
        return output_triples

    def _GetMut2OutputTriples(self, op_conf, parallel_desc_sym):
        mut2_output_triples = []
        # TODO(lixinqi)
        return mut2_output_triples

    def _DeprecatedGetMut2OutputTriples(self, op_conf, bns_in_op, parallel_desc_sym):
        field = op_conf.WhichOneof('op_type')
        assert field is not None
        deprecated_op_conf = getattr(op_conf, field)
        mut2_output_triples = []
        # TODO(lixinqi)
        return mut2_output_triples

    def _NewBlobObject(self, blob_name, parallel_desc_sym):
        object_id = self._NewObjectId(parallel_desc_sym)
        blob_object = object_util.Object(object_id, parallel_desc_sym)
        object_cache.SetObject4BlobName(blob_name, blob_object)
        return blob_object

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

    def _StatelessCall(self, instr_name, parallel_desc_sym, job_conf_sym, op_conf_sym,
                       shared_opkernel_obj, input_triples, output_triples, mut2_output_triples):
        instruction = instr_util.InstructionProto()
        instruction.instr_type_name = "%s.%s" % (parallel_desc_sym.device_tag, instr_name)
        instruction.parallel_desc_symbol_id = parallel_desc_sym.symbol_id
        instruction.operand.append(_SymbolOperand(job_conf_sym.symbol_id))
        instruction.operand.append(_SymbolOperand(op_conf_sym.symbol_id))
        instruction.operand.append(_MutOperand(shared_opkernel_obj.object_id))
        instruction.operand.append(_OperandSeparator())
        for ibn_sym, _, _ in input_triples:
            instruction.operand.append(_SymbolOperand(ibn_sym.symbol_id))
        for _, index, _ in input_triples:
            instruction.operand.append(_Int64Operand(index))
        for _, _, blob_object in input_triples:
            instruction.operand.append(_ConstOperand(blob_object.object_id))
        instruction.operand.append(_OperandSeparator())
        for obn_sym, _, _ in output_triples:
            instruction.operand.append(_SymbolOperand(obn_sym.symbol_id))
        for _, index, _ in output_triples:
            instruction.operand.append(_Int64Operand(index))
        for _, _, blob_object in output_triples:
            instruction.operand.append(_MutOperand(blob_object.object_id))
        instruction.operand.append(_OperandSeparator())
        for obn_sym, _, _ in mut2_output_triples:
            instruction.operand.append(_SymbolOperand(obn_sym.symbol_id))
        for _, index, _ in mut2_output_triples:
            instruction.operand.append(_Int64Operand(index))
        for _, _, blob_object in mut2_output_triples:
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
