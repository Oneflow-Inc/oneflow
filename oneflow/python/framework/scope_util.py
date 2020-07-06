from __future__ import absolute_import

from oneflow.python.eager.symbol import Symbol
import oneflow.python.eager.symbol_cache as symbol_cache
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.core.job.scope_pb2 as scope_pb
import collections
import re


class ScopeSymbol(Symbol):
    def __init__(self, symbol_id, scope_proto, parent_scope_symbol=None):
        Symbol.__init__(self, symbol_id, scope_proto)
        self.parent_scope_symbol_ = parent_scope_symbol
        self.job_desc_symbol_ = symbol_cache.GetSymbol4Id(
            scope_proto.job_desc_symbol_id
        )
        self.device_parallel_desc_symbol_ = symbol_cache.GetSymbol4Id(
            scope_proto.device_parallel_desc_symbol_id
        )
        self.host_parallel_desc_symbol_ = symbol_cache.GetSymbol4Id(
            scope_proto.host_parallel_desc_symbol_id
        )

    @property
    def job_desc_symbol(self):
        return self.job_desc_symbol_

    @property
    def parent_scope_symbol(self):
        return self.parent_scope_symbol_

    def BuildWithNewParallelDesc(self, builder, device_tag, machine_device_ids):
        parallel_conf = MakeParallelConf(device_tag, machine_device_ids)
        device_parallel_desc_sym = builder.GetParallelDescSymbol(parallel_conf)
        parallel_conf = MakeParallelConf("cpu", machine_device_ids)
        host_parallel_desc_sym = builder.GetParallelDescSymbol(parallel_conf)
        scope_proto = self._CloneScopeProto()
        scope_proto.device_parallel_desc_symbol_id = device_parallel_desc_sym.symbol_id
        scope_proto.host_parallel_desc_symbol_id = host_parallel_desc_sym.symbol_id
        return builder.GetScopeSymbol(scope_proto, self)

    def BuildWithNewIsMirrored(self, builder, is_mirrored):
        scope_proto = self._CloneScopeProto()
        if is_mirrored:
            scope_proto.opt_mirrored_parallel_conf.mirrored_parallel.SetInParent()
        else:
            scope_proto.opt_mirrored_parallel_conf.ClearField("mirrored_parallel")
        return builder.GetScopeSymbol(scope_proto, self)

    def BuildWithNewScopeName(self, builder, scope_name):
        scope_proto = self._CloneScopeProto()
        scope_proto.scope_op_name_prefixes.append(scope_name)
        return builder.GetScopeSymbol(scope_proto, self)

    def _CloneScopeProto(self):
        scope_proto = scope_pb.ScopeProto()
        scope_proto.CopyFrom(self.data)
        scope_proto.ClearField("symbol_id")
        return scope_proto


def BuildInitialScope(builder, job_conf, device_tag, machine_device_ids, is_mirrored):
    scope_proto = scope_pb.ScopeProto()
    job_conf_sym = builder.GetJobConfSymbol(job_conf)
    scope_proto.job_desc_symbol_id = job_conf_sym.symbol_id
    parallel_conf = MakeParallelConf(device_tag, machine_device_ids)
    device_parallel_desc_sym = builder.GetParallelDescSymbol(parallel_conf)
    scope_proto.device_parallel_desc_symbol_id = device_parallel_desc_sym.symbol_id
    parallel_conf = MakeParallelConf("cpu", machine_device_ids)
    host_parallel_desc_sym = builder.GetParallelDescSymbol(parallel_conf)
    scope_proto.host_parallel_desc_symbol_id = host_parallel_desc_sym.symbol_id
    if is_mirrored:
        scope_proto.opt_mirrored_parallel_conf.mirrored_parallel.SetInParent()
    else:
        scope_proto.opt_mirrored_parallel_conf.ClearField("mirrored_parallel")
    return builder.GetScopeSymbol(scope_proto, None)


def MakeParallelConf(device_tag, machine_device_ids):
    assert isinstance(machine_device_ids, collections.Sized)
    device_names = []
    for machine_device_id in machine_device_ids:
        assert isinstance(
            machine_device_id, str
        ), "type of machine_device_id (%s) is not string" % type(machine_device_id)
        assert re.match("^\d+:\d+(-\d+)?$", machine_device_id) is not None, (
            "machine_device_id: %s is not valid" % machine_device_id
        )
        pair = machine_device_id.split(":")
        device_names.append("%s:%s:%s" % (pair[0], device_tag, pair[1]))

    parallel_conf = placement_pb.ParallelConf()
    parallel_conf.device_name.extend(device_names)
    return parallel_conf
