from __future__ import absolute_import

import functools
import copy
import re
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.function_desc import FunctionDesc
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.distribute_context as distribute_ctx
import oneflow.python.lib.core.pb_util as pb_util

@oneflow_export("FunctionConfig", 'function_config')
class FunctionConfig(object):
    def __init__(self):
        self.function_desc = FunctionDesc()
        
    def __getattr__(self, attr_name):
        name2default = session_ctx.GetDefaultSession().function_flag_name2default_val
        assert attr_name in name2default
        flag_name2flag_value = self.function_desc.job_config_proto.flag_name2flag_value
        default_val = name2default[attr_name]
        def FunctionConfigSetter(attr_value):
            if default_val.HasField('at_bool'):
                assert type(attr_value) is bool
                flag_name2flag_value[attr_name].at_bool = attr_value
            elif default_val.HasField('at_int64'):
                assert type(attr_value) is int
                flag_name2flag_value[attr_name].at_int64 = attr_value
            elif default_val.HasField('at_double'):
                assert type(attr_value) is float
                flag_name2flag_value[attr_name].at_double = attr_value
            elif default_val.HasField('at_string'):
                assert type(attr_value) is str
                flag_name2flag_value[attr_name].at_string = attr_value
            else:
                raise NotImplementedError("config_flag `%s' with type %s is not supported"
                                          % (attr_name, type(attr_value)))
        return FunctionConfigSetter

@oneflow_export('function')
def oneflow_function(function_config = FunctionConfig()):
    assert isinstance(function_config, FunctionConfig)
    def Decorator(job_func):
        sess = session_ctx.GetDefaultSession()
        @functools.wraps(job_func)
        def Func(*args):
            return _RunJob(sess, job_func, *args)
        for x in dir(job_func):
            if x.startswith('__oneflow_'): setattr(Func, x, getattr(job_func, x))
        sess.AddJob(_CloneFunctionDesc(function_config.function_desc, job_func))
        return Func
    return Decorator

def _CloneFunctionDesc(func_desc, job_func):
    new_func_desc = FunctionDesc(job_func=job_func)
    new_func_desc.job_config_proto.CopyFrom(func_desc.job_config_proto)
    _TryCompleteDefaultJobConfigProto(new_func_desc.job_config_proto)
    new_func_desc.function_attribute = copy.deepcopy(func_desc.function_attribute)
    return new_func_desc

def _TryCompleteDefaultJobConfigProto(job_conf):
    if job_conf.WhichOneof("job_type") is None:
        job_conf.predict_conf.SetInParent()

def oneflow_function_config(*field_paths):
    def Decorator(func):
        global _class_property2return_obj_class
        for field_path in field_paths:
            fields = field_path.split(".")
            assert len(fields) > 0
            cls = FunctionConfig
            for index, field in enumerate(fields):
                assert field != 'function_desc'
                assert re.match("^[_\w]+[_\w\d]*$", field)
                if (cls, field) not in _class_property2return_obj_class:
                    class_name = ".".join(['function_config'] + fields[:index + 1])
                    def Init(self, function_desc): self.function_desc = function_desc
                    config_class = type(class_name, (object,), dict(__init__= Init))
                    setattr(cls, field, _MakeInnerJobConfigClassProperty(config_class))
                    _class_property2return_obj_class[cls, field] = config_class
                cls = _class_property2return_obj_class[cls, field]
            cls.__call__ = _MakeLeafJobConfigCall(func)
        return func
    return Decorator

_class_property2return_obj_class = {}

def _MakeInnerJobConfigClassProperty(return_obj_class):
    return property(lambda self: return_obj_class(self.function_desc))

def _MakeLeafJobConfigCall(method):
    return lambda self, *argv, **kwarg: method(self.function_desc, *argv, **kwarg)

def _RunJob(session, job_func, *args):
    return session.TryInit().Run(job_func, *args)

@oneflow_function_config('default_data_type')
def set_default_data_type(func_desc, value):
    func_desc.job_config_proto.default_data_type = value

@oneflow_function_config('default_initializer_conf')
def set_default_initializer_conf(func_desc, value):
    assert type(value) is dict
    pb_util.PythonDict2PbMessage(value, func_desc.job_config_proto.default_initializer_conf)

@oneflow_function_config('exp_run_conf')
def set_exp_run_conf(value):
    assert type(func_desc, value) is dict
    pb_util.PythonDict2PbMessage(value, func_desc.job_config_proto.exp_run_conf)

@oneflow_function_config('use_memory_allocation_algorithm_v2')
def set_use_memory_allocation_algorithm_v2(func_desc, value):
    func_desc.job_config_proto.use_memory_allocation_algorithm_v2 = value

@oneflow_function_config('static_mem_alloc_policy_white_list.has')
def static_mem_alloc_policy_white_list_has_policy(func_desc, policy):
    return getattr(func_desc.job_config_proto.memory_allocation_algorithm_conf, policy)

@oneflow_function_config('static_mem_alloc_policy_white_list.add')
def static_mem_alloc_policy_white_list_add_policy(func_desc, policy):
    setattr(func_desc.job_config_proto.memory_allocation_algorithm_conf, policy, True)

@oneflow_function_config('static_mem_alloc_policy_white_list.remove')
def static_mem_alloc_policy_white_list_remove_policy(func_desc, policy):
    setattr(func_desc.job_config_proto.memory_allocation_algorithm_conf, policy, False)

@oneflow_function_config('static_mem_alloc_policy_white_list.policy_mem_size_first')
def policy_mem_size_first(func_desc):
    return "use_mem_size_first_algo"

@oneflow_function_config('static_mem_alloc_policy_white_list.policy_mutual_exclusion_first')
def policy_mutual_exclusion_first(func_desc):
    return "use_mutual_exclusion_first_algo"

@oneflow_function_config('static_mem_alloc_policy_white_list.policy_time_line')
def policy_time_line(func_desc):
    return "use_time_line_algo"

@oneflow_function_config('static_mem_alloc_algo_white_list.show')
def show_static_mem_alloc_algo_white_list(func_desc):
    return ["use_mem_size_first_algo", "use_mutual_exclusion_first_algo", "use_time_line_algo"]

@oneflow_function_config('enable_cudnn')
def set_enable_cudnn(func_desc, value = True):
    func_desc.job_config_proto.enable_cudnn = value

@oneflow_function_config('cudnn_buf_limit_mbyte')
def set_cudnn_buf_limit_mbyte(func_desc, value):
    func_desc.job_config_proto.cudnn_buf_limit_mbyte = value

@oneflow_function_config('cudnn_conv_force_fwd_algo')
def set_cudnn_conv_force_fwd_algo(func_desc, value):
    func_desc.job_config_proto.cudnn_conv_force_fwd_algo = value

@oneflow_function_config('cudnn_conv_force_bwd_data_algo')
def set_cudnn_conv_force_bwd_data_algo(func_desc, value):
    func_desc.job_config_proto.cudnn_conv_force_bwd_data_algo = value

@oneflow_function_config('cudnn_conv_force_bwd_filter_algo')
def set_cudnn_conv_force_bwd_filter_algo(func_desc, value):
    func_desc.job_config_proto.cudnn_conv_force_bwd_filter_algo = value

@oneflow_function_config('enable_reused_mem')
def set_enable_reused_mem(func_desc, value = True):
    func_desc.job_config_proto.enable_reused_mem = value

@oneflow_function_config('enable_inplace')
def set_enable_inplace(func_desc, value = True):
    func_desc.job_config_proto.enable_inplace = value

@oneflow_function_config('enable_inplace_in_reduce_struct')
def set_enable_inplace_in_reduce_struct(func_desc, value = True):
    func_desc.job_config_proto.enable_inplace_in_reduce_struct = value

@oneflow_function_config('enable_nccl')
def set_enable_nccl(func_desc, value = True):
    func_desc.job_config_proto.enable_nccl = value

@oneflow_function_config('use_nccl_inter_node_communication')
def set_use_nccl_inter_node_communication(func_desc, value = True):
    func_desc.job_config_proto.use_nccl_inter_node_communication = value

@oneflow_function_config('use_boxing_v2')
def set_use_boxing_v2(func_desc, value = True):
    func_desc.job_config_proto.use_boxing_v2 = value

@oneflow_function_config('enable_all_reduce_group')
def set_enable_all_reduce_group(func_desc, value = True):
    func_desc.job_config_proto.enable_all_reduce_group = value

@oneflow_function_config('all_reduce_group_num')
def set_all_reduce_group_num(func_desc, value):
    func_desc.job_config_proto.all_reduce_group_num = value

@oneflow_function_config('all_reduce_lazy_ratio')
def set_all_reduce_lazy_ratio(func_desc, value):
    func_desc.job_config_proto.all_reduce_lazy_ratio = value

@oneflow_function_config('all_reduce_group_min_mbyte')
def set_all_reduce_group_min_mbyte(func_desc, value):
    func_desc.job_config_proto.all_reduce_group_min_mbyte = value

@oneflow_function_config('all_reduce_group_size_warmup')
def set_all_reduce_group_size_warmup(func_desc, value):
    func_desc.job_config_proto.all_reduce_group_size_warmup = value

@oneflow_function_config('all_reduce_fp16')
def set_all_reduce_fp16(func_desc, value = True):
    func_desc.job_config_proto.all_reduce_fp16 = value

@oneflow_function_config('enable_non_distributed_optimizer')
def set_enable_non_distributed_optimizer(func_desc, value = True):
    func_desc.job_config_proto.enable_non_distributed_optimizer = value

@oneflow_function_config('disable_all_reduce_sequence')
def set_disable_all_reduce_sequence(func_desc, value = True):
    func_desc.job_config_proto.disable_all_reduce_sequence = value

@oneflow_function_config('non_distributed_optimizer_group_size_mbyte')
def set_non_distributed_optimizer_group_size_mbyte(func_desc, value):
    func_desc.job_config_proto.non_distributed_optimizer_group_size_mbyte = value

@oneflow_function_config('enable_true_half_config_when_conv', 'cudnn_conv_enable_true_half')
def set_cudnn_conv_enable_true_half(func_desc, value = True):
    func_desc.job_config_proto.cudnn_conv_enable_true_half = value

@oneflow_function_config('enable_float_compute_for_half_gemm')
def set_enable_float_compute_for_half_gemm(func_desc, value = True):
    func_desc.job_config_proto.enable_float_compute_for_half_gemm = value

@oneflow_function_config('enable_auto_mixed_precision')
def set_enable_auto_mixed_precision(func_desc, value = True):
    func_desc.job_config_proto.enable_auto_mixed_precision = value

@oneflow_function_config('concurrency_width')
def set_concurrency_width(func_desc, value):
    func_desc.job_config_proto.concurrency_width = value

@oneflow_function_config('train.model_update_conf')
def set_model_update_conf(func_desc, value):
    assert type(value) is dict
    pb_msg = func_desc.job_config_proto.train_conf.model_update_conf
    pb_util.PythonDict2PbMessage(value, pb_msg)

@oneflow_function_config('train.loss_scale_factor')
def set_loss_scale_factor(func_desc, value):
    func_desc.job_config_proto.train_conf.loss_scale_factor = value

@oneflow_function_config('train.primary_lr')
def set_primary_lr(func_desc, value):
    func_desc.job_config_proto.train_conf.primary_lr = value

@oneflow_function_config('train.secondary_lr')
def set_secondary_lr(func_desc, value):
    func_desc.job_config_proto.train_conf.secondary_lr = value

@oneflow_function_config('train.weight_l1')
def set_weight_l1(func_desc, value):
    func_desc.job_config_proto.train_conf.weight_l1 = value

@oneflow_function_config('train.bias_l1')
def set_bias_l1(func_desc, value):
    func_desc.job_config_proto.train_conf.bias_l1 = value

@oneflow_function_config('train.weight_l2')
def set_weight_l2(func_desc, value):
    func_desc.job_config_proto.train_conf.weight_l2 = value

@oneflow_function_config('train.bias_l2')
def set_bias_l2(func_desc, value):
    func_desc.job_config_proto.train_conf.bias_l2 = value

@oneflow_function_config('default_placement_scope')
def set_default_placement(func_desc, value):
    assert isinstance(value, placement_ctx.PlacementScope)
    func_desc.function_attribute.default_placement_scope = value

@oneflow_function_config('use_xla_jit')
def set_use_xla_jit(func_desc, value = True):
    func_desc.job_config_proto.xrt_config.use_xla_jit = value

@oneflow_function_config('use_tensorrt')
def set_use_tensorrt(func_desc, value = True):
    func_desc.job_config_proto.xrt_config.use_tensorrt = value

@oneflow_function_config('tensorrt.use_fp16')
def set_tensorrt_use_fp16(func_desc, value = True):
    set_use_tensorrt(func_desc, True)
    func_desc.job_config_proto.xrt_config.tensorrt_config.use_fp16 = value

@oneflow_function_config('tensorrt.use_int8')
def set_tensorrt_use_int8(func_desc, value = True):
    set_use_tensorrt(func_desc, True)
    func_desc.job_config_proto.xrt_config.tensorrt_config.use_int8 = value

@oneflow_function_config('default_distribute_strategy')
def set_default_distribute_strategy(func_desc, value):
    assert isinstance(value, distribute_ctx.DistributeStrategy)
    func_desc.function_attribute.default_distribute_strategy = value
