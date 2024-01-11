from infer_compiler import register

import oneflow as flow
import diffusers_quant

torch2oflow_class_map = {
    diffusers_quant.FakeQuantModule: diffusers_quant.OneFlowFakeQuantModule,
    diffusers_quant.StaticQuantConvModule: diffusers_quant.OneFlowStaticQuantConvModule,
    diffusers_quant.DynamicQuantConvModule: diffusers_quant.OneFlowDynamicQuantConvModule,
    diffusers_quant.StaticQuantLinearModule: diffusers_quant.OneFlowStaticQuantLinearModule,
    diffusers_quant.DynamicQuantLinearModule: diffusers_quant.OneFlowDynamicLinearQuantModule,
}


def convert_func(mod: flow.Tensor, verbose=False):
    return mod


register(torch2oflow_class_map=torch2oflow_class_map, torch2oflow_funcs=[convert_func])
