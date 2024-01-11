import os
from typing import Any, List, Optional, Union
import torch
from diffusers import StableDiffusionXLPipeline
import diffusers_quant
from diffusers_quant.utils import (
    rewrite_sdxl_pipeline_attention,
    replace_sub_module_with_quantizable_module,
)
from torch._dynamo import allow_in_graph as maybe_allow_in_graph


def _use_graph():
    os.environ["with_graph"] = "1"
    os.environ["ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION"] = "1"
    os.environ["ONEFLOW_MLIR_CSE"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
    os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
    os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
    os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
    os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
    os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
    os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
    os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
    os.environ["ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
    os.environ["ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL"] = "1"
    os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
    os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
    os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
    # os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "0"
    os.environ["ONEFLOW_MLIR_GROUP_MATMUL_QUANT"] = "1"
    os.environ["ONEFLOW_FUSE_QUANT_TO_MATMUL"] = "0"
    # os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
    # os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"
    
class QuantDiffusionPipeline:
    def __init__(
        self,
        pretrained_model_name_or_path,
        pipe,
        fake_quant=False,
        static=False,
        bits=8,
        graph=True,
    ):
        self._model = pretrained_model_name_or_path
        self._pipe = pipe
        self._fake_quant = fake_quant
        self._static = static
        self._bits = bits
        self._graph = graph

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        fake_quant=False,
        static=False,
        bits=8,
        graph=True,
        **kwargs
    ):
        diffusers_quant.enable_load_quantized_model()

        pipe = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            **kwargs
        )
        if graph:
            _use_graph()
        quant_pipe = cls(
            pretrained_model_name_or_path, pipe, fake_quant, static, bits, graph
        )

        return quant_pipe

    def _to(self, device):
        self._pipe.to(device)
        self._quantize_model()
        return self

    def _load_calib_info(self):
        calibrate_info = {}
        with open(
            os.path.join(self._model, "calibrate_info.txt"), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                line = line.strip()
                items = line.split(" ")
                calibrate_info[items[0]] = [
                    float(items[1]),
                    int(items[2]),
                    [float(x) for x in items[3].split(",")],
                ]
        return calibrate_info

    def _quantize_model(self):
        calibrate_info = self._load_calib_info()
        for sub_module_name, sub_calibrate_info in calibrate_info.items():
            replace_sub_module_with_quantizable_module(
                self._pipe.unet,
                sub_module_name,
                sub_calibrate_info,
                self._fake_quant,
                self._static,
                self._bits,
                maybe_allow_in_graph,
            )
        rewrite_sdxl_pipeline_attention(self._pipe)

    def __getattribute__(self, __name: str) -> Any:
        if __name == "to":
            return self._to
        if __name.startswith("_"):
            return object.__getattribute__(self, __name)
        return getattr(self._pipe, __name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            return object.__setattr__(self, name, value)
        return setattr(self._pipe, name, value)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        **kwargs
    ):
        return self._pipe(
            prompt, prompt_2, height, width, num_inference_steps, **kwargs
        )
