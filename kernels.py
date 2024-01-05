#! /usr/bin/python

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch import LayerNormMLP
from transformer_engine.common.recipe import Format, DelayedScaling
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
        LlamaMLP,
        LlamaRMSNorm
)
import numpy as np
import matplotlib.pyplot as plt
import nvtx

"""
import msamp
from msamp.common.dtype import Dtypes
from msamp.operators.gemm import Gemm
from msamp.operators.arithmetic import Arithmetic
from msamp.common.tensor import ScalingMeta
from msamp.common.tensor import TypeCast
from msamp.te.modules import (
        MSAMPLinear,
        MSAMPLayerNormMLP,
        MSAMPLayerNormLinear
)
"""

import layernorm_mlp as qtest

class TENaiveNet(nn.Module):
    """Feed-forward network in Transformer layer
    Built with plain PyTorch modules."""
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        ffn_hidden_size = config.intermediate_size
        self.layernorm = te.LayerNorm(hidden_size)
        self.linear1 = te.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = te.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, use_fp8: bool) -> torch.Tensor:
        if not use_fp8:
            x = self.layernorm(x)
            x = self.linear1(x)
            x = torch.nn.functional.silu(x)
            # x = tex.swiglu (
                    # fc1_out,
                    # fp8_meta["scaling_fwd"],
                    # tex.FP8FwdTensors.GEMM2_INPUT,
                    # fp8_dtype_forward,
                # )
            x = self.linear2(x)
        else:
            x = self.layernorm(x)
            # x = tex.cast_to_fp8(x)
            x = self.linear1(x)
            x = torch.nn.functional.silu(x)
            # x = tex.swiglu (
                    # fc1_out,
                    # fp8_meta["scaling_fwd"],
                    # tex.FP8FwdTensors.GEMM2_INPUT,
                    # fp8_dtype_forward,
                # )
            x = self.linear2(x)

        return x

class BaselineNet(nn.Module):

    def __init__(self, config):
        super(BaselineNet, self).__init__()
        self.layernorm = LlamaRMSNorm(config.hidden_size)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.mlp(x)
        return x

class QSiluNet(nn.Module):
    def __init__(self, config):
        super(QSiluNet, self).__init__()
        self.layernorm = LlamaRMSNorm(config.hidden_size)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.mlp(x)
        return x


def share_weights():
    pass


def benchmark_kernels():
    print('benchmark_kernels')
    enable_fp8 = False
    enable_fp16 = True
    num_trials = 20
    warm_up = 10
    torch.manual_seed(1234)

    hsizes = [4096, 5120, 8192]
    inter_sizes=[11008,13824,28672]
    sizes = zip(hsizes, inter_sizes)
    # dtypes = [Dtypes.kbfloat16]
    dtypes = [torch.float32, torch.float16]
    dtypes = [torch.bfloat16]
    batch_sizes=[32, 64]
    # batch_sizes=[32]
    # qtypes = [Dtypes.kfloat8_e4m3]
    results = {}

    for size, dtype, batch_size in itertools.product(sizes, dtypes,
            # batch_sizes, [False, True]):
            batch_sizes):
        # string = 'h={}, inter={}, dtype={}, batch={}, fp8={}'.format(size[0], size[1], dtype, batch_sizes, use_fp8)
        string = 'h={}:{}, b={}'.format(size[0], size[1], batch_size)
        # print(string)
        hidden_size = size[0]
        inter_size = size[1]

        config = LlamaConfig(hidden_size=hidden_size,
                intermediate_size=inter_size)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        baseline_model = BaselineNet(config).cuda()
        baseline_model.train()
        naive_te_model = TENaiveNet(config).cuda()
        naive_te_model.train()
        fused_model = LayerNormMLP(
                config.hidden_size,
                config.intermediate_size,
                eps=config.rms_norm_eps,
                normalization='RMSNorm',
                activation='swiglu',
                #TODO
                #params_dtype=,
                #seq_length=seq_length,
                micro_batch_size=batch_size
                ).cuda()
        fused_model.train()

        # print(baseline_model.mlp.

        data = torch.rand(batch_size, config.hidden_size).cuda()
        # print('shape:', data.shape)

        for _ in range(warm_up):
            output = baseline_model(data)

        start.record()
        with nvtx.annotate('base '+ string, color="red"):
            for i in range(num_trials):
                output = baseline_model(data)
        end.record()
        torch.cuda.synchronize()
        print(string + ', baseline_model,', start.elapsed_time(end)/num_trials)


        for _ in range(warm_up):
            output = naive_te_model(data, False)

        start.record()
        with nvtx.annotate('naive '+ string, color="blue"):
            for i in range(num_trials):
                output = naive_te_model(data, False)
        end.record()
        torch.cuda.synchronize()
        print(string + ', naive_te_model,', start.elapsed_time(end)/num_trials)


        for _ in range(warm_up):
            output = fused_model(data)

        start.record()
        with nvtx.annotate('fused '+ string, color="green"):
            for i in range(num_trials):
                output = fused_model(data)
        end.record()
        torch.cuda.synchronize()
        print(string + ', fused_model,', start.elapsed_time(end)/num_trials)

        if enable_fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for _ in range(warm_up):
                    output = baseline_model(data)

                start.record()
                with nvtx.annotate('base '+ string, color="red"):
                    for i in range(num_trials):
                        output = baseline_model(data)
                end.record()
                torch.cuda.synchronize()
                print(string + ', fp16_baseline_model,', start.elapsed_time(end)/num_trials)


                for _ in range(warm_up):
                    output = naive_te_model(data, False)

                start.record()
                with nvtx.annotate('naive '+ string, color="blue"):
                    for i in range(num_trials):
                        output = naive_te_model(data, False)
                end.record()
                torch.cuda.synchronize()
                print(string + ', fp16_naive_te_model,', start.elapsed_time(end)/num_trials)


                for _ in range(warm_up):
                    output = fused_model(data)

                start.record()
                with nvtx.annotate('fused '+ string, color="green"):
                    for i in range(num_trials):
                        output = fused_model(data)
                end.record()
                torch.cuda.synchronize()
                print(string + ', fp16_fused_model,', start.elapsed_time(end)/num_trials)

        if enable_fp8:
            fp8_format = Format.HYBRID
            fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                    amax_history_len=16, amax_compute_algo="max")
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                for _ in range(warm_up):
                    output = naive_te_model(data, True)

                start.record()
                with nvtx.annotate('fp8_naive '+ string, color="blue"):
                    for i in range(num_trials):
                        output = naive_te_model(data, True)
                end.record()
                torch.cuda.synchronize()
                print(string + ', fp8_naive_te_model,', start.elapsed_time(end)/num_trials)


                for _ in range(warm_up):
                    output = fused_model(data)

                start.record()
                with nvtx.annotate('fp8_fused '+ string, color="green"):
                    for i in range(num_trials):
                        output = fused_model(data)
                end.record()
                torch.cuda.synchronize()
                print(string + ', fp8_fused_model,', start.elapsed_time(end)/num_trials)
        print('===')


    return


def run_test():
    # test_casting()
    # test_gemm()
    # test_scaling_factor()
    benchmark_kernels()

print('testing msamp')
run_test()
print('done run test')
