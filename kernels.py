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
import functools
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

import layernorm_mlp as prototype

fwd = False
bwd = True
num_trials = 2000
warm_up = 10
enable_fp8 = False
enable_fp16 = True

class TEOptimalNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        ffn_hidden_size = config.intermediate_size
        self.layernorm = te.LayerNorm(hidden_size)
        self.linear1 = te.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = te.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, use_fp8=False) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, use_fp8=False) -> torch.Tensor:
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

def benchmark_kernel_bwd(string, model, model_name, data, dy, use_fp8=False, use_fp16=False):
    assert not (use_fp8 and use_fp16)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.cuda()
    for _ in range(warm_up):
        output = model(data)

    prefix = ''
    if use_fp8:
        prefix = 'fp8_'
    if use_fp16:
        prefix = 'fp16_'
    string += ', ' + prefix + model_name

    # if not use_fp8 and not use_fp16:
        # print('kernel 32')
        # start.record()
        # for i in range(num_trials):
            # # with nvtx.annotate('base '+ string, color="red"):
            # output = model(data)
            # output.backward(dy)
        # end.record()
        # torch.cuda.synchronize()


    if use_fp16:
        print('kernel 16')
        start.record()
        for i in range(num_trials):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # with nvtx.annotate('base '+ string, color="red"):
                output = model(data)
            output.backward(dy)
        end.record()
        torch.cuda.synchronize()

    if use_fp8:
        print('kernel 8')
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                amax_history_len=16, amax_compute_algo="max")
        start.record()
        for i in range(num_trials):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                # with nvtx.annotate('base '+ string, color="red"):
                output = model(data)
            output.backward(dy)
        end.record()
        torch.cuda.synchronize()

    model.cpu()
    torch.cuda.empty_cache()
    # zero grad
    #data.detach()
    string += ','
    print(string, start.elapsed_time(end)/num_trials)

def benchmark_kernel(start, end, string, model, model_name, data, use_fp8=False, use_fp16=False):
    assert not (use_fp8 and use_fp16)

    with torch.no_grad():

        model.cuda()
        for _ in range(warm_up):
            output = model(data)

        prefix = ''
        if use_fp8:
            prefix = 'fp8_'
        if use_fp16:
            prefix = 'fp16_'
        string += ', ' + prefix + model_name


        start.record()
        with nvtx.annotate('base '+ string, color="red"):
            for i in range(num_trials):
                output = model(data)
        end.record()
        torch.cuda.synchronize()

        model.cpu()
        torch.cuda.empty_cache()
        # zero grad
        #data.detach()
        string += ','
        print(string, start.elapsed_time(end)/num_trials)



def benchmark_kernels():
    global benchmark_kernel
    torch.manual_seed(1234)

    hsizes = [4096, 5120, 8192]
    inter_sizes=[11008,13824,28672]
    sizes = zip(hsizes, inter_sizes)
    # dtypes = [Dtypes.kbfloat16]
    dtypes = [torch.float32, torch.float16]
    dtypes = [torch.bfloat16]
    batch_sizes=[32, 64]
    batch_sizes=[32]
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

        bench = functools.partial(benchmark_kernel, start, end,
                string)
        bench_bwd = functools.partial(benchmark_kernel_bwd, string)


        baseline_model = BaselineNet(config)
        baseline_model.train()
        naive_te_model = TENaiveNet(config)
        naive_te_model.train()
        fused_model = LayerNormMLP(
                config.hidden_size,
                config.intermediate_size,
                eps=config.rms_norm_eps,
                normalization='RMSNorm',
                activation='relu',
                #TODO
                params_dtype=torch.float16,
                #seq_length=seq_length,
                micro_batch_size=batch_size
                )
        fused_model.train()
        custom_fused_model = prototype.LayerNormMLP(
                config.hidden_size,
                config.intermediate_size,
                eps=config.rms_norm_eps,
                normalization='RMSNorm',
                activation='relu',
                #TODO
                params_dtype=torch.float16,
                #seq_length=seq_length,
                micro_batch_size=batch_size
                )
        custom_fused_model.train()

        if fwd:
            print('fwd')
            # print(baseline_model.mlp.

            # data = torch.rand(batch_size, config.hidden_size, dtype=torch.float16).cuda()
            # data = torch.rand(batch_size, config.hidden_size).cuda()
            # data.detach()
            # print('shape:', data.shape)

            # bench(baseline_model, "hf-llama", data)
            # bench(naive_te_model, "te-naive", data)
            # bench(fused_model, "te-fused", data)

            if enable_fp16:
                data = torch.rand(batch_size, config.hidden_size, dtype=torch.float16).cuda()
                data.detach()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    bench(baseline_model, "hf-llama", data, use_fp16=True)
                    # bench(naive_te_model, "te-naive", data, use_fp16=True)
                    bench(fused_model, "te-fused", data, use_fp16=True)
                    bench(custom_fused_model, "te-cus-fused", data, use_fp16=True)

            if enable_fp8:
                fp8_format = Format.HYBRID
                # for l in range(0, 16):
                for l in [1, 16]:
                    print('len,', l)
                    # fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                            # amax_history_len=l, amax_compute_algo="max")
                    # with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        # bench(naive_te_model, "te-naive", data, use_fp8=True)
                    fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                            amax_history_len=l, amax_compute_algo="max")
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        bench(fused_model, "te-fused", data, use_fp8=True)
                    fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                            amax_history_len=l, amax_compute_algo="max")
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        bench(custom_fused_model, "te-cus-fused", data, use_fp8=True)
                    print('|||||||')

            torch.cuda.empty_cache()
            print('===')

        if bwd:
            print('fwd+bwd')
            # print(baseline_model.mlp.

            # data = torch.rand(batch_size, config.hidden_size, dtype=torch.float16).cuda()
            # dy = torch.rand(batch_size, config.hidden_size, dtype=torch.float16).cuda()
            # data = torch.rand(batch_size, config.hidden_size).cuda()
            # dy = torch.rand(batch_size, config.hidden_size).cuda()
            # print('shape:', data.shape)

            # bench_bwd(baseline_model, "hf-llama", data, dy)
            # bench_bwd(naive_te_model, "te-naive", data)
            # bench_bwd(fused_model, "te-fused", data, dy)

            if enable_fp16:
                data = torch.rand(batch_size, config.hidden_size, dtype=torch.float16).cuda()
                dy = torch.rand(batch_size, config.hidden_size, dtype=torch.float16).cuda()
                # data.detach()
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                bench_bwd(baseline_model, "hf-llama", data, dy, use_fp16=True)
                # bench_bwd(naive_te_model, "te-naive", data, use_fp16=True)
                bench_bwd(fused_model, "te-fused", data, dy, use_fp16=True)
                bench_bwd(custom_fused_model, "te-cus-fused", data, dy, use_fp16=True)

            if enable_fp8:
                fp8_format = Format.HYBRID
                # for l in range(0, 16):
                for l in [1, 16]:
                    print('len,', l)
                    # fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                            # amax_history_len=l, amax_compute_algo="max")
                    # with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        # bench_bwd(naive_te_model, "te-naive", data, use_fp8=True)
                    fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                            amax_history_len=l, amax_compute_algo="max")
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        bench_bwd(fused_model, "te-fused", data, dy, use_fp8=True)
                    fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                            amax_history_len=l, amax_compute_algo="max")
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        bench_bwd(custom_fused_model, "te-cus-fused", data, dy, use_fp8=True)
                    print('|||||||')

            torch.cuda.empty_cache()
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
