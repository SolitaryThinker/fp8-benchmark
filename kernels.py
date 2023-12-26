#! /usr/bin/python

import transformer_engine.pytorch as te
from transformer_engine.pytorch import LayerNormMLP
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import msamp
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
        LlamaMLP,
        LlamaRMSNorm
)
import numpy as np
import matplotlib.pyplot as plt

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


def test_casting():
    print('test casting')
    def test_cast_fp8():
        """Test the cast_to_fp8 and cast_from_fp8 functions in TypeCast."""
        torch.manual_seed(100)
        input_fp16 = torch.rand((4, 4), dtype=torch.float16, device='cuda')
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        output_fp8 = TypeCast.cast_to_fp8(input_fp16, meta)
        print(meta.amax)
        output_fp16 = TypdeCast.cast_from_fp8(output_fp8, meta, Dtypes.kfloat16)
        print(meta.amax)

        assert torch.allclose(input_fp16, output_fp16, 0, 0.1)

    def test_cast_fp16():
        """Test the cast_to_fp16 and cast_from_fp16 functions in TypeCast."""
        torch.manual_seed(100)
        input_fp32 = torch.rand((4, 4), device='cuda')
        meta = ScalingMeta(Dtypes.kfloat16)
        output_fp16 = TypeCast.cast_to_fp16(input_fp32, meta)
        output_fp32 = TypeCast.cast_from_fp16(output_fp16, meta, Dtypes.kfloat32)

        assert torch.allclose(input_fp32, output_fp32, 0, 1e-03)
    test_cast_fp8()
    test_cast_fp16()
    print('done test casting')

def test_scaling_factor():
    def test_compute_scaling_factor(self):
        """Test compute_scaling_factor in ScalingMeta."""
        amax = torch.zeros([], device='cuda')
        scale = torch.ones((), device='cuda')
        fp_max = Floating.qfp_max[Dtypes.kfloat8_e4m3]
        margin = 0
        scale.copy_(ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin))

        assert scale.item() == 1.0

        # 2^(floor(log2(448.0/10)))=32
        amax = torch.tensor(10, device='cuda')
        scale = torch.ones((), device='cuda')
        scale.copy_(ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin))
        assert scale.item() == 32

        # 1/(2^abs(floor(log2(448.0/10000))))
        amax = torch.tensor(10000, device='cuda')
        scale = torch.ones((), device='cuda')
        scale.copy_(ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin))
        assert scale.item() == 1.0 / 32

class BaselineNet(nn.Module):

    def __init__(self, config):
        super(BaselineNet, self).__init__()
        self.layernorm = LlamaRMSNorm(config.hidden_size)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.mlp(x)
        return x

class FusedNet(nn.Module):
    pass



def benchmark_kernels():
    print('benchmark_kernels')
    num_trials = 10
    warm_up = 10

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

    for size, dtype, batch_size, use_fp8 in itertools.product(sizes, dtypes,
            batch_sizes, [False, True]):
        print('h={}, inter={}, dtype={}, batch={}, fp8={}'.format(size[0],
            size[1], dtype, batch_sizes, use_fp8))
        hidden_size = size[0]
        inter_size = size[1]

        config = LlamaConfig(hidden_size=hidden_size,
                intermediate_size=inter_size)

        with te.fp8_autocast(enabled=use_fp8):
            baseline_model = BaselineNet(config).cuda()
            baseline_model.train()
            data = torch.rand(batch_size, config.hidden_size).cuda()
            # print('shape:', data.shape)

            for _ in range(warm_up):
                output = baseline_model(data)

            for i in range(num_trials):
                output = baseline_model(data)
                print(i)

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

            for _ in range(warm_up):
                output = fused_model(data)

            for i in range(num_trials):
                output = fused_model(data)
                print(i)


    return


def run_test():
    # test_casting()
    # test_gemm()
    # test_scaling_factor()
    benchmark_kernels()

print('testing msamp')
run_test()
print('done run test')
