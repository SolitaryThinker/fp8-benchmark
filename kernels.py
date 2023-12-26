#! /usr/bin/python

# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
import torch
import itertools
import msamp

import numpy as np
import matplotlib.pyplot as plt

from msamp.common.dtype import Dtypes
from msamp.operators.gemm import Gemm
from msamp.operators.arithmetic import Arithmetic
from msamp.common.tensor import ScalingMeta
from msamp.common.tensor import TypeCast


def _check_scaling_tensor(scaling_tensor1, scaling_tensor2):
    assert(torch.all(torch.eq(scaling_tensor1.value, scaling_tensor2.value)))
    assert(torch.all(torch.eq(scaling_tensor1.meta.scale, scaling_tensor2.meta.scale)))
    assert(torch.all(torch.eq(scaling_tensor1.meta.scale_inv, scaling_tensor2.meta.scale_inv)))
    assert(torch.all(torch.eq(scaling_tensor1.meta.amax, scaling_tensor2.meta.amax)))

def test_add_to_fp8():
    """Test the function Arithmetic.add_to_fp8()."""
    torch.manual_seed(100)
    sizes = list(range(1024, 8193, 1024))
    dtypes = [torch.float16]
    qtypes = [Dtypes.kfloat8_e4m3]
    # sizes = list(range(1024, 8193, 1024))
    # dtypes = [torch.float16, torch.bfloat16, torch.float32]
    # qtypes = [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2]
    for i, j, dtype, qtype, in itertools.product(sizes, sizes, dtypes, qtypes):
        print(i, j, dtype, qtype)
        size = (i, j)
        input1 = torch.rand(size, dtype=dtype, device='cuda')
        scaling_tensor1 = input1.cast(qtype)
        scaling_tensor2 = input1.cast(qtype)

        for i in range(10):
            print(i)
            input2 = torch.rand(size, dtype=dtype, device='cuda')
            meta = scaling_tensor1.meta
            Arithmetic.add_to_fp8(scaling_tensor1.value, meta, input2)
            scaling_tensor2.copy_((scaling_tensor2.to(dtype) + input2).cast(qtype, meta=scaling_tensor2.meta))
            _check_scaling_tensor(scaling_tensor1, scaling_tensor2)

def test_gemm():
    sizes = list(range(1024, 8193*2, 1024*2))
    # sizes = list(range(1024, 1024*4, 1024*2))
    sizes = [4096, 5120, 8192]
    # list of square matrix sizes to test

    dtypes = [Dtypes.kbfloat16]
    qtypes = [Dtypes.kfloat8_e4m3]
    #dtypes = [Dtypes.kfloat16, Dtypes.kbfloat16, Dtypes.kfloat32]
    #qtypes = [Dtypes.kfloat8_e4m3, Dtypes.kfloat8_e5m2]
    # get the string of the dtypes and qtypes
    input_sizes = []
    # create a map to hold the results for the itertools.product
    results = {}
    r2= {}
    for i, dtype, qtype, in itertools.product(sizes, dtypes, qtypes):
        r2[str(i*i)+'cast_to_fp8'] = []
        r2[str(i*i)+'gemm'] = []
    for i, dtype, qtype, in itertools.product(sizes, dtypes, qtypes):
        input_sizes.append(i*i)
        # set the result
        print(i, i, dtype, qtype)
        tensorA = torch.ones((i, i), dtype=torch.float32, device='cuda')
        tensorB = torch.ones((i, i), dtype=torch.float32, device='cuda')
        scaling_tensorA = tensorA.cast(qtype)
        scaling_tensorB = tensorB.cast(qtype)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(100):
            out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, dtype)

        elapsed_times = []
        start.record()
        num = 5000
        for _ in range(num):
            out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, dtype)
        end.record()
        torch.cuda.synchronize()
        el = start.elapsed_time(end)
        results[(i*i, dtype.name, qtype.name)] = el / num
        r2[str(i*i)+'gemm'] = el / num
        #elapsed_times.append(el)
        #elapsed_times = np.array(elapsed_times)
        print('fp8_gemm mean', el/num)
        #print('std', np.std(elapsed_times))
        # print('fp8_gemm: ', np.sum(elapsed_times) / 10000)
        #
        # casting ops

        torch.manual_seed(100)
        input_fp16 = torch.rand((i, i), dtype=torch.float16, device='cuda')
        start.record()
        for _ in range(num):
            meta = ScalingMeta(Dtypes.kfloat8_e4m3)
            output_fp8 = TypeCast.cast_to_fp8(input_fp16, meta)
        end.record()
        torch.cuda.synchronize()
        el = start.elapsed_time(end)
        results[(i*i, qtype.name, torch.float16)] = el / num
        print('cast_to_fp8 mean', el/num)
        r2[str(i*i)+'cast_to_fp8'].append(el / num)

        start.record()
        for _ in range(num):
            meta = ScalingMeta(Dtypes.kfloat8_e4m3)
            #output_fp8 = TypeCast.cast_to_fp8(input_fp16, meta)
            output_fp16 = TypeCast.cast_from_fp8(output_fp8, meta, Dtypes.kfloat16)
        end.record()
        torch.cuda.synchronize()
        el = start.elapsed_time(end)
        results[(i*i, torch.float16, qtype.name)] = el / num
        print('cast_from_fp8', el/num)



    # plot results
    # create a figure

    fig = plt.figure()
    # plot the data
    # plt.plot(input_sizes, elapsed_times)
    # plot that data with error bars
    # iterate through results and plot using scatter plot and label with the 1st and 2nd element of the key
    for key, value in results.items():
        plt.scatter(key[0], value, label=(key[1], key[2]), s=1)
    #plot r2 on a new page
    fig2 = plt.figure()
    for key, value in r2.items():
        plt.scatter(key[0], value, label=key[0], s=1)
    # add a legend
    plt.legend()
    # use smaller dots for the plot

    # save the plot as a pdf
    plt.savefig('gemm.pdf')

    elapsed_times = []
    start.record()
    for _ in range(10000):
        out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, Dtypes.kfloat32)

    end.record()
    torch.cuda.synchronize()
    el = start.elapsed_time(end)
    # elapsed_times.append(el)
    # elapsed_times = np.array(elapsed_times)
    print('mean 2', el/ 10000)
    # print('std', np.std(elapsed_times))

    for _ in range(100):
        expected = torch.matmul(tensorB, tensorA.t())
    elapsed_times = []
    start.record()
    for _ in range(10000):
        # out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, Dtypes.kfloat32)
        expected = torch.matmul(tensorB, tensorA.t())

    end.record()
    torch.cuda.synchronize()
    el = start.elapsed_time(end)
    # elapsed_times.append(el)
    # elapsed_times = np.array(elapsed_times)
    print('mean 3', el/ 10000)
    # print('std', np.std(elapsed_times))

    start.record()
    expected = torch.matmul(tensorB, tensorA.t())
    end.record()
    print(start.elapsed_time(end))
    assert (out.equal(expected))

    # out = torch.ones((3000, 4000), dtype=torch.float32, device='cuda')
    # out = Gemm.fp8_gemm(scaling_tensorA, scaling_tensorB, Dtypes.kfloat32, out=out)
    # assert (out.equal(expected))
# test_add_to_fp8()
# print('done')

def test_casting():
    print('test casting')
    def test_cast_fp8():
        """Test the cast_to_fp8 and cast_from_fp8 functions in TypeCast."""
        torch.manual_seed(100)
        input_fp16 = torch.rand((4, 4), dtype=torch.float16, device='cuda')
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        output_fp8 = TypeCast.cast_to_fp8(input_fp16, meta)
        print(meta.amax)
        output_fp16 = TypeCast.cast_from_fp8(output_fp8, meta, Dtypes.kfloat16)
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

def benchmark_kernels():
    print('benchmark_kernels')

def run_test():
    # test_casting()
    # test_gemm()
    # test_scaling_factor()
    benchmark_kernels()

print('testing te')
run_test()
print('done run test')

"""

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)

# Create an FP8 recipe. Note: All input args are optional.
# fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
# fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
# with te.fp8_autocast(enabled=False, fp8_recipe=fp8_recipe):

total_time = 0
for _ in range(10):
    inp = torch.randn(hidden_size, in_features, device="cuda")
    out = model(inp)

for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    inp = torch.randn(hidden_size, in_features, device="cuda")
    start.record()
    out = model(inp)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end)
    # print(t)
    total_time += t


loss = out.sum()
loss.backward()
print(loss)
print(total_tim / 100)
"""
