import time
import numpy as np
import torch
import os
from torch.utils.cpp_extension import load

def timer_decorator(func):
    '''计算算子执行时间'''
    def wrapper(*args, **kwargs):
        times = list()
        res = None
        # GPU warm up
        for _ in range(WARM_UP_ROUND):
            func(*args, **kwargs)
        for _ in range(TEST_ROUND):
            # sync the threads to get accurate cuda running time
            torch.cuda.synchronize(device=DEVICE)
            start_time = time.time()
            res = func(*args, **kwargs)
            torch.cuda.synchronize(device=DEVICE)
            end_time = time.time()
            times.append((end_time-start_time)*1e6)
        return np.mean(times), res
    return wrapper

@timer_decorator
def run_cuda():
    cuda_module.matmul(cuda_c, A, B, N, K, M)
    return cuda_c

@timer_decorator
def run_torch():
    torch_c = torch.matmul(A, B)#.contiguous()
    return torch_c

# 全局变量
DEVICE = "cuda:0"
WARM_UP_ROUND = 1
TEST_ROUND = 1
OPERATOR_NAME = "matmul"

# 初始化数据
N = 1025
K = 10240
M = 1025
MAX = 10
A = torch.rand(N, K, device=DEVICE) * 2*MAX - MAX
B = torch.rand(K, M, device=DEVICE) * 2*MAX - MAX
# N = 2
# K = 2
# M = 2
# A = torch.tensor([[1, 2], [3, 4]], device=DEVICE, dtype=torch.float32)
# B = torch.tensor([[1, 2], [3, 4]], device=DEVICE, dtype=torch.float32)
cuda_c = torch.zeros(N, M, device=DEVICE)
torch_c = cuda_c.clone()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    cuda_module = load(name=OPERATOR_NAME,
                        # extra_cflags=['-t recompact'],
                        extra_include_paths=["./include"],
                        sources=["./kernel/" + OPERATOR_NAME + ".cpp", "./kernel/" + OPERATOR_NAME + ".cu"],
                        # build_directory=r".\cache",
                        verbose=False)

    cuda_time, cuda_res = run_cuda()
    torch_time, torch_res = run_torch()

    print("Current operator:", OPERATOR_NAME)
    # print(cuda_res)
    # print(torch_res)
    print(cuda_res.flatten()[0:10])
    print(torch_res.flatten()[0:10])
    print("Custom operator strong validation:", cuda_res.equal(torch_res))
    print("Custom operator weak validation: ", torch.allclose(cuda_res, torch_res, atol=1e-1, rtol=1e-4))
    print("Cuda time:   {:.3f}us".format(cuda_time))
    print("Torch time:  {:.3f}us".format(torch_time))
