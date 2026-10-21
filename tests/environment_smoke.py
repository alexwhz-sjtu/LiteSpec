"""Verify the installed LiteSpec dependencies, CUDA kernels, and NCCL."""
import argparse
import importlib
import importlib.metadata
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true')
args = parser.parse_args()
directory = Path(__file__).resolve().parent
os.sys.path.insert(0, str(directory.parent))

if args.distributed:
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    value = torch.tensor([float(rank)], device=f'cuda:{rank}')
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() - 1) / 2
    assert value.item() == expected, (rank, value.item(), expected)
    dist.barrier()
    if rank == 0:
        result = {'nccl': 'passed', 'ranks': dist.get_world_size(), 'sum': value.item()}
        (directory / 'environment_nccl.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()
else:
    assert torch.cuda.is_available(), 'CUDA unavailable'
    result = {'python': os.sys.version.split()[0], 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'gpus': [], 'imports': {}, 'packages': {}}
    for package in ['litespec', 'torch', 'torchvision', 'torchaudio', 'transformers',
                    'sglang', 'sgl-kernel', 'causal-conv1d', 'flash-linear-attention']:
        result['packages'][package] = importlib.metadata.version(package)
    for module in ['torchvision', 'torchaudio', 'transformers', 'sglang', 'sgl_kernel',
                   'flashinfer', 'fla', 'causal_conv1d', 'litespec.core.dlite',
                   'scripts.dlite.train_sft', 'scripts.dlite.train_teacher',
                   'scripts.dlite.train_two_stage']:
        importlib.import_module(module)
        result['imports'][module] = 'passed'
        print('IMPORT_OK', module, flush=True)
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            assert torch.cuda.is_bf16_supported()
            x = torch.randn(128, 128, device=f'cuda:{index}', dtype=torch.bfloat16)
            y = x @ x.T
            assert torch.isfinite(y).all().item()
            torch.cuda.synchronize()
            result['gpus'].append({'index': index, 'name': torch.cuda.get_device_name(index),
                                   'capability': list(torch.cuda.get_device_capability(index)),
                                   'bf16_matmul': 'passed'})
    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(2, 32, 64, device='cuda', dtype=torch.bfloat16) * 0.1
    weight = torch.randn(32, 4, device='cuda', dtype=torch.bfloat16) * 0.1
    x.requires_grad_()
    weight.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    actual = causal_conv1d_fn(x, weight)
    expected = F.conv1d(reference_x, reference_weight.unsqueeze(1), padding=3, groups=32)[..., :64]
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0.03, atol=0.005)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0.03, atol=0.005)
    result['causal_conv1d_gpu'] = 'forward_backward_passed'
    (directory / 'environment_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
