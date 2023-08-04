import torch.nn as nn
import torch
import os

def get_args():
    envvars = [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "NODE_COUNT",
        "HOSTNAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "AZ_BATCHAI_MPI_MASTER_NODE",
    ]
    args = dict(gpus_per_node=torch.cuda.device_count())
    missing = []
    for var in envvars:
        if var in os.environ:
            args[var] = os.environ.get(var)
            try:
                args[var] = int(args[var])
            except ValueError:
                pass
        else:
            missing.append(var)
    print(f"II Args: {args}")
    if missing:
        print(f"II Environment variables not set: {', '.join(missing)}.")
    return args


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(
        f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")
