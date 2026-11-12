import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tools.streaming.robust_schedule import global_gradient_norm


def _rank(rank, init_file):
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    try:
        local = torch.tensor(15.0 if rank == 0 else float("nan"))
        result = global_gradient_norm(local)
        assert torch.isinf(result) and result.item() > 1000
        local = torch.tensor(15.0 if rank == 0 else 7182.0)
        assert global_gradient_norm(local).item() == 7182.0
    finally:
        dist.destroy_process_group()


def test_both_ranks_reject_remote_nan_and_large_norm(tmp_path):
    mp.spawn(_rank, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
