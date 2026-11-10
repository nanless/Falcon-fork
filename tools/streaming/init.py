"""Create an explicitly untrained, reproducible streaming checkpoint."""
import argparse
import torch
if __package__:
    from .common import CHECKPOINT_FORMAT, new_checkpoint
else:
    from common import CHECKPOINT_FORMAT, new_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
model = FalconStreaming()
new_checkpoint(args.output, {"format": CHECKPOINT_FORMAT, "state_dict": model.state_dict(),
                              "steps": 0, "training_status": "random_initialization", "seed": args.seed})
print(f"saved untrained random checkpoint to {args.output}")
