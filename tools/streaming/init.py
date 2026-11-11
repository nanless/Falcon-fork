"""Create an explicitly untrained, reproducible streaming checkpoint."""
import argparse
import torch
if __package__:
    from .common import checkpoint_format_for_visual_mode, new_checkpoint
else:
    from common import checkpoint_format_for_visual_mode, new_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--channels", type=int, default=320)
parser.add_argument("--visual-mode", choices=("causal", "chunk_centered"), default="causal")
parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
args = parser.parse_args()
torch.manual_seed(args.seed)
model = FalconStreaming(channels=args.channels, dilations=tuple(args.dilations), visual_mode=args.visual_mode)
model_config = {"channels": args.channels, "dilations": args.dilations}
if args.visual_mode != "causal":
    model_config["visual_mode"] = args.visual_mode
new_checkpoint(args.output, {"format": checkpoint_format_for_visual_mode(args.visual_mode), "state_dict": model.state_dict(), "model_config": model_config,
                              "steps": 0, "training_status": "random_initialization", "seed": args.seed})
print(f"saved untrained random checkpoint to {args.output}")
