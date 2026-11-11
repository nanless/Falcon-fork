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
parser.add_argument("--synthesis-mode", choices=("nonnegative", "signed_residual"), default="nonnegative")
parser.add_argument("--subband-refiner", action="store_true")
parser.add_argument("--attention-history-chunks", type=int, default=0)
parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
args = parser.parse_args()
torch.manual_seed(args.seed)
model = FalconStreaming(channels=args.channels, dilations=tuple(args.dilations), visual_mode=args.visual_mode, synthesis_mode=args.synthesis_mode,
                        attention_history_chunks=args.attention_history_chunks,
                        subband_refiner=args.subband_refiner)
model_config = {"channels": args.channels, "dilations": args.dilations}
if args.visual_mode != "causal":
    model_config["visual_mode"] = args.visual_mode
if args.synthesis_mode != "nonnegative":
    model_config["synthesis_mode"] = args.synthesis_mode
if args.attention_history_chunks:
    model_config["attention_history_chunks"] = args.attention_history_chunks
if args.subband_refiner:
    model_config["subband_refiner"] = True
new_checkpoint(args.output, {"format": checkpoint_format_for_visual_mode(
    args.visual_mode, args.synthesis_mode, args.attention_history_chunks, args.subband_refiner), "state_dict": model.state_dict(), "model_config": model_config,
                              "steps": 0, "training_status": "random_initialization", "seed": args.seed})
print(f"saved untrained random checkpoint to {args.output}")
