import numpy as np
for dt in (np.uint8, np.float32):
    frames = np.full((5, 96, 96), 128, dtype=dt)
    out = (frames - 0.0) / 255.0
    out2 = (out - 0.421) / 0.165
    print(f"stored {np.dtype(dt).name:8s} -> after Normalize(0,255): {out.dtype}, after second: {out2.dtype}")
