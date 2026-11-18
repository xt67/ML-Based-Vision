# ML Based Vision

Estimate relative monocular depth with Depth Anything V2 Small and turn RGB pixels into a colored 3D point cloud.

Built for **Rayan Rahman** as part of the Space And Computer Vision Project.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The app includes a labeled synthetic example; no downloads are needed to explore that example after installing the base dependencies. On macOS/Linux, use `.venv/bin/python`.

## Workflow


```powershell
python -m monocular_vision.pipeline input/street.jpg
python -m monocular_vision.pipeline input/dashcam.mp4 --every 10 --max-frames 120
python -m monocular_vision.pipeline input/street.jpg --depth-npy input/inverse_depth.npy
```

First inference downloads `depth-anything/Depth-Anything-V2-Small-hf`. You can point `--model` at a previously downloaded model folder and add `--local-only`.

Outputs: colored PLY, depth PNG, preview PNG, float depth array and assumptions JSON. Video mode creates a three-panel MP4 and per-frame exports, with duration based on the actual source FPS and sampling interval.

Depth is **relative inverse depth**, mapped through an explicitly assumed 4–60 range. Those units are not metres. Horizontal FOV defaults to 75 degrees and is configurable; the reference project used 90 degrees. An input `.npy` must be a finite 2D inverse-depth array at the image's exact resolution. A constant map is rejected. No motion estimation, SLAM or frame fusion is claimed. Per-frame normalization can cause temporal scale drift.


## Verification and limitations

The geometry and video pipeline were tested with synthetic inverse depth. Pretrained Depth Anything inference has not been run in this restricted build environment. Model weights are downloaded on first use. Scale is relative, not metres, and frames are not fused into a map.

Synthetic examples are pipeline checks, not real-world benchmark results. See `verification/STATUS.md`.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m tools.verify_ui
```

## Attribution

Reference project by tubakhxn: https://github.com/tubakhxn/MLbasedvision

The original source snapshot and MIT license are retained under `references/`. Upstream README measurements describe upstream runs, not this implementation.

Model source: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
