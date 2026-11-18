import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cv_common.geometry import intrinsics, backproject, save_cloud, render_cloud

MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'


class DepthEstimator:
    def __init__(self, model_path=MODEL_ID, device='cpu', local_only=False):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=local_only)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=local_only).to(device).eval()
        self.device = device

    def __call__(self, image):
        import torch
        from torch.nn import functional as F
        from cv_common.tensors import tensor_array
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            depth = F.interpolate(output.unsqueeze(1), size=(image.height, image.width), mode='bicubic', align_corners=False)[0, 0]
        return tensor_array(depth).astype(np.float32)


def relative_depth(inverse, near=4., far=60.):
    inverse = np.asarray(inverse, dtype=np.float32)
    valid = np.isfinite(inverse)
    if inverse.ndim != 2 or not valid.any() or not 0 < near < far:
        raise ValueError('Expected a finite 2D inverse-depth map and 0 < near < far.')
    lo, hi = np.percentile(inverse[valid], [2, 98])
    if hi - lo < 1e-8:
        raise ValueError('Depth map has no variation; reconstruction is ambiguous.')
    normalized = np.clip((inverse - lo) / (hi - lo), 0, 1)
    z = 1 / (normalized * (1 / near - 1 / far) + 1 / far)
    z[~valid] = np.nan
    return z, normalized


def reconstruct(image, inverse, hfov=75., stride=3, crop_bottom=0., near=4., far=60.):
    image = image.convert('RGB'); rgb = np.asarray(image)
    inverse = np.asarray(inverse, dtype=np.float32)
    if inverse.shape != rgb.shape[:2]:
        raise ValueError('Depth map and image dimensions differ.')
    if not 0 <= crop_bottom < .8:
        raise ValueError('Bottom crop must be in [0, 0.8).')
    z, normalized = relative_depth(inverse, near, far)
    valid = np.isfinite(z)
    if crop_bottom > 0:
        valid[int(image.height * (1 - crop_bottom)):] = False
    points, colors = backproject(rgb, z, intrinsics(image.width, image.height, hfov), stride, valid)
    depth_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.nan_to_num(normalized) * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return {'points': points, 'colors': colors, 'depth': z, 'depth_rgb': depth_rgb,
            'metadata': {'model': MODEL_ID, 'scale': 'relative; not metres', 'hfov_degrees': hfov,
                         'assumed_near': near, 'assumed_far': far, 'points': len(points), 'frame_fusion': False}}


def save_result(result, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    save_cloud(out / 'point_cloud.ply', result['points'], result['colors'])
    Image.fromarray(result['depth_rgb']).save(out / 'depth.png')
    Image.fromarray(render_cloud(result['points'], result['colors'])).save(out / 'cloud.png')
    np.save(out / 'relative_depth.npy', result['depth'])
    (out / 'metadata.json').write_text(json.dumps(result['metadata'], indent=2), encoding='utf-8')


def process_video(path, estimator, output, every=10, max_frames=120, hfov=75., stride=4):
    if every < 1 or max_frames < 1: raise ValueError('Frame interval and limit must be positive.')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError('Video could not be opened.')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0: fps = 30.
    writer = cv2.VideoWriter(str(out / 'reconstruction.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps / every, (1440, 480))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('MP4 writer unavailable in this OpenCV build.')
    n = 0; idx = 0; manifest = []
    try:
        while n < max_frames:
            ok, frame = cap.read()
            if not ok: break
            if idx % every == 0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (max(2, int(w * min(1, 768 / w))), max(2, int(h * min(1, 768 / w)))))
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = reconstruct(image, estimator(image), hfov, stride)
                save_result(result, out / f'frame-{idx:06d}')
                panels = [np.asarray(image), result['depth_rgb'], render_cloud(result['points'], result['colors'])]
                panels = [cv2.resize(v, (480, 480)) for v in panels]
                composite = np.concatenate(panels, axis=1)
                for x, title in [(8, 'RGB'), (488, 'Relative inverse depth'), (968, 'Independent 3D frame')]:
                    cv2.putText(composite, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                manifest.append({'source_frame': idx, 'seconds': idx / fps, **result['metadata']})
                n += 1; print(f'Processed frame {idx}', flush=True)
            idx += 1
    finally:
        cap.release(); writer.release()
    if not n: raise ValueError('Video contains no decodable frames.')
    (out / 'frames.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description='Depth Anything V2 -> relative RGB-D point cloud.')
    p.add_argument('input'); p.add_argument('--output', default='outputs/monocular')
    p.add_argument('--model', default=MODEL_ID); p.add_argument('--local-only', action='store_true')
    p.add_argument('--depth-npy', help='Use supplied relative inverse depth instead of running a model.')
    p.add_argument('--hfov', type=float, default=75); p.add_argument('--stride', type=int, default=3)
    p.add_argument('--every', type=int, default=10); p.add_argument('--max-frames', type=int, default=120)
    a = p.parse_args()
    if not Path(a.input).is_file(): p.error('Input does not exist.')
    video = Path(a.input).suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}
    if video and a.depth_npy: p.error('--depth-npy supports still images only.')
    estimator = None if a.depth_npy else DepthEstimator(a.model, local_only=a.local_only)
    if video: process_video(a.input, estimator, a.output, a.every, a.max_frames, a.hfov, a.stride)
    else:
        image = Image.open(a.input).convert('RGB')
        inverse = np.load(a.depth_npy, allow_pickle=False) if a.depth_npy else estimator(image)
        result = reconstruct(image, inverse, a.hfov, a.stride)
        if a.depth_npy: result['metadata']['model'] = 'User-supplied inverse-depth map; provenance not verified'
        save_result(result, a.output)


if __name__ == '__main__': main()
