"""Launch: python -m streamlit run app.py"""
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from cv_common.geometry import ply_bytes
from cv_common.fixtures import depth_fixture
from cv_common.ui import cloud_view

st.set_page_config(page_title='ML-Based-Vision', page_icon='🪐', layout='wide')
st.markdown('''<style>
.stApp {background:#080e17;color:#e5edf7} [data-testid="stSidebar"] {background:#101927}
.block-container {max-width:1360px;padding-top:2.2rem} h1 {letter-spacing:-.045em;font-weight:750!important}
[data-testid="stMetric"] {border:1px solid #28364a;border-radius:14px;padding:16px;background:#111c2b}
.eyebrow {font:12px monospace;letter-spacing:.2em;color:#efad76;margin-bottom:12px}
.intro {color:#9aacc1;font-size:17px;max-width:780px}.rule {border-top:1px solid #28364a;margin:24px 0}
div.stButton>button[kind="primary"] {background:#e4965e;color:#10151d;border:0}
</style>''', unsafe_allow_html=True)
st.sidebar.markdown('### 🪐 ORBITAL / VISION')
st.sidebar.caption('SPACE & COMPUTER VISION PROJECT')
project = 'Monocular vision'
st.sidebar.markdown('Monocular vision')
st.sidebar.markdown('---')
st.sidebar.caption('Rayan Rahman · Research workbench')
st.sidebar.caption('Synthetic samples verify the pipeline. They are not trained model predictions or real mission results.')
st.markdown('<div class="eyebrow">ORBITAL VISION LAB / EXPERIMENT WORKSPACE</div>', unsafe_allow_html=True)


def image_bytes(im):
    buf = io.BytesIO(); im.save(buf, format='PNG'); return buf.getvalue()


def safe_image(upload):
    with Image.open(upload) as im:
        if im.width * im.height > 24_000_000: raise ValueError('Use an image under 24 megapixels.')
        return im.convert('RGB')


def error_box(exc):
    st.error(str(exc))
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        st.code('python -m pip install -r requirements.txt')


from monocular_vision.pipeline import reconstruct, DepthEstimator
st.title('One image. A new perspective.')
st.markdown('<p class="intro">Estimate relative depth, back-project RGB pixels, and explore a colored 3D point cloud.</p>', unsafe_allow_html=True)
left, right = st.columns([1, 2.1], gap='large')
with left:
    source = st.radio('Input', ['Synthetic geometry example', 'Upload image'])
    hfov = st.slider('Assumed horizontal FOV', 30, 120, 75)
    stride = st.slider('Point sampling stride', 1, 10, 3)
    crop = st.slider('Exclude bottom portion', 0., .5, 0.)
    result = None
    if source == 'Synthetic geometry example':
        image, inverse = depth_fixture()
        result = reconstruct(image, inverse, hfov, stride, crop)
        result['metadata']['model'] = 'Synthetic inverse-depth fixture; no AI inference'
        st.caption('Designed scene and known depth for geometry verification.')
    else:
        upload = st.file_uploader('Scene image', type=['jpg', 'jpeg', 'png'])
        supplied = st.file_uploader('Optional inverse-depth array (.npy)', type=['npy'])
        st.caption('Without a depth array, this uses pretrained Depth Anything V2 Small. First use downloads its weights.')
        if st.button('Reconstruct scene', type='primary', disabled=not upload):
            try:
                image = safe_image(upload)
                image.thumbnail((1280, 1280)) if not supplied else None
                with st.spinner('Estimating depth and building point cloud…'):
                    inverse = np.load(io.BytesIO(supplied.getvalue()), allow_pickle=False) if supplied else DepthEstimator()(image)
                    result = reconstruct(image, inverse, hfov, stride, crop)
                    if supplied: result['metadata']['model'] = 'User-supplied inverse depth'
                st.session_state['mono'] = (upload.getvalue(), supplied.getvalue() if supplied else None, hfov, stride, crop, image, result)
            except Exception as exc: error_box(exc)
        saved = st.session_state.get('mono')
        if saved and upload and saved[:5] == (upload.getvalue(), supplied.getvalue() if supplied else None, hfov, stride, crop): image, result = saved[5:]
    st.caption('Relative scale only. Assumed camera intrinsics. Independent frames; no SLAM or global map.')
with right:
    if result is not None:
        a, b = st.columns(2); a.image(image, caption='RGB input', width='stretch'); b.image(result['depth_rgb'], caption='Inverse depth · warm = near', width='stretch')
        cloud_view(result['points'], result['colors'])
        a, b, c = st.columns(3); a.metric('Points', f'{len(result["points"]):,}'); b.metric('Camera FOV', f'{hfov}°'); c.metric('Scale', 'Relative')
        st.download_button('Download colored PLY', ply_bytes(result['points'], result['colors']), 'scene.ply')
        st.download_button('Download assumptions', json.dumps(result['metadata'], indent=2), 'scene_metadata.json')
    else: st.info('Upload a scene or use the synthetic geometry example.')
with st.expander('Process a dashcam video'):
    st.code('python -m monocular_vision.pipeline input/dashcam.mp4 --every 10 --max-frames 120')
    st.write('Exports a three-panel MP4 and depth/cloud files for each sampled frame. The point clouds are independent and are not fused into a map.')
