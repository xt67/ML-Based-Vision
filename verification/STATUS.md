# Verification status

The geometry and video pipeline were tested with synthetic inverse depth. Pretrained Depth Anything inference has not been run in this restricted build environment. Model weights are downloaded on first use. Scale is relative, not metres, and frames are not fused into a map.

The independent repository test suite and synchronous Streamlit AppTest are executed before publishing. In this environment the UI test uses `--offline-test-loop` because local socket creation is blocked. That mocks only the unused asyncio event loop; it does not verify browser rendering.

The installed Torch/NumPy versions emit an ABI warning. Array conversion has a byte-buffer fallback; fresh installs use NumPy 1.26 as specified in requirements.txt.
