import tempfile, unittest
from pathlib import Path
import cv2
import numpy as np
from cv_common.fixtures import depth_fixture
from monocular_vision.pipeline import process_video

class WorkflowTests(unittest.TestCase):

    def test_video_outputs_and_sampling(self):
        with tempfile.TemporaryDirectory(prefix='space-cv-test-') as temp:
            root = Path(temp)
            video = root / 'input.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 12, (160, 120))
            self.assertTrue(writer.isOpened())
            image, inv = depth_fixture(160, 120)
            for _ in range(6):
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            writer.release()
            frames = process_video(video, lambda im: inv, root / 'output', every=2, max_frames=3)
            self.assertEqual([r['source_frame'] for r in frames], [0, 2, 4])
            self.assertTrue((root / 'output/frame-000000/point_cloud.ply').is_file())
            cap = cv2.VideoCapture(str(root / 'output/reconstruction.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 6, places=1)
            cap.release()
