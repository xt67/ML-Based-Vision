import unittest
import numpy as np
from PIL import Image
from cv_common.geometry import intrinsics, backproject, ply_bytes, render_cloud
from cv_common.fixtures import depth_fixture
from monocular_vision.pipeline import reconstruct as mono, relative_depth

class GeometryTests(unittest.TestCase):

    def test_backprojection_and_ply(self):
        K = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1.0]])
        p, c = backproject(np.full((3, 3, 3), 120, np.uint8), np.full((3, 3), 4.0), K, 1)
        np.testing.assert_allclose(p[4], [0, 0, 4])
        np.testing.assert_allclose(p[0], [-2, -2, 4])
        data = ply_bytes(p, c)
        self.assertIn(b'element vertex 9', data)
        self.assertEqual(render_cloud(p, c).shape, (540, 800, 3))

    def test_depth_direction_and_constant_rejected(self):
        z, _ = relative_depth(np.arange(20, dtype=np.float32).reshape(4, 5))
        self.assertGreater(z[0, 0], z[-1, -1])
        with self.assertRaises(ValueError):
            relative_depth(np.ones((10, 10)))
        image, d = depth_fixture()
        result = mono(image, d)
        self.assertGreater(len(result['points']), 1000)
