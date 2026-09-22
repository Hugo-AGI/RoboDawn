"""Check camera conventions and geometry used by visual policy grounding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL"))
if AVAILABLE:
    import numpy as np

    MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation/policies/vlm_agent/grounding.py"
    spec = importlib.util.spec_from_file_location("vlm_grounding", MODULE_PATH)
    grounding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grounding)


@unittest.skipUnless(AVAILABLE, "numpy and Pillow are required")
class GroundingTest(unittest.TestCase):
    def test_wrist_overlay_draws_world_and_jaw_axes_without_mutating_input(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        calibration = {"intrinsic_matrix": [[200, 0, 160], [0, 200, 120], [0, 0, 1]],
                       "extrinsic_matrix": np.eye(4).tolist()}
        marked = grounding.annotate_wrist(rgb, calibration, [0, 0, -1], [1, 0, 0, 0])
        self.assertTrue(np.any(marked))
        self.assertFalse(np.any(rgb))
        self.assertEqual(marked.shape, rgb.shape)

    def test_tool_axes_and_offset(self):
        yaw90 = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        down = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]
        np.testing.assert_allclose(grounding.tool_forward(yaw90), [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(grounding.tool_forward(down), [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(grounding.tool_tip([1, 2, 3], yaw90), [1, 2.145, 3])
        np.testing.assert_allclose(grounding.tool_tip([1, 2, 0.91], down), [1, 2, 0.765])
        np.testing.assert_allclose(grounding.tool_forward(np.asarray(yaw90) * 2), [0, 1, 0], atol=1e-12)
        with self.assertRaises(ValueError):
            grounding.tool_forward([0, 0, 0, 0])

    def test_usd_projection_signs_and_unequal_focal_lengths(self):
        k = np.array([[100, 0, 320], [0, 200, 240], [0, 0, 1]])
        uv, valid = grounding.project_points([[0, 0, -2], [1, 1, -2], [-1, -1, -2]], k, np.eye(4))
        np.testing.assert_allclose(uv, [[320, 240], [370, 140], [270, 340]])
        self.assertTrue(valid.all())

    def test_camera_rotation_translation_and_invalid_depth(self):
        transform = np.array([[1, 0, 0, 5], [0, 0, -1, 6], [0, 1, 0, 7], [0, 0, 0, 1]])
        k = np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]])
        uv, valid = grounding.project_points([[5, 8, 7], [6, 8, 8], [5, 5, 7], [5, 6, 7]], k, transform)
        np.testing.assert_allclose(uv[:2], [[320, 240], [370, 190]])
        np.testing.assert_array_equal(valid, [True, True, False, False])
        self.assertTrue(np.isnan(uv[2:]).all())

    def test_annotation_preserves_source_and_corrects_env_origin(self):
        image = np.full((480, 640, 3), 100, dtype=np.uint8)
        rotation = np.array([[1, 0, 0], [0, np.cos(np.pi / 6), -0.5], [0, 0.5, np.cos(np.pi / 6)]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0, -0.41, 1.308]
        calibration = {"intrinsic_matrix": [[288.13, 0, 320], [0, 336.46, 240], [0, 0, 1]], "extrinsic_matrix": transform}
        states = {"left": SimpleNamespace(pos=np.array([-0.3, -0.2, 0.91]), quat=[0.5**0.5, 0, 0.5**0.5, 0])}
        result = grounding.annotate_head(image, calibration, states)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue((image == 100).all())
        altered = np.any(result != image, axis=2).mean()
        self.assertGreater(altered, 0.005)
        self.assertLess(altered, 0.15)
        shifted = transform.copy()
        shifted[:3, 3] += [5, -8, 2]
        shifted_calibration = dict(calibration, extrinsic_matrix=shifted, env_origin=[5, -8, 2])
        np.testing.assert_array_equal(result, grounding.annotate_head(image, shifted_calibration, states))

    def test_resized_image_calibration(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        transform = np.eye(4)
        transform[2, 3] = 2
        k = np.array([[300.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
        original = {"intrinsic_matrix": k, "extrinsic_matrix": transform, "image_size": [640, 480]}
        scaled_k = k.copy()
        scaled_k[:2] *= 0.5
        scaled = {"intrinsic_matrix": scaled_k, "extrinsic_matrix": transform}
        np.testing.assert_array_equal(grounding.annotate_head(image, original, {}), grounding.annotate_head(image, scaled, {}))


if __name__ == "__main__":
    unittest.main()
