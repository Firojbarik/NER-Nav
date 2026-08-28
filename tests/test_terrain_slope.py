import unittest

import numpy as np
from affine import Affine

from scripts.rebuild_terrain_full import _slope_from_geographic_dem


class TestGeographicTerrainSlope(unittest.TestCase):
    def test_geographic_gradient_is_converted_to_metres(self):
        # A 9 m rise per 90 m pixel is a 5.71 degree slope, not 90 degrees.
        elevation = np.tile(np.arange(5, dtype=float)[:, None] * 9.0,
                            (1, 5))
        slope = _slope_from_geographic_dem(
            elevation,
            Affine(1 / 1200, 0, 90,
                   0, -1 / 1200, 27),
        )
        expected = np.degrees(np.arctan(9 / (110574 / 1200)))
        self.assertTrue(np.allclose(slope[1:-1, 1:-1],
                                    expected,
                                    atol=0.05))

    def test_nodata_is_not_treated_as_extreme_slope(self):
        elevation = np.ma.array(np.ones((5, 5)) * 100,
                                 mask=np.zeros((5, 5), dtype=bool))
        elevation.mask[2, 2] = True
        slope = _slope_from_geographic_dem(
            elevation,
            Affine(1 / 1200, 0, 90,
                   0, -1 / 1200, 27),
        )
        self.assertTrue(np.isnan(slope[2, 2]))
        self.assertTrue(np.nanmax(slope) < 1.0)


if __name__ == "__main__":
    unittest.main()
