import unittest
import numpy as np
from trajectory import solve_time_np, predict_plate_crossing

class TestPitchPhysics(unittest.TestCase):
    def test_solve_time(self):
        # 50ft at 100fps, no acceleration
        y0 = 50.0
        vy0 = -100.0
        ay = 0.0
        # T = -y0 / vy0 = 0.5
        # Quadratic formula handles ay=0 cautiously or we choose small ay
        ay_small = 0.00001
        t = solve_time_np(y0, vy0, ay_small)
        self.assertAlmostEqual(t, 0.5, places=4)

    def test_prediction_no_accel(self):
        params = {
            'x0': 0.0, 'y0': 50.0, 'z0': 6.0,
            'vx0': 0.0, 'vy0': -100.0, 'vz0': 0.0,
            'ax': 0.0, 'ay': 0.0, 'az': 0.0
        }
        x, z = predict_plate_crossing(params, time=0.5)
        self.assertEqual(x, 0.0)
        self.assertEqual(z, 6.0)

    def test_prediction_with_gravity(self):
        params = {
            'x0': 0.0, 'y0': 50.0, 'z0': 6.0,
            'vx0': 0.0, 'vy0': -100.0, 'vz0': 0.0,
            'ax': 0.0, 'ay': 0.0, 'az': -32.2
        }
        # T = 0.5
        # z = 6.0 + 0*0.5 + 0.5 * (-32.2) * (0.5^2) = 6.0 - 4.025 = 1.975
        x, z = predict_plate_crossing(params, time=0.5)
        self.assertEqual(x, 0.0)
        self.assertAlmostEqual(z, 1.975)

if __name__ == '__main__':
    unittest.main()
