import unittest
from analysis.motion_coherence.data_structures import Fixed, SessionData
from analysis.motion_coherence.analysis import calculate_temporal_success, smooth_temporal_data
import numpy as np

class TestTemporalAnalysis(unittest.TestCase):
    def test_calculate_temporal_success(self):
        # Session 1: Coh 0.1 -> [Fail, Success]
        s1 = SessionData(
            timestamp=1,
            coherences=np.array([0.1, 0.1]),
            successes=np.array([0, 1]),
            directions=np.array([0, 0])
        )
        
        # Session 2: Coh 0.1 -> [Success, Success]
        s2 = SessionData(
            timestamp=2,
            coherences=np.array([0.1, 0.1]),
            successes=np.array([1, 1]),
            directions=np.array([0, 0])
        )

        experiments = [Fixed(session=s1), Fixed(session=s2)]
        
        matrix, cohs = calculate_temporal_success(experiments)
        
        # Expect 1 coherence
        self.assertEqual(len(cohs), 1)
        self.assertEqual(cohs[0], 0.1)
        
        # Matrix shape: (1, 2 trials)
        self.assertEqual(matrix.shape, (1, 2))
        
        # Trial 0: 0 + 1 / 2 = 0.5
        # Trial 1: 1 + 1 / 2 = 1.0
        np.testing.assert_array_almost_equal(matrix[0], np.array([0.5, 1.0]))

    def test_smooth_temporal_data(self):
        # 1 coherence, 5 trials
        matrix = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]])
        
        # Window 3, 'valid' mode
        # Output length: 5 - 3 + 1 = 3
        # smoothed[0]: mean([0.0, 0.0, 1.0]) = 1/3
        # smoothed[1]: mean([0.0, 1.0, 1.0]) = 2/3
        # smoothed[2]: mean([1.0, 1.0, 1.0]) = 1.0
        
        smoothed = smooth_temporal_data(matrix, window_size=3)
        
        self.assertEqual(smoothed.shape, (1, 3)) # Shape changes
        np.testing.assert_array_almost_equal(smoothed[0], np.array([1/3, 2/3, 1.0]))

if __name__ == '__main__':
    unittest.main()
