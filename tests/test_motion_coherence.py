import unittest
import tempfile
import os
import numpy as np
from experiments.analysis.motion_coherence import text_into_coherences_and_successes
from tests.log_generator import generate_log_file, generate_log_file_with_weibull
from unittest.mock import patch

class TestMotionCoherence(unittest.TestCase):
    def test_weibull_parameter_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_filepath = os.path.join(tmpdir, "test-motion_coherence_fixed-12345.log")

            alpha_gt = 0.5
            beta_gt = 1.5

            coherences = np.repeat(np.linspace(0.1, 0.9, 10), 100)
            directions = [(0.1, 0.2)] * len(coherences)

            generate_log_file_with_weibull(log_filepath, coherences, directions, alpha_gt, beta_gt)

            from experiments.analysis.motion_coherence import get_all_subjects_data, fit_weibull

            subjects_data = get_all_subjects_data(tmpdir)
            fixed_data = subjects_data['test']['fixed']

            unique_coherences = np.unique(fixed_data['coherences'])
            avg_successes = [np.mean(fixed_data['successes'][fixed_data['coherences'] == c]) for c in unique_coherences]

            (alpha, beta), _ = fit_weibull(unique_coherences, np.array(avg_successes))

            self.assertAlmostEqual(alpha, alpha_gt, delta=0.1)
            self.assertAlmostEqual(beta, beta_gt, delta=0.3)
    @patch('matplotlib.pyplot.show')
    def test_new_analysis(self, mock_show):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_filepath = os.path.join(tmpdir, "test-motion_coherence_fixed-12345.log")

            coherences = [0.9, 0.2, 0.9, 0.3, 0.9, 0.4]
            successes =  [True, True, True, True, True, True]
            directions = [
                (0.1, 0.1),
                (0.1, 0.1),
                (0.1, 0.1),
                (0.1, np.pi + 0.1),
                (0.1, 0.1),
                (0.1, np.pi/2+0.1)
            ]
            generate_log_file(log_filepath, coherences, successes, directions)

            from experiments.analysis.motion_coherence import get_all_subjects_data, group_trials_by_prev_trial
            subjects_data = get_all_subjects_data(tmpdir)
            grouped_data = group_trials_by_prev_trial(subjects_data)

            self.assertEqual(len(grouped_data['same']['coherences']), 1)
            self.assertEqual(grouped_data['same']['coherences'][0], 0.2)
            self.assertEqual(len(grouped_data['opposite']['coherences']), 1)
            self.assertEqual(grouped_data['opposite']['coherences'][0], 0.3)
            self.assertEqual(len(grouped_data['90deg']['coherences']), 1)
            self.assertEqual(grouped_data['90deg']['coherences'][0], 0.4)
    @patch('matplotlib.pyplot.show')
    def test_single_subject_analysis(self, mock_show):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_filepath_fixed = os.path.join(tmpdir, "test-motion_coherence_fixed-12345.log")
            log_filepath_roving = os.path.join(tmpdir, "test-motion_coherence_roving-12346.log")

            coherences = np.repeat(np.linspace(0.1, 0.9, 10), 20)
            directions = [(0.1, 0.2)] * len(coherences)

            generate_log_file_with_weibull(log_filepath_fixed, coherences, directions, 0.5, 1.5)
            generate_log_file_with_weibull(log_filepath_roving, coherences, directions, 0.5, 1.5)

            from experiments.analysis.motion_coherence import analyze_subject
            analyze_subject("test", folder_path=tmpdir)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "analysis_curves.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "psychometric_curves_by_prev_trial.png")))

    @patch('matplotlib.pyplot.show')
    def test_population_analysis(self, mock_show):
        with tempfile.TemporaryDirectory() as tmpdir:
            coherences = np.repeat(np.linspace(0.1, 0.9, 10), 20)
            directions = [(0.1, 0.2)] * len(coherences)

            # Subject 1
            log_filepath1_fixed = os.path.join(tmpdir, "subject1-motion_coherence_fixed-12345.log")
            log_filepath1_roving = os.path.join(tmpdir, "subject1-motion_coherence_roving-12346.log")
            generate_log_file_with_weibull(log_filepath1_fixed, coherences, directions, 0.5, 1.5)
            generate_log_file_with_weibull(log_filepath1_roving, coherences, directions, 0.5, 1.5)

            # Subject 2
            log_filepath2_fixed = os.path.join(tmpdir, "subject2-motion_coherence_fixed-12347.log")
            log_filepath2_roving = os.path.join(tmpdir, "subject2-motion_coherence_roving-12348.log")
            generate_log_file_with_weibull(log_filepath2_fixed, coherences, directions, 0.5, 1.5)
            generate_log_file_with_weibull(log_filepath2_roving, coherences, directions, 0.5, 1.5)

            from experiments.analysis.motion_coherence import analyze_population
            analyze_population(tmpdir)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "analysis_curves.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "psychometric_curves_by_prev_trial.png")))

if __name__ == "__main__":
    unittest.main()
