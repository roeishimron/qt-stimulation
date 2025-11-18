import unittest
import tempfile
import os
import numpy as np
from experiments.analysis.motion_coherence import text_into_coherences_and_successes
from tests.log_generator import generate_log_file

class TestMotionCoherence(unittest.TestCase):
    def test_single_subject_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_filepath_fixed = os.path.join(tmpdir, "test-motion_coherence_fixed-12345.log")
            log_filepath_roving = os.path.join(tmpdir, "test-motion_coherence_roving-12346.log")

            coherences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            successes_fixed = [True, True, False, True, False, False, True, True, True, True, False, False, True, True, False, True, True, True]
            successes_roving = [True, False, False, False, True, False, True, True, False, True, True, False, True, True, True, True, True, True]
            directions = [(0.1, 0.2)] * 18

            generate_log_file(log_filepath_fixed, coherences, successes_fixed, directions)
            generate_log_file(log_filepath_roving, coherences, successes_roving, directions)

            from experiments.analysis.motion_coherence import analyze_subject
            analyze_subject("test", folder_path=tmpdir)

            # Check that the plot was generated
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "analysis_curves.png")))

    def test_population_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Subject 1
            log_filepath1_fixed = os.path.join(tmpdir, "subject1-motion_coherence_fixed-12345.log")
            log_filepath1_roving = os.path.join(tmpdir, "subject1-motion_coherence_roving-12346.log")
            coherences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            successes1_fixed = [True, True, False, True, False, False, True, True, True]
            successes1_roving = [True, False, False, False, True, False, True, True, False]
            directions = [(0.1, 0.2)] * 9
            generate_log_file(log_filepath1_fixed, coherences, successes1_fixed, directions)
            generate_log_file(log_filepath1_roving, coherences, successes1_roving, directions)

            # Subject 2
            log_filepath2_fixed = os.path.join(tmpdir, "subject2-motion_coherence_fixed-12347.log")
            log_filepath2_roving = os.path.join(tmpdir, "subject2-motion_coherence_roving-12348.log")
            successes2_fixed = [True, False, True, False, True, True, False, True, True]
            successes2_roving = [False, True, False, True, False, True, False, True, False]
            generate_log_file(log_filepath2_fixed, coherences, successes2_fixed, directions)
            generate_log_file(log_filepath2_roving, coherences, successes2_roving, directions)

            from experiments.analysis.motion_coherence import analyze_population
            analyze_population(tmpdir)

            # Check that the plot was generated
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "analysis_curves.png")))

if __name__ == "__main__":
    unittest.main()
