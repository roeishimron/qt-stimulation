import os

from analysis.motion_coherence.parsing import get_all_subjects_data
from analysis.motion_coherence.plotting import plot_analysis_curves, plot_psychometric_curves_by_prev_trial


def run_population_analysis(folder_path: str):
    """
    Analyzes the entire population in a folder.
    """
    all_data = get_all_subjects_data(folder_path)
    if not all_data:
        print(f"No data found in the specified folder. {folder_path}")
        return
    plot_analysis_curves(all_data, folder_path)
    plot_psychometric_curves_by_prev_trial(all_data, folder_path)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output")
# FOLDER_PATH = 'output'

def run():
    import sys

    if len(sys.argv) > 1:
        run_population_analysis(f"{FOLDER_PATH}/{sys.argv[1]}")
    else:
        print("Please provide the output/<population>")
