from re import search, findall
from typing import Tuple
from numpy import fromstring, array2string, array, float64, argsort, log
from numpy.typing import NDArray
from matplotlib.pyplot import legend, subplots, show
import glob
import os


FOLDER_PATH = 'output'
LOGFILE = f"{FOLDER_PATH}/roei17-motion_coherence_roving-1756148886"


def analyze_latest():
    list_of_files = glob.glob(os.path.join(FOLDER_PATH, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    subject_search = search(r"^(.*)-motion_coherence", latest_file)
    assert subject_search is not None

    analyze_subject(subject_search.group(1))


# returns
def text_into_coherences_and_successes(text: str) -> Tuple[NDArray, NDArray]:
    coherences = search(r"\[(.*)\] and direction", text)
    success = findall(r"(True|False)", text)

    assert coherences is not None

    coherences = fromstring(coherences.group(1), dtype=float64, sep=" ")
    success = array([s == "True" for s in success])

    assert len(coherences) == len(success)

    averages = {c: 0.0 for c in coherences}

    assert len(coherences) % len(averages) == 0
    amount_of_repetitions = len(coherences) / len(averages)

    for c, s in zip(coherences, success):
        averages[c] += int(s) / amount_of_repetitions

    coherences, successes = zip(*averages.items())
    coherences, successes = array(coherences), array(successes)
    sort_indices = argsort(coherences)
    coherences, successes = coherences[sort_indices], successes[sort_indices]

    return coherences, successes


def analyze_subject(subject_name: str):
    list_of_files = glob.glob(os.path.join(FOLDER_PATH, f'{subject_name}-*'))
    kinds = [search(r"(fixed|roving)", filename).group(1)
             for filename in list_of_files]
    times = [int(search(r"(\d*)$", filename).group(1))
             for filename in list_of_files]
    ordered = times[0] < times[1]

    if not ordered:
        list_of_files.reverse()
        kinds.reverse()

    assert len(list_of_files) > 0
    _, ax = subplots(label=f"analysis of {subject_name}")

    for filename, kind in zip(list_of_files, kinds):
        coherences, successes = text_into_coherences_and_successes(
            open(filename).read().replace("\n", ""))
        ax.semilogx(coherences, successes, label=f"{kind}")

    legend()
    show()


if __name__ == "__main__":
    analyze_subject("10461")
