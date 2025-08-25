from re import search, findall
from numpy import fromstring, array2string, array, float64, sort, log
from matplotlib.pyplot import subplots, show
import glob
import os


LOGFILE = "output/roei17-motion_coherence_roving-1756148886"

def analyze_latest(latest=True):

    folder_path = 'output'
    list_of_files = glob.glob(os.path.join(folder_path, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    text = open(latest_file if latest else LOGFILE, "r").read().replace("\n","")
    print(text)
    coherences = search(r"\[(.*)\] and direction", text)
    success = findall(r"(True|False)", text)


    if coherences is None:
        exit(1)

    coherences = fromstring(coherences.group(1), dtype=float64, sep=" ")
    success = array([s == "True" for s in success])

    averages = {c:0.0 for c in coherences}

    assert len(coherences) % len(averages) == 0
    amount_of_repetitions = len(coherences) / len(averages)

    for c, s in zip(coherences, success):
        averages[c] += int(s) / amount_of_repetitions

    coherences, successes = zip(*averages.items())
    coherences_and_successes = sort(array([coherences, successes]))


    print(averages)

    _, ax = subplots()
    ax.semilogx(coherences_and_successes[0], coherences_and_successes[1])
    show()

if __name__ == "__main__":
    analyze_latest(False)