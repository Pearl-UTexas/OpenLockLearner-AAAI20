import pickle
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(datadir: str, outdir: str):
    chains = pickle.load(open(Path(datadir) / "chains.pickle", "rb"))
    solutions = pickle.load(open(Path(datadir) / "solutions.pickle", "rb"))

    chains["action"] = chains.attempt * 5 + chains.timestep

    # print(chains.experiment[0])
    # print(chains.experiment)

    for experiment in chains.experiment.unique():
        # print(experiment)
        experiment_chains = chains[chains.experiment == experiment]
        sns.lineplot(x="action", y="chains_remaining", data=experiment_chains)
        plt.title(f"{experiment} Chain Pruning")
        plt.xlabel("Action #")
        plt.ylabel("Chains remaining")
        plt.tight_layout()
        plt.savefig(Path(outdir) / f"{experiment}.chains.png")
        plt.close()



if __name__ == "__main__":
    fire.Fire(main)
