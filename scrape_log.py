import re
from pathlib import Path
from typing import List, Optional

import fire
import pandas as pd

# Let's scrape for each replication
# 1. The number of chains remaining within each trial
#   1. Within that, we can separate out the chains at each timestep
# 2. The trials, and how many solutions we found in each trial


def main(logdir: str, logs: str, outdir: str):
    chains = pd.DataFrame(columns=["experiment", "replication", "trial", "attempt", "timestep", "chains_remaining"])
    for col, dtype in zip(chains.columns, [pd.StringDtype(), int, int, int, int, int]):
        chains[col] = chains[col].astype(dtype)
    chains_index = 0

    solutions = pd.DataFrame(columns=["experiment", "replication", "trial", "total_solutions", "solutions_remaining", "attempt"])
    for col, dtype in zip(solutions.columns, [pd.StringDtype(), int, int, int, int, int]):
        solutions[col] = solutions[col].astype(dtype)
    solutions_index = 0
    for log_name in logs.split(","):
        experiment = log_name.split(".")[0]
        print(experiment)
        
        replication = 0
        n_replications: Optional[int] = None
        trial: Optional[int] = None
        trial_n_solutions: Optional[int] = None
        solutions_remaining: Optional[int] = None
        attempt = 0
        n_attempts: Optional[int] = None
        n_chains: Optional[int] = None
        timestep = 0 # 0 is before the first action, 1 is after the first action, 2 is after the second action, etc

        for line in (Path(logdir) / log_name).open("r"):

            chains_start_match = re.search("([0-9]+) total causal chains this trial", line)
            chains_match = re.search("([0-9]+) chains remaining", line)
            match = chains_match if chains_match is not None else chains_start_match
            if match is not None: 
                new_n_chains = int(match.group(1))
                if n_chains is not None:
                    assert n_chains >= new_n_chains
                n_chains = new_n_chains
                chains.loc[chains_index] = (experiment, replication, trial, attempt, timestep, n_chains)
                chains_index += 1
                continue

            timestep_match = re.search("timestep=([0-9]+)", line)
            if timestep_match is not None:
                next_timestep = int(timestep_match.group(1)) + 1
                assert timestep + 1 == next_timestep, f"timestep={timestep}, next_timestep={next_timestep}"
                timestep = next_timestep
                continue

            attempt_match = re.search("Ending attempt\. Action limit reached\. There are ([0-9]+) unique solutions remaining\. You have ([0-9]+) attempts remaining\.", line)
            if attempt_match is not None:
                new_solutions_remaining = int(attempt_match.group(1))
                if solutions_remaining is not None:
                    assert solutions_remaining >= new_solutions_remaining
                solutions_remaining = new_solutions_remaining

                attempt += 1
                if attempt == 1:
                    n_attempts = attempt + int(attempt_match.group(2))
                else:
                    assert attempt + int(attempt_match.group(2)) == n_attempts 

                timestep = 0
                continue

            solution_match = re.search("You found a solution\. There are ([0-9]+) unique solutions remaining\.", line)
            if solution_match is not None:
                solutions_remaining = int(solution_match.group(1))
                solutions.loc[solutions_index] = (experiment, replication, trial, trial_n_solutions, solutions_remaining, attempt)
                solutions_index += 1
                attempt += 1
                timestep = 0
                continue

            if "You found all of the solutions." in line:
                solutions_remaining = 0
                solutions.loc[solutions_index] = (experiment, replication, trial, trial_n_solutions, solutions_remaining, attempt)
                solutions_index += 1
                attempt = 0
                n_chains = None
                continue

            if "Testing agent" in line:
                trial = -1 # negative trial for test trial
                timestep = 0
                attempt = 0
                n_chains = None
                continue

            trial_match = re.search("New trial trial([0-9]+)\. There are ([0-9]+) unique solutions remaining\.", line)
            if trial_match is not None:
                trial = int(trial_match.group(1))
                trial_n_solutions = int(trial_match.group(2))
                solutions_remaining = trial_n_solutions
                timestep = 0
                attempt = 0
                n_chains = None
                continue

            replication_match = re.search("Starting agent run ([0-9]+) of ([0-9]+)", line)
            if replication_match is not None:
                replication = int(replication_match.group(1))
                n_replications = int(replication_match.group(2))
                continue
        
    pd.to_pickle(chains, Path(outdir) / "chains.pickle" )
    pd.to_pickle(solutions, Path(outdir) / "solutions.pickle" )



if __name__ == "__main__":
    fire.Fire(main)
