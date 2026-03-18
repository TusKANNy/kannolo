import argparse
import sys
import os

import json
from datetime import datetime
from termcolor import colored

import itertools

# from run_experiments import 
from run_experiments import (
    run_experiment,
    parse_toml,
)

def generate_indexing_parameters_combinations(params):
    # Extract keys and values from the dictionary
    keys, values = zip(*params.items())
    # Generate all possible combinations
    all_combinations = itertools.product(*values)
    
    unique_combinations = set()
    for combination in all_combinations:
        combo_dict = dict(zip(keys, combination))
        
        # Convert dict to tuple of sorted items for deduplication and add to set
        combo_tuple = tuple(sorted(combo_dict.items()))
        unique_combinations.add(combo_tuple)
    
    # Convert back to list of dictionaries
    return [dict(combo) for combo in unique_combinations]

def generate_query_combinations(params, comb_id = 1):

    keys, values = zip(*params.items())
    all_combinations = itertools.product(*values)
    
    combination_dict = {}
    for i, combination in enumerate(all_combinations, start=comb_id):
        combo_key = f"combination_{i}"
        combo_dict = dict(zip(keys, combination))
        combination_dict[combo_key] = combo_dict
    
    return combination_dict


def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print("Error: Configuration data is empty.")
        sys.exit(1)

    # Get the experiment name from the configuration
    grid_name = config_data.get("name")
    print(f"Running Grid: {grid_name}")

    # Create an experiment folder with date and hour
    timestamp = str(datetime.now()).replace(" ", "_")
    grid_folder = os.path.join(
        config_data["folder"]["experiment"], f"{grid_name}_{timestamp}"
    )
    os.makedirs(grid_folder, exist_ok=True)
    
    print()
    print(colored("Grid search information:", "yellow"))
    
    print(json.dumps(config_data["indexing_parameters"], indent=4))

    query_combinations = {}

    import copy

    if "early-termination" not in config_data["querying_parameters"] or len(config_data["querying_parameters"]["early-termination"]) == 1:
        query_combinations = generate_query_combinations(config_data["querying_parameters"])
    else:
        # To avoid generating all combinations of early-termination, we will run a separate grid for each early-termination strategy
        # for example, patience is needed only when early-termination is patience-proximity but not when is none
        if "none" in config_data["querying_parameters"]["early-termination"]:
            config_data_none = copy.deepcopy(config_data)
            config_data_none["querying_parameters"]["early-termination"] = ["none"]
            if "patience" in config_data_none["querying_parameters"]:
                del(config_data_none["querying_parameters"]["patience"])
            if "proximity-threshold" in config_data_none["querying_parameters"]:
                del(config_data_none["querying_parameters"]["proximity-threshold"])
            query_combinations = generate_query_combinations(config_data_none["querying_parameters"])
        print(config_data["querying_parameters"]["early-termination"])
        if "patience-proximity" in config_data["querying_parameters"]["early-termination"]:
            config_data_patience = copy.deepcopy(config_data)
            config_data_patience["querying_parameters"]["early-termination"] = ["patience-proximity"]
            for comb, v in generate_query_combinations(config_data_patience["querying_parameters"], comb_id = len(query_combinations)+1).items():
                query_combinations[comb] = v


    print("Run an experiment for each building configuration")

    for i, building_config in enumerate(generate_indexing_parameters_combinations(config_data["indexing_parameters"])):
        print()
        print(f"Running buiding combination {i} with {config_data['indexing_parameters']}")
        print(f"Running buiding combination {i} with {json.dumps(config_data['indexing_parameters'], indent=4)}")

        experiment_config = {}

        experiment_config = config_data.copy()

        experiment_config["folder"] = config_data["folder"]
        experiment_config["folder"]["experiment"] = grid_folder

        experiment_config["filename"] = config_data["filename"]
        experiment_config["settings"] = config_data["settings"]

        experiment_config["name"] = f"building_combination_{i}"

        experiment_config["query"] = query_combinations

        experiment_config["indexing_parameters"] = building_config
        with open(os.path.join(grid_folder, f"building_combination_{i}.json"), "w") as f:
            json.dump(building_config, f, indent=4)
        run_experiment(experiment_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid search of kANNolo experiments on a dataset and find the best configurations to query it."
    )
    parser.add_argument(
        "--exp", required=True, help="Path to the grid configuration TOML file."
    )
    args = parser.parse_args()

    main(args.exp)
