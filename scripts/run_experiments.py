
import re 
import os
import sys
import time
import socket
import argparse
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import ir_measures
import toml
import psutil

from termcolor import colored

def parse_toml(filename):
    """Parse the TOML configuration file."""
    try:
        return toml.load(filename)
    except Exception as e:
        print(f"Error reading the TOML file: {e}")
        return None


def get_git_info(experiment_dir):
    """Get Git repository information and save it to git.output."""
    print()
    print(colored("Git info", "green"))
    git_output_file = os.path.join(experiment_dir, "git.output")

    try:
        with open(git_output_file, "w") as git_output:
            # Get current branch
            branch_process = subprocess.Popen("git rev-parse --abbrev-ref HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            branch_name = branch_process.stdout.read().decode().strip()
            branch_process.wait()

            # Get current commit id
            commit_process = subprocess.Popen("git rev-parse HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commit_id = commit_process.stdout.read().decode().strip()
            commit_process.wait()

            # Write to git.output
            git_output.write(f"Current Branch: {branch_name}\n")
            git_output.write(f"Commit ID: {commit_id}\n")
            print(f"Current Branch: {branch_name}")
            print(f"Commit ID: {commit_id}")

    except Exception as e:
        print("An error occurred while retrieving Git information:", e)
        sys.exit(1)


def compile_rust_code(configs, experiment_dir):
    """Compile the Rust code and save output."""
    print()
    print(colored("Compiling the Rust code", "green"))
    
    compile_command = configs.get("compile-command", "RUSTFLAGS='-C target-cpu=native' cargo build --release")

    compilation_output_file = os.path.join(experiment_dir, "compiler.output")

    try:
        print("Compiling Rust code with", compile_command)
        with open(compilation_output_file, "w") as comp_output:
            compile_process = subprocess.Popen(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(compile_process.stdout.readline, b''):
                decoded_line = line.decode()
                print(decoded_line, end='')  # Print each line as it is produced
                comp_output.write(decoded_line)  # Write each line to the output file
            compile_process.stdout.close()
            compile_process.wait()

        if compile_process.returncode != 0:
            print("Rust compilation failed.")
            sys.exit(1)
        print("Rust code compiled successfully.")

    except Exception as e:
        print()
        print(colored("ERROR: Problems during Rust compilation:", "red"), e)
        sys.exit(1)


def get_index_filename(base_filename, configs):
    """Generate the index filename based on the provided parameters."""
    name = []
    
    name.append(base_filename)

    # Check if pq_parameters and m-pq exist
    if "pq_parameters" in configs and "m-pq" in configs["pq_parameters"]:
        name.append(f"m-pq_{configs['pq_parameters']['m-pq']}")
    
    # Append indexing parameters
    name += sorted(f"{k}_{v}" for k, v in configs["indexing_parameters"].items())
    
    return "_".join(str(l) for l in name)


def build_index(configs, experiment_dir):
    """Build the index using the provided configuration."""
    input_file =  os.path.join(configs["folder"]["data"], configs["filename"]["dataset"])
    index_folder = configs["folder"]["index"]

    os.makedirs(index_folder, exist_ok=True)
    output_file = os.path.join(index_folder, get_index_filename(configs["filename"]["index"], configs))
    
    print()
    print(colored(f"Dataset filename:", "blue"), input_file)
    print(colored(f"Index filename:", "blue"), output_file)

    build_command = configs.get("build-command", None)
    if not build_command:
        raise ValueError("Build command must be specified!!!")

    command_and_params = [
        build_command,
        f"--data-file {input_file}",
        f"--output-file {output_file}",
        f"--m {configs['indexing_parameters']['m']}",
        f"--efc {configs['indexing_parameters']['efc']}",
        f"--metric {configs['indexing_parameters']['metric']}",
    ] 

    # If there is a section [pq_params] in the configuration file, add the parameters to the command
    if "pq_parameters" in configs:
        for k, v in configs["pq_parameters"].items():
            command_and_params.append(f"--{k} {v}")

    command = ' '.join(command_and_params)

    # Print the command that will be executed
    print()
    print(colored(f"Indexing", "green"))
    print(colored(f"Indexing command:", "blue"), command)

    building_output_file = os.path.join(experiment_dir, "building.output")

    # Build the index and display output in real-time
    print(colored("Building index...", "yellow"))
    building_time = 0
    with open(building_output_file, "w") as build_output:
        build_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(build_process.stdout.readline, b''):
            decoded_line = line.decode()
            print(decoded_line, end='')  # Print each line as it is produced
            build_output.write(decoded_line)  # Write each line to the output file
            if decoded_line.startswith("Time to build:") and decoded_line.strip().endswith("s (before serializing)"):
                building_time = int(decoded_line.split()[3])
        build_process.stdout.close()
        build_process.wait()

    if build_process.returncode != 0:
        print(colored("ERROR: Indexing failed!", "red"))
        sys.exit(1)

    print(colored(f"Index built successfully in {building_time} secs!", "yellow"))
    return building_time


def compute_metric(configs, output_file, gt_file, metric): 

    if metric == None or metric == "":
        print("No metric specified. Skipping evaluation.")
        return None

    column_names = ["query_id", "doc_id", "rank", "score"]
    gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
    res_pd = pd.read_csv(output_file, sep='\t', names=column_names)
    
    query_ids_path = os.path.join(configs['folder']['data'], configs['filename']['query_ids'])
    queries_ids = np.load(query_ids_path, allow_pickle=True)

    document_ids_path = os.path.join(configs['folder']['data'], configs['filename']['doc_ids'])
    doc_ids = np.load(os.path.realpath(document_ids_path), allow_pickle=True)
    
    gt_pd['query_id'] = gt_pd['query_id'].apply(lambda x: queries_ids[x])
    res_pd['query_id'] = res_pd['query_id'].apply(lambda x: queries_ids[x])
    
    gt_pd['doc_id'] = gt_pd['doc_id'].apply(lambda x: doc_ids[x])
    res_pd['doc_id'] = res_pd['doc_id'].apply(lambda x: doc_ids[x])
    
    qrels_path = configs['folder']['qrels_path']
    
    df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "useless", "doc_id", "relevance"])
    #if "nq" in configs['name']: # the order of the fields in nq is different. 
    if len(pd.unique(df_qrels['useless'])) != 1:
        df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "doc_id", "relevance", "useless"])

    gt_pd['doc_id'] = gt_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    res_pd['doc_id'] = res_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    
    gt_pd['query_id'] = gt_pd['query_id'].astype(df_qrels.query_id.dtype)
    res_pd['query_id'] = res_pd['query_id'].astype(df_qrels.query_id.dtype)
    
    ir_metric = ir_measures.parse_measure(metric)
    
    metric_val = ir_measures.calc_aggregate([ir_metric], df_qrels, res_pd)[ir_metric]
    metric_gt = ir_measures.calc_aggregate([ir_metric], df_qrels, gt_pd)[ir_metric]
    
    print(f"Metric of the run: {ir_metric}: {metric_val}")
    print(f"Metric of the gt : {ir_metric}: {metric_gt}")
    
    return metric_val
    

def compute_accuracy(query_file, gt_file):
    # if files are csv
    if gt_file.endswith(".csv") or gt_file.endswith(".tsv"):
        column_names = ["query_id", "doc_id", "rank", "score"]
        if gt_file.endswith(".csv"):
            gt_pd = pd.read_csv(gt_file, sep=',', names=column_names)
            res_pd = pd.read_csv(query_file, sep=',', names=column_names)
        else:
            gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
            res_pd = pd.read_csv(query_file, sep='\t', names=column_names)

        # Group both dataframes by 'query_id' and get unique 'doc_id' sets
        gt_pd_groups = gt_pd.groupby('query_id')['doc_id'].apply(set)
        res_pd_groups = res_pd.groupby('query_id')['doc_id'].apply(set)

        # Compute the intersection size for each query_id in both dataframes
        intersections_size = {
            query_id: len(gt_pd_groups[query_id] & res_pd_groups[query_id]) if query_id in res_pd_groups else 0
            for query_id in gt_pd_groups.index
        }

        # Computes total number of results in the groundtruth
        total_results = len(gt_pd)
        total_intersections = sum(intersections_size.values())

    elif gt_file.endswith(".npy"):
        # Read csv results and transform to numpy array
        column_names = ["query_id", "doc_id", "rank", "score"]
        res_pd = pd.read_csv(query_file, sep='\t', names=column_names)
        res_npy = res_pd['doc_id'].to_numpy()
        # Group results by query id and transform to npy array with shape (num_queries, num_results)
        res_npy = res_npy.reshape(-1, res_pd.groupby('query_id').size().max())
        k = res_npy.shape[1]

        # Read npy groundtruth
        doc_ids = np.load(gt_file, allow_pickle=True)
        
        # compute total results and total intersections
        total_results = res_npy.shape[0] * res_npy.shape[1]
        total_intersections = 0
        for i in range(res_npy.shape[0]):
            total_intersections += len(np.intersect1d(res_npy[i], doc_ids[i][:k]))
    else:
        raise ValueError("Groundtruth file must be in csv or numpy format!!!")
        
    return round((total_intersections/total_results) * 100, 3)


def query_execution(configs, query_config, experiment_dir, subsection_name):
    """Execute a query based on the provided configuration."""
    index_file = os.path.join(configs["folder"]["index"], get_index_filename(configs["filename"]["index"], configs))
    print("Searching index at:", index_file)
    query_file =  os.path.join(configs["folder"]["data"], configs["filename"]["queries"] ) 
    
    output_file = os.path.join(experiment_dir, f"results_{subsection_name}")
    log_output_file =  os.path.join(experiment_dir, f"log_{subsection_name}") 

    query_command = configs.get("query-command", None)
    if not query_command:
        raise ValueError("Query command must be specified!!!")

    #TODO: this must be updated after refactorin the search script in Rust
    
    command_and_params = [
        configs['settings']['NUMA'] if "NUMA" in configs['settings'] else "",
        query_command, 
        f"--index-file {index_file}",
        f"--query-file {query_file}",
        f"--k {configs['settings']['k']}",
        f"--ef-search {query_config['ef-search']}",
        f"--output-path {output_file}",
        f"--m-pq {configs['pq_parameters']['m-pq']}" if "pq_parameters" in configs and "m-pq" in configs['pq_parameters'] else "",
    ]

    command = " ".join(command_and_params)

    print(f"Executing query for subsection '{subsection_name}' with command:")
    print(command)

    pattern = r"\tTotal: (\d+) Bytes" # Pattern to match the total memory usage

    query_time = 0
    # Run the query and display output in real-time
    print(f"Running query for subsection: {subsection_name}...")
    with open(log_output_file, "w") as log:
        query_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(query_process.stdout.readline, b''):
            decoded_line = line.decode()
            if decoded_line.startswith("[######] Average Query Time"):
                match = re.search(r"Average Query Time: (\d+)", decoded_line)
                if match:
                    query_time = int(match.group(1))

            match = re.search(pattern, decoded_line)
            if match:
                memory_usage = int(match.group(1))
            else:
                memory_usage = 0
            print(decoded_line, end='')  # Print each line as it is produced
            log.write(decoded_line)  # Write each line to the output file
        query_process.stdout.close()
        query_process.wait()
    
    if query_process.returncode != 0:
        print(f"Query execution for subsection '{subsection_name}' failed.")
        sys.exit(1)

    print(f"Query for subsection '{subsection_name}' executed successfully.")

    gt_file = os.path.join(configs['folder']['data'], configs['filename']['groundtruth'])
    metric = configs['settings']['metric']
    return query_time, compute_accuracy(output_file, gt_file), compute_metric(configs, output_file, gt_file, metric), memory_usage


def get_machine_info(configs, experiment_folder):
    machine_info_file = os.path.join(experiment_folder, "machine.output")
    machine_info = open(machine_info_file, "w")

    date = datetime.now()
    machine = socket.gethostname()
    cpu = psutil.cpu_percent(interval=1)
    
    memory_free = psutil.virtual_memory().free // (1024 ** 3)
    memory_avail = psutil.virtual_memory().available // (1024 ** 3)
    memory_total = psutil.virtual_memory().total // (1024 ** 3)
    
    load = str(psutil.getloadavg())[1:-1]
    num_cpus = psutil.cpu_count()
    
    machine_info.write(f"----------------------\n")
    machine_info.write(f"Hardware configuration\n")
    machine_info.write(f"----------------------\n")
    machine_info.write(f"Date: {date}\n")
    machine_info.write(f"Machine: {machine}\n")
    machine_info.write(f"CPU usage (%): {cpu}\n")
    machine_info.write(f"Machine load: {load}\n")
    machine_info.write(f"Memory (free, GiB): {memory_free}\n")
    machine_info.write(f"Memory (avail, GiB): {memory_avail}\n")
    machine_info.write(f"Memory (total, GiB): {memory_total}\n")
    
    print()
    print(colored("Hardware configuration", "green"))
    print(f"Date: {date}")
    print(f"Machine: {machine}")
    print(f"CPU usage (%): {cpu}")
    print(f"Machine load: {load}")
    print(f"Memory (free, GiB): {memory_free}")
    print(f"Memory (avail, GiB): {memory_avail}")
    print(f"Memory (total, GiB): {memory_total}")
    print(f"for detailed information, check the hardware log file: {machine_info_file}")

    machine_info.write(f"\n---------------------\n")
    machine_info.write(f"cpufreq configuration\n")
    machine_info.write(f"---------------------\n")

    command_governor = 'cpufreq-info | grep "performance" | grep -v "available" | wc -l'
    governor = subprocess.Popen(command_governor, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    governor.wait()

    for line in iter(governor.stdout.readline, b''):
        cpus_with_performance_governor = int(line.decode())
        machine_info.write(f'Number of CPUs with governor set to "performance" (should be equal to the number of CPUs below): {cpus_with_performance_governor}\n')

    # checking if the hardware looks well configured...
    if (num_cpus != cpus_with_performance_governor):
        print()
        print(colored("ERROR: Problems with hardware configuration found!", "red"))
        print(colored("Your CPU is not set to performance mode. Please, run `cpufreq-info` for more details.", "red"))
        print()

    machine_info.write(f"\n-----------------\n")
    machine_info.write(f"CPU configuration\n")
    machine_info.write(f"-----------------\n")

    command_cpu = 'lscpu'
    cpu = subprocess.Popen(command_cpu, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cpu.wait()

    for line in iter(cpu.stdout.readline, b''):
        decoded_line = line.decode()
        machine_info.write(decoded_line)

    if ("NUMA" in configs['settings']):
        machine_info.write(f"\n------------------------------------------------------------------------------\n")
        machine_info.write(f"NUMA execution command (check if CPU IDs corresponds to physical ones (no HT))\n")
        machine_info.write(f"------------------------------------------------------------------------------\n")
        machine_info.write(f'Shell command: "{configs["settings"]["NUMA"]}"\n')

        machine_info.write(f"\n------------------\n")
        machine_info.write(f"NUMA configuration\n")
        machine_info.write(f"------------------\n")

        command_numa = 'numactl --hardware'
        numa = subprocess.Popen(command_numa, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        numa.wait()

        for line in iter(numa.stdout.readline, b''):
            decoded_line = line.decode()
            machine_info.write(decoded_line)

    machine_info.close()
    return


def run_experiment(config_data):
    """Run the kannolo experiment based on the provided configuration."""

     # Get the experiment name from the configuration
    experiment_name = config_data.get("name")
    print(f"Running experiment:", colored(experiment_name, "green"))

    for k, v in config_data["folder"].items():
        if v.startswith("~"):
            v = os.path.expanduser(v)
            config_data["folder"][k] = v

   #print(config_data)

    # Create an experiment folder with date and hour
    timestamp  = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    experiment_folder = os.path.join(config_data["folder"]["experiment"], f"{experiment_name}_{timestamp}")

    os.makedirs(experiment_folder, exist_ok=True)

    # Dump the configuration settings to a TOML file
    with open(os.path.join(experiment_folder, "experiment_config.toml"), 'w') as report_file:
        report_file.write(toml.dumps(config_data))

    # Retrieving hardware information
    get_machine_info(config_data, experiment_folder)

    # Store the output of the Rust compilation and index building processes
    get_git_info(experiment_folder)
    
    compile_rust_code(config_data, experiment_folder)

    building_time = 0
    if config_data['settings']['build']:
        building_time = build_index(config_data, experiment_folder)
    else:
        print("Index is already built!")

    metric = config_data['settings']['metric']
    print(f"Evaluation runs with metric {metric}")
    
    # Execute queries for each subsection under [query]
    with open(os.path.join(experiment_folder, "report.tsv"), 'w') as report_file:
        if metric != "":
            # Concatenate \t{metric} 
            metric = f"\t{metric}"
        report_file.write(f"Subsection\tQuery Time (microsecs)\tRecall{metric}\tMemory Usage (Bytes)\tBuilding Time (secs)\n")
        if 'query' in config_data:
            for subsection, query_config in config_data['query'].items():
                query_time, recall, metric, memory_usage = query_execution(config_data, query_config, experiment_folder, subsection)
                if metric is not None:
                    report_file.write(f"{subsection}\t{query_time}\t{recall}\t{metric}\t{memory_usage}\t{building_time}\n")
                else:
                    report_file.write(f"{subsection}\t{query_time}\t{recall}\t{memory_usage}\t{building_time}\n")

def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print()
        print(colored("ERROR: Configuration data is empty.", "red"))
        sys.exit(1)
    run_experiment(config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a kANNolo experiment on a dataset and query it.")
    parser.add_argument("--exp", required=True, help="Path to the experiment configuration TOML file.")
    args = parser.parse_args()

    main(args.exp)
    sys.exit(0)