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

    if configs["filename"].get("knn_path", None):
        knn_path = os.path.join(configs['folder']['data'], configs['filename']['knn_path'])
        knn_path_arg = f"--knn-path {knn_path}"
        command_and_params.append(knn_path_arg)

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
            if decoded_line.startswith("Time to build ") and decoded_line.strip().endswith("(before serializing)"):
                building_time = int(decoded_line.split()[3])
        build_process.stdout.close()
        build_process.wait()

    if build_process.returncode != 0:
        print(colored("ERROR: Indexing failed!", "red"))
        sys.exit(1)

    print(colored(f"Index built successfully in {building_time} secs!", "yellow"))
    return building_time
    

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

    # for line in iter(governor.stdout.readline, b''):
    #     cpus_with_performance_governor = int(line.decode())
    #     machine_info.write(f'Number of CPUs with governor set to "performance" (should be equal to the number of CPUs below): {cpus_with_performance_governor}\n')

    # checking if the hardware looks well configured...
    # if (num_cpus != cpus_with_performance_governor):
    #     print()
    #     print(colored("ERROR: Problems with hardware configuration found!", "red"))
    #     print(colored("Your CPU is not set to performance mode. Please, run `cpufreq-info` for more details.", "red"))
    #     print()

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
    experiment_folder = os.path.join(config_data["folder"]["experiment"], f"experiments/{experiment_name}_{timestamp}")

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
    building_time = build_index(config_data, experiment_folder)
    
    # Execute queries for each subsection under [query]
    with open(os.path.join(experiment_folder, "report.tsv"), 'w') as report_file:
        report_file.write(f"Building Time (secs)\n")
        report_file.write(f"{building_time}\n")

def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print()
        print(colored("ERROR: Configuration data is empty.", "red"))
        sys.exit(1)
    run_experiment(config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a kANNolo index.")
    parser.add_argument("--exp", required=True, help="Path to the experiment configuration TOML file.")
    args = parser.parse_args()

    main(args.exp)
    sys.exit(0)
