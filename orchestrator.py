import sys
import os
import json
import subprocess
import time
import logging
import signal
import pandas as pd
from typing import Mapping, Iterable, Callable, Dict, List, Tuple, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cosng.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("COSNG")

class NoiseProfiler:
    """Class to manage noise profiles for the COSNG tool"""
    
    RT_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt-app", "src", "rt-app")
    
    @staticmethod
    def machine_setup(cores: Iterable, path: str = None, affinity_map: Mapping = None) -> Dict:
        """
        Setup machine configuration for experiments
        
        Args:
            cores: List of available CPU cores
            path: Working directory path
            affinity_map: Optional mapping for core affinity
            
        Returns:
            Dictionary with machine configuration
        """
        m = dict()
        m['cores'] = list(cores)
        if path is not None:
            realpath = os.path.realpath(path)
            if os.path.isdir(realpath) is False:
                raise Exception(f"Machine working directory is invalid: {realpath}")
            m['path'] = realpath
        else:
            m['path'] = os.getcwd()
            
        if affinity_map is not None:
            m['affinity_map'] = affinity_map
            
        # Verify RT-App exists
        if not os.path.exists(NoiseProfiler.RT_APP_PATH):
            logger.warning(f"rt-app not found at {NoiseProfiler.RT_APP_PATH}. You may need to run build-rt-app.sh")
        
        logger.info(f"Machine setup complete with cores: {m['cores']}")
        return m
    
    @staticmethod
    def generate_noise_singlecore(machine: Mapping, max_duration: int, busytime_us: int, 
                                 totaltime_us: int, cpu: int) -> Dict:
        """
        Generate a single-core noise profile configuration
        
        Args:
            machine: Machine configuration
            max_duration: Maximum duration of the noise in seconds
            busytime_us: CPU busy time in microseconds
            totaltime_us: Total cycle time in microseconds (busy + sleep)
            cpu: Target CPU core
            
        Returns:
            Dictionary with noise profile configuration
        """
        np = dict()
        if cpu not in machine['cores']:
            raise Exception(f"core:{cpu} not in the list of cores in the machine")
        np['core'] = cpu
        np['maxduration'] = max_duration
        if totaltime_us <= busytime_us:
            raise Exception(f'busytime:{busytime_us} > totaltime:{totaltime_us}')
        np['busytime'] = busytime_us
        np['totaltime'] = totaltime_us
        np['noiselevel'] = float(busytime_us/totaltime_us)
        np['name'] = f"nprofile_c{cpu}_max{max_duration}_t{totaltime_us}_b{busytime_us}"
        name = np['name']
        np['noisefile'] = os.path.join(machine['path'], f"{name}.json")
        
        # Create the rt-app JSON configuration
        rtapp_config = {
            "global": {
                "duration": max_duration,
                "calibration": "CPU4",
                "logdir": "./logs",
                "gnuplot": True
            },
            "tasks": {
                f"noise_core_{cpu}": {
                    "cpus": [cpu],
                    "instance": 1,
                    "run": busytime_us,
                    "sleep": totaltime_us - busytime_us
                }
            }
        }
        
        # Write the configuration file
        with open(np['noisefile'], 'w') as f:
            json.dump(rtapp_config, f, indent=4)
            
        logger.info(f"Generated noise profile: {np['name']} with noise level: {np['noiselevel']:.2f}")
        return np
    
    @staticmethod
    def generate_noise_multicore(machine: Mapping, max_duration: int, 
                                configs: List[Tuple[int, int, int]]) -> Dict:
        """
        Generate a multi-core noise profile configuration
        
        Args:
            machine: Machine configuration
            max_duration: Maximum duration of the noise in seconds
            configs: List of tuples (cpu, busytime_us, totaltime_us) for each core
            
        Returns:
            Dictionary with noise profile configuration
        """
        np = dict()
        np['cores'] = []
        np['maxduration'] = max_duration
        np['configurations'] = configs

        tot_busytime_us = 0 
        tot_totaltime_us = 0 
            
        # Build name from concatenated core configurations
        name_parts = []
        for cpu, busytime, totaltime in configs:
            if cpu not in machine['cores']:
                raise Exception(f"core:{cpu} not in the list of cores in the machine")
            if totaltime <= busytime:
                raise Exception(f'busytime:{busytime} > totaltime:{totaltime}')
            np['cores'].append(cpu)
            name_parts.append(f"c{cpu}_b{busytime}_t{totaltime}")
            tot_totaltime_us += totaltime
            tot_busytime_us += busytime
    
        np['name'] = f"nprofile_multi_max{max_duration}_{'_'.join(name_parts)}"
        np['noisefile'] = os.path.join(machine['path'], f"{np['name']}.json")
        #FIXME: This is not the correct noise level, it should factor in the noise among all cores
        # not just the ones with noise (probably the same is true for the single core)
        np['noiselevel'] = float(tot_busytime_us/tot_totaltime_us)

        # Create the rt-app JSON configuration
        rtapp_config = {
            "global": {
                "duration": max_duration,
                "calibration": "CPU4",
                "logdir": "./logs",
                "gnuplot": True
            },
            "tasks": {}
        }
        
        # Add each core configuration
        for idx, (cpu, busytime, totaltime) in enumerate(configs):
            rtapp_config["tasks"][f"noise_core_{cpu}"] = {
                "cpus": [cpu],
                "instance": 1,
                "run": busytime,
                "sleep": totaltime - busytime
            }
        
        # Write the configuration file
        with open(np['noisefile'], 'w') as f:
            json.dump(rtapp_config, f, indent=4)
            
        logger.info(f"Generated multi-core noise profile: {np['name']}")
        return np
    
    @staticmethod
    def start_noise(machine: Mapping, noise_profile: Mapping) -> Dict:
        """
        Start the noise generator using rt-app
        
        Args:
            machine: Machine configuration
            noise_profile: Noise profile configuration
            
        Returns:
            Dictionary with process information
        """
        if not os.path.exists(noise_profile['noisefile']):
            raise Exception(f"Noise profile file not found: {noise_profile['noisefile']}")
        
        # Start rt-app with the noise profile
        cmd = [NoiseProfiler.RT_APP_PATH, noise_profile['noisefile']]
        logger.info(f"Starting noise with command: {' '.join(cmd)}")
        
        # Execute in a new process and return handle
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=machine['path']
        )
        
        # Wait briefly to ensure rt-app starts properly
        time.sleep(1)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"Noise process failed to start. Exit code: {process.returncode}")
            logger.error(f"STDOUT: {stdout.decode('utf-8')}")
            logger.error(f"STDERR: {stderr.decode('utf-8')}")
            raise Exception(f"Failed to start noise process. Exit code: {process.returncode}")
        
        result = {
            'process': process,
            'start_time': datetime.now(),
            'noise_profile': noise_profile
        }
        
        logger.info(f"Noise started with PID: {process.pid}")
        return result
    
    @staticmethod
    def stop_noise(noise_process: Dict) -> None:
        """
        Stop a running noise process
        
        Args:
            noise_process: Running noise process information
        """
        if 'process' not in noise_process or noise_process['process'] is None:
            logger.warning("No active noise process to stop")
            return
            
        process = noise_process['process']
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"Stopping noise process with PID: {process.pid}")
            process.terminate()
            
            # Give it a moment to terminate gracefully
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Noise process did not terminate gracefully, sending SIGKILL")
                process.kill()
                
        # Get output for logging
        stdout, stderr = process.communicate()
        
        if process.returncode != 0 and process.returncode != -15:  # -15 is SIGTERM
            logger.warning(f"Noise process exited with code: {process.returncode}")
            logger.debug(f"STDOUT: {stdout.decode('utf-8')}")
            logger.debug(f"STDERR: {stderr.decode('utf-8')}")
        else:
            logger.info(f"Noise process stopped successfully. Duration: "
                      f"{datetime.now() - noise_process['start_time']}")
    
    @staticmethod
    def cleanup_noise_files(machine: Mapping, keep_configs: bool = False) -> None:
        """
        Clean up generated noise files
        
        Args:
            machine: Machine configuration
            keep_configs: If True, keep the JSON config files
        """
        # Clean up log files
        log_dir = os.path.join(machine['path'], "logs")
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.startswith("noise_core_"):
                    os.remove(os.path.join(log_dir, file))
                    
        # Clean up JSON config files if not keeping them
        if not keep_configs:
            for file in os.listdir(machine['path']):
                if file.startswith("nprofile_") and file.endswith(".json"):
                    os.remove(os.path.join(machine['path'], file))
        
        logger.info(f"Cleaned up noise files in {machine['path']}")


class BenchmarkRunner:
    """Class to manage benchmark execution for the COSNG tool"""
    
    @staticmethod
    def create_benchmark(name: str, bench_path: str, invocation_command: str, 
                        result_parser: Callable, cores: List[int], 
                        iterations: int = 3) -> Dict:
        """
        Create a benchmark configuration
        
        Args:
            name: Benchmark name
            bench_path: Path to the benchmark executable
            invocation_command: Command string to run the benchmark
            result_parser: Function to parse benchmark results
            cores: List of cores to run on
            iterations: Number of times to run the benchmark
            
        Returns:
            Dictionary with benchmark configuration
        """
        if not os.path.exists(bench_path):
            raise Exception(f"Benchmark path does not exist: {bench_path}")
            
        return {
            'name': name,
            'path': bench_path,
            'command': invocation_command,
            'result_parser': result_parser,
            'cores': cores,
            'iterations': iterations
        }
    
    @staticmethod
    def run_benchmark(machine: Mapping, benchmark: Mapping, 
                     noise_profile: Optional[Mapping] = None) -> Dict:
        """
        Run a benchmark with optional noise
        
        Args:
            machine: Machine configuration
            benchmark: Benchmark configuration
            noise_profile: Optional noise profile configuration
            
        Returns:
            Dictionary with benchmark results
        """
        results = []
        noise_process = None
        
        logger.info(f"Running benchmark: {benchmark['name']} with "
                  f"{'noise' if noise_profile else 'no noise'}")
        
        # Start noise if specified
        if noise_profile:
            try:
                noise_process = NoiseProfiler.start_noise(machine, noise_profile)
            except Exception as e:
                logger.error(f"Failed to start noise: {str(e)}")
                return {'error': str(e), 'success': False}
       
        omp_num_threads = len(benchmark['cores'])
        core_str_list=','.join(str(c) for c in benchmark['cores'])

        omp_cmd = f'export OMP_NUM_THREADS={omp_num_threads}'

        logger.info(f"Executing: {omp_cmd}")
        subprocess.Popen(omp_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Run the benchmark for the specified number of iterations
        try:
            for i in range(benchmark['iterations']):
                logger.info(f"Running iteration {i+1}/{benchmark['iterations']}")
                # Format the command string
                cmd = benchmark['command'].format(
                    path=benchmark['path'],
                    #cores=','.join(map(str, benchmark['cores']))
                )

                cmd = f'taskset {core_str_list} {cmd}'
                
                logger.info(f"Executing: {cmd}")
                
                # Run the benchmark
                start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.path.dirname(benchmark['path'])
                )
                
                stdout, stderr = process.communicate()
                end_time = time.time()
                
                stdout = stdout.decode('utf-8')
                stderr = stderr.decode('utf-8')
                
                # Parse results
                iteration_result = {
                    'iteration': i+1,
                    'exit_code': process.returncode,
                    'duration': end_time - start_time,
                    'stdout': stdout,
                    'stderr': stderr
                }
                
                if process.returncode != 0:
                    logger.error(f"Benchmark failed with exit code: {process.returncode}")
                    logger.error(f"STDERR: {stderr}")
                    iteration_result['success'] = False
                else:
                    # Parse the benchmark results
                    try:
                        parsed_results = benchmark['result_parser'](stdout)
                        iteration_result.update(parsed_results)
                        iteration_result['success'] = True
                    except Exception as e:
                        logger.error(f"Failed to parse benchmark results: {str(e)}")
                        iteration_result['success'] = False
                        iteration_result['parse_error'] = str(e)
                
                results.append(iteration_result)
                
                # Add a short delay between iterations
                if i < benchmark['iterations'] - 1:
                    time.sleep(2)
                    
        finally:
            # Always stop noise if it was started
            if noise_process:
                NoiseProfiler.stop_noise(noise_process)
        # Aggregate results
        successful_runs = [r for r in results if r.get('success', False)]
        failed_runs = [r for r in results if not r.get('success', False)]
        
        aggregated_results = {
            'benchmark': benchmark['name'],
            'with_noise': noise_profile is not None,
            'noise_profile': noise_profile['name'] if noise_profile else 'baseline',
            'noise_level': noise_profile['noiselevel'] if noise_profile else 0,
            'total_runs': len(results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate averages for successful runs
        if successful_runs:
            for key in successful_runs[0].keys():
                if key not in ['iteration', 'exit_code', 'stdout', 'stderr', 'success'] and \
                   isinstance(successful_runs[0][key], (int, float)):
                    values = [run[key] for run in successful_runs]
                    aggregated_results[f'avg_{key}'] = sum(values) / len(values)
                    aggregated_results[f'min_{key}'] = min(values)
                    aggregated_results[f'max_{key}'] = max(values)
                    aggregated_results[f'std_{key}'] = (sum((x - (sum(values) / len(values))) ** 2 
                                                      for x in values) / len(values)) ** 0.5
        
        logger.info(f"Benchmark completed: {benchmark['name']}. "
                  f"Success rate: {len(successful_runs)}/{len(results)}")
                  
        return aggregated_results


class ExperimentManager:
    """Class to manage COSNG experiments"""
    
    @staticmethod
    def create_experiment(machine: Mapping, benchmark: Mapping, 
                         noise_profiles: List[Optional[Mapping]] = None) -> Dict:
        """
        Create an experiment configuration
        
        Args:
            machine: Machine configuration
            benchmark: Benchmark configuration
            noise_profiles: List of noise profiles to test (None for baseline)
            
        Returns:
            Dictionary with experiment configuration
        """
        if noise_profiles is None:
            # Default: Run once with no noise (baseline)
            noise_profiles = [None]
            
        return {
            'machine': machine,
            'benchmark': benchmark,
            'noise_profiles': noise_profiles,
            'created_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def run_experiment(experiment: Mapping) -> Dict:
        """
        Run a complete experiment
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            Dictionary with experiment results
        """
        machine = experiment['machine']
        benchmark = experiment['benchmark']
        noise_profiles = experiment['noise_profiles']
        
        logger.info(f"Starting experiment with benchmark: {benchmark['name']}")
        logger.info(f"Running with {len(noise_profiles)} noise profiles "
                  f"(including baseline if None is in the list)")
        
        results = []
        
        # Run baseline (no noise) if None is in the list
        for profile in noise_profiles:
            #import pdb; pdb.set_trace()
            profile_name = "baseline" if profile is None else profile['name']
            logger.info(f"Running with noise profile: {profile_name}")
            
            try:
                result = BenchmarkRunner.run_benchmark(machine, benchmark, profile)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run benchmark with profile {profile_name}: {str(e)}")
                results.append({
                    'benchmark': benchmark['name'],
                    'noise_profile': profile_name,
                    'error': str(e),
                    'success': False
                })
        
        # Compile all results
        experiment_results = {
            'benchmark': benchmark['name'],
            'machine': {
                'cores': machine['cores'],
                'path': machine['path']
            },
            'results': results,
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Experiment completed for benchmark: {benchmark['name']}")
        return experiment_results
    
    @staticmethod
    def save_experiment_results(experiment_results: Dict, output_file: str) -> None:
        """
        Save experiment results to a file
        
        Args:
            experiment_results: Experiment results
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        logger.info(f"Saved experiment results to: {output_file}")
    
    @staticmethod
    def generate_csv_report(experiment_results: Dict, output_csv: str) -> None:
        """
        Generate a CSV report from experiment results
        
        Args:
            experiment_results: Experiment results
            output_csv: Output CSV file path
        """
        rows = []
        benchmark_name = experiment_results['benchmark']
        
        for result in experiment_results['results']:
            is_baseline = result['noise_profile'] == "baseline"
            
            # Only include successful runs
            if result.get('successful_runs', 0) > 0:
                row = {
                    'Benchmark': benchmark_name,
                    'NoiseProfile': result['noise_profile'],
                    'NoiseLevel': result['noise_level'],
                    'SuccessRate': f"{result['successful_runs']}/{result['total_runs']}",
                    'IsBaseline': is_baseline
                }
                
                # Add all average metrics
                for key in result.keys():
                    if key.startswith('avg_'):
                        metric_name = key.replace('avg_', '')
                        row[f'Avg_{metric_name}'] = result[key]
                        
                        # Add min, max, std if available
                        if f'min_{metric_name}' in result:
                            row[f'Min_{metric_name}'] = result[f'min_{metric_name}']
                        if f'max_{metric_name}' in result:
                            row[f'Max_{metric_name}'] = result[f'max_{metric_name}']
                        if f'std_{metric_name}' in result:
                            row[f'Std_{metric_name}'] = result[f'std_{metric_name}']
                
                rows.append(row)
        
        # Convert to DataFrame and save
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
            logger.info(f"Generated CSV report: {output_csv}")
        else:
            logger.warning(f"No valid results to generate CSV report")


# Example NAS benchmark result parser
def parse_nas_benchmark_results(stdout: str) -> Dict:
    """
    Parse NAS benchmark results from stdout
    
    Args:
        stdout: Benchmark output
        
    Returns:
        Dictionary with parsed results
    """
    results = {}
    
    # Example pattern: look for the time and Mop/s metrics
    for line in stdout.splitlines():
        if "Time in seconds" in line:
            try:
                results['execution_time'] = float(line.split("=")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "Mop/s total" in line:
            try:
                results['mops'] = float(line.split("=")[1].strip())
            except (IndexError, ValueError):
                pass
    
    return results


# Example usage
if __name__ == "__main__":
    # Setup machine configuration
    machine = NoiseProfiler.machine_setup(cores=[0, 1, 2, 3], path=".")
    
    # Generate a single-core noise profile
    noise_profile1 = NoiseProfiler.generate_noise_singlecore(
        machine=machine,
        max_duration=300,  # 5 minutes
        busytime_us=50,
        totaltime_us=150,
        cpu=1
    )
    
    # Generate a multi-core noise profile
    noise_profile2 = NoiseProfiler.generate_noise_multicore(
        machine=machine,
        max_duration=300,  # 5 minutes
        configs=[
            (1, 50, 150),   # Core 1: 33% CPU usage
            (2, 100, 300)   # Core 2: 33% CPU usage
        ]
    )
    
    # Define a benchmark (example for NAS Parallel Benchmark)
    # Note: Update the path to your actual NPB installation
    nas_benchmark = BenchmarkRunner.create_benchmark(
        name="NPB-FT-A",
        bench_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "NPB", "run_nas_benchmark.sh"),
        invocation_command="{path} ft C",
        result_parser=parse_nas_benchmark_results,
        cores=[0, 1, 2, 3],  # Cores to use for the benchmark
        iterations=3
    )
    
    # Create an experiment
    experiment = ExperimentManager.create_experiment(
        machine=machine,
        benchmark=nas_benchmark,
        noise_profiles=[None, noise_profile1, noise_profile2]  # None = baseline (no noise)
    )
    
    # Run the experiment
    results = ExperimentManager.run_experiment(experiment)


    # Save results
    ExperimentManager.save_experiment_results(
        experiment_results=results,
        output_file="experiment_results.json"
    )
    
    # Generate CSV report
    ExperimentManager.generate_csv_report(
        experiment_results=results,
        output_csv="experiment_report.csv"
    )
    
    # Clean up
    NoiseProfiler.cleanup_noise_files(machine, keep_configs=True)

