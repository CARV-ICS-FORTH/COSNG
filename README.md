
# COSNG: Controlled Operating System Noise Generator

A tool for evaluating the impact of system noise on the performance of workloads. COSNG uses `rt-app` to generate
precise CPU load patterns and measures their effect on workloads (currently has built in support for NAS parallel benchmarks).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/CARV-ICS-FORTH/COSNG.git
cd COSNG

# Build the components
./scripts/build-rt-app.sh
./scripts/build-nas-benchmarks.sh

# In case you dont have the json-c library do the following
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(realpath ./rt-app/json-c)

# Run the example experiment
python3 orchestrator.py
```

## Prerequisites

- **System**: Linux with `taskset`, GCC, and build tools
- **Python**: 3.6+ with `pandas`, `numpy` (`pip install pandas numpy`)

## Usage

### Building Components

```bash
# Build rt-app noise generator
./scripts/build-rt-app.sh

# Build NAS Parallel Benchmarks
./scripts/build-nas-benchmarks.sh
```

### Basic Experiment

```python

# 1. Configure machine
machine = NoiseProfiler.machine_setup(cores=[0, 1, 2, 3])

# 2. Create noise profiles
noise_profile = NoiseProfiler.generate_noise_singlecore(
    machine=machine,
    max_duration=300,    # 5 minutes
    busytime_us=50,      # 50μs busy
    totaltime_us=150,    # 150μs total cycle (33% CPU)
    cpu=1                # Target core
)

# 3. Define benchmark
benchmark = BenchmarkRunner.create_benchmark(
    name="NPB-FT-A",
    bench_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "NPB", "run_nas_benchmark.sh"),
    invocation_command="{path} ft C",
    result_parser=parse_nas_benchmark_results,
    cores=[0, 1, 2, 3],
    iterations=3
)

# 4. Run experiment
experiment = ExperimentManager.create_experiment(
    machine=machine,
    benchmark=benchmark,
    noise_profiles=[None, noise_profile]  # None = baseline (no noise)
)

results = ExperimentManager.run_experiment(experiment)

# 5. Save results
ExperimentManager.save_experiment_results(results, "results.json")
ExperimentManager.generate_csv_report(results, "report.csv")
```

## Creating Custom Experiments

### Noise Profiles

```python
# Single-core noise (50% CPU load on core 2)
noise_profile1 = NoiseProfiler.generate_noise_singlecore(
    machine=machine, max_duration=300, 
    busytime_us=500, totaltime_us=1000, cpu=2
)

# Multi-core noise (different patterns on multiple cores)
noise_profile2 = NoiseProfiler.generate_noise_multicore(
    machine=machine, max_duration=300,
    configs=[
        (1, 200, 1000),  # Core 1: 20% CPU
        (2, 500, 1000)   # Core 2: 50% CPU
    ]
)
```

### Custom Benchmarks

```python
# Custom benchmark parser function
def parse_custom_benchmark(stdout):
    results = {}
    # Extract relevant metrics from stdout
    return results

# Create custom benchmark
custom_benchmark = BenchmarkRunner.create_benchmark(
    name="CustomBench",
    bench_path="/path/to/benchmark",
    invocation_command="{path} -t {cores_count}",
    result_parser=parse_custom_benchmark,
    cores=[0, 1, 2, 3],
    iterations=5
)
```

## Result Analysis

The experiment produces:
- JSON file with detailed results
- CSV report with summary metrics

## Advanced Usage

### Testing Multiple Noise Levels

```python
# Generate noise profiles at different intensities
noise_profiles = []
for cpu_pct in [10, 20, 30, 50, 70]:
    busytime = int(cpu_pct * 10)  # Convert percent to microseconds (scaled)
    profile = NoiseProfiler.generate_noise_singlecore(
        machine=machine, max_duration=300,
        busytime_us=busytime, totaltime_us=1000, cpu=1
    )
    noise_profiles.append(profile)

# Run experiment with all profiles
experiment = ExperimentManager.create_experiment(
    machine=machine, benchmark=benchmark,
    noise_profiles=[None] + noise_profiles  # Add baseline
)
```

### Multiple Benchmarks

```python
# Define different benchmarks
benchmarks = [
    BenchmarkRunner.create_benchmark(name="NPB-FT-A", bench_path= ...),
    BenchmarkRunner.create_benchmark(name="NPB-CG-B", bench_path= ...),
    BenchmarkRunner.create_benchmark(name="NPB-LU-C", bench_path= ...)
]

# Run experiments for each benchmark
for benchmark in benchmarks:
    experiment = ExperimentManager.create_experiment(
        machine=machine, benchmark=benchmark, noise_profiles=[None, noise_profile]
    )
    results = ExperimentManager.run_experiment(experiment)
    ExperimentManager.save_experiment_results(
        results, f"{benchmark['name']}_results.json"
    )
```

## Troubleshooting

- **rt-app fails**: Check permissions and JSON validity
- **Inconsistent results**: Increase iterations, check for other processes
- **No noise effect**: Verify noise is on correct cores, increase intensity


## In Summary

COSNG orchestrates experiments by:
1. Generating JSON configurations for rt-app
2. Running rt-app to create controlled CPU load
3. Executing benchmarks while noise is active
4. Collecting and analyzing performance metrics

The noise level is controlled by the busy/sleep cycle ratio in rt-app tasks.


### Acknowledgements
We thankfully acknowledge support for this research from the European High Performance Computing Joint Undertaking (EuroHPC JU) under Framework Partnership Agreement No 800928 (European Processor Initiative) and Specific Grant Agreement No 101036168 (EPI-SGA2). The EuroHPC JU receives support from the European Union’s Horizon 2020 research and innovation programme and from Croatia, France, Germany, Greece, Italy, Netherlands, Portugal, Spain, Sweden, and Switzerland. National contributions from the involved state members (including the Greek General Secretariat for Research and Innovation) match the EuroHPC funding.
