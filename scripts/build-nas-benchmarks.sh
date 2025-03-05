#!/usr/bin/env bash
#
# Script to download and build the NAS Parallel Benchmarks (OpenMP version)
# Builds both C and Fortran implementations
# This should be run from the project root directory
#

set -e  # Exit on error

SCRIPT_PATH=$(dirname "$0")
ORIG_PATH=$(pwd)
NPB_VERSION="3.4.2"
NPB_DIR="NPB"
BUILD_DIR="$SCRIPT_PATH/../$NPB_DIR"
NAS_URL="https://www.nas.nasa.gov/assets/npb/NPB${NPB_VERSION}.tar.gz"
NUM_PROCS=$(nproc)

echo "=== Building NAS Parallel Benchmarks v${NPB_VERSION} ==="
echo "=== OpenMP Version (C and Fortran) ==="
echo "Using ${NUM_PROCS} processors for compilation"

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Download NAS Parallel Benchmarks if not already present
if [ ! -f "NPB${NPB_VERSION}.tar.gz" ]; then
    echo "Downloading NAS Parallel Benchmarks..."
    wget $NAS_URL || curl -O $NAS_URL
else
    echo "NAS Parallel Benchmarks archive already downloaded"
fi

# Extract archive if not already extracted
if [ ! -d "NPB${NPB_VERSION}" ]; then
    echo "Extracting archive..."
    tar -xzf "NPB${NPB_VERSION}.tar.gz"
else
    echo "Archive already extracted"
fi

# Create a directory for binaries
mkdir -p bin

# Build the OpenMP version
echo "Building OpenMP version (C and Fortran)..."
cd "NPB${NPB_VERSION}/NPB3.4-OMP"

# Create config file from template
cp config/make.def.template config/make.def

# Handle differences in Fortran compiler flags
FORTRAN_FLAG=""
if gfortran --version | grep -q "GCC"; then
    if gfortran --help=warnings | grep -q -- "-Wno-argument-mismatch"; then
        FORTRAN_FLAG="-Wno-argument-mismatch"
    elif gfortran --help=warnings | grep -q -- "-fallow-argument-mismatch"; then
        FORTRAN_FLAG="-fallow-argument-mismatch"
    fi
fi

# Customize the make.def file for both C and Fortran
sed -i "s|^CC.*|CC = gcc|g" config/make.def
sed -i "s|^CFLAGS.*|CFLAGS = -O3 -fopenmp|g" config/make.def
sed -i "s|^CLINKFLAGS.*|CLINKFLAGS = -O3 -fopenmp|g" config/make.def
sed -i "s|^FC.*|FC = gfortran|g" config/make.def
sed -i "s|^FFLAGS.*|FFLAGS = -O3 -fopenmp $FORTRAN_FLAG|g" config/make.def
sed -i "s|^FLINKFLAGS.*|FLINKFLAGS = -O3 -fopenmp|g" config/make.def

# Create suite file for all benchmarks (C and Fortran)
cat > config/suite.def << EOF
# Class A (small)
# C benchmarks
is A
ep A
cg A
ft A
mg A
# Fortran benchmarks
bt A
sp A
lu A

# Class B (medium)
# C benchmarks
is B
ep B
cg B
ft B
mg B
# Fortran benchmarks
bt B
sp B
lu B

# Class C (large)
# C benchmarks
is C
ep C
cg C
ft C
mg C
# Fortran benchmarks
bt C
sp C
lu C
EOF

echo "Making all benchmarks (this may take a while)..."
make suite || echo "Some benchmarks may have failed to compile, continuing with those that succeeded"

# Copy binaries to the central bin directory
echo "Copying binaries to bin directory..."
cp -v bin/*.x ../../bin/ || true

# Return to the NPB directory
cd ../../

# Check for successful build
BIN_COUNT=$(find bin -type f -name "*.x" | wc -l)
if [ $BIN_COUNT -gt 0 ]; then
    echo "=== NAS Parallel Benchmarks built successfully ==="
    echo "Found $BIN_COUNT benchmark executables"
    echo "Binaries are in $(pwd)/bin/"
    
    # List C benchmarks
    echo "C benchmarks:"
    for bench in is ep cg ft mg; do
        find bin -type f -name "${bench}.*.x" | sort
    done
    
    # List Fortran benchmarks
    echo "Fortran benchmarks:"
    for bench in bt sp lu; do
        find bin -type f -name "${bench}.*.x" | sort
    done
else
    echo "=== Build failed, no benchmark binaries found ==="
    echo "Check the output above for errors"
    exit 1
fi

# Create a simple run script
cat > run_nas_benchmark.sh << 'EOF'
#!/bin/bash
# Simple script to run NAS OpenMP benchmarks
# Usage: ./run_nas_benchmark.sh <benchmark> <class> [num_threads]
# Example: ./run_nas_benchmark.sh ft A 4  # Runs FT class A with 4 threads

if [ $# -lt 2 ]; then
    echo "Usage: $0 <benchmark> <class> [num_threads]"
    echo "Example: $0 ft A 4"
    echo ""
    echo "Available benchmarks:"
    echo "  C implementations: is, ep, cg, ft, mg"
    echo "  Fortran implementations: bt, sp, lu"
    echo "Available classes: A, B, C"
    exit 1
fi

BENCHMARK=$1
CLASS=$2
NUM_THREADS=${3:-$(nproc)}
BINARY="bin/${BENCHMARK}.${CLASS}.x"

# Check if the benchmark exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Benchmark $BINARY not found"
    echo "Available binaries:"
    find bin -name "*.x" | sort
    exit 1
fi

echo "Running $BENCHMARK (Class $CLASS) with $NUM_THREADS threads..."

# Set OpenMP environment variables
export OMP_NUM_THREADS=$NUM_THREADS
export NPB_TIMER_FLAG=1  # Detailed timing

# Run the benchmark
$BINARY
EOF

chmod +x run_nas_benchmark.sh


# Return to original directory
cd $ORIG_PATH

