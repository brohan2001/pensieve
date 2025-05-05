#!/bin/bash

# Automated testing script for Pensieve
# Run this from the project root directory

echo "Starting automated testing of all Pensieve models..."
echo "=================================================="

# Check if in correct directory
if [ ! -d "test" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Clean previous test results and summary files
echo "Cleaning previous test results..."
cd test
rm -f summary_*.txt
rm -rf test_results_*
rm -f testing_report.json
cd ..

# Run automated testing
echo "Running automated tests..."
python test_all_models.py

# Generate visualization if matplotlib is available
cd test
if python -c "import matplotlib" &> /dev/null; then
    echo "Generating result visualizations..."
    python ../visualize_test_results.py
    echo "Visualization complete! Check PNG files in test directory."
else
    echo "Matplotlib not available. Skipping visualization."
fi

echo "=================================================="
echo "Testing complete!"
echo "Results saved to:"
echo "  - testing_report.json"
echo "  - summary_*.txt files"
echo "  - test_results_* directories"
echo "=================================================="
