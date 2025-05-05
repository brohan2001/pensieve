#!/bin/bash

# Run All Tests Script for Pensieve Models
# Execute this script from the project root directory

echo "Starting comprehensive testing of all Pensieve models..."
echo "=================================================="

# Check if in correct directory
if [ ! -d "test" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Change to test directory
cd test

# Create models directory
mkdir -p models

# Create summary of the final training results before testing
echo "Checking final training results..."
cd ../sim

# Quick check of each trained model
for model_dir in improved_quick_results network_pattern_results buffer_aware_results; do
    if [ -d "$model_dir" ]; then
        latest_epoch=$(ls $model_dir/nn_model_ep_*.ckpt.meta 2>/dev/null | sed 's/.*ep_\([0-9]*\).*/\1/' | sort -n | tail -1)
        if [ ! -z "$latest_epoch" ]; then
            echo "Found $model_dir model at epoch $latest_epoch"
        else
            echo "Warning: No model found in $model_dir"
        fi
    else
        echo "Warning: $model_dir does not exist"
    fi
done

cd ../test

# Test all models using the automated script
echo "Starting automated testing..."
python ../test_all_models.py

# Check if testing was successful
if [ $? -eq 0 ]; then
    echo "Testing completed successfully!"
    
    # Generate visualizations if results exist
    if [ -f "testing_report.json" ]; then
        echo "Generating visualization plots..."
        python ../visualize_test_results.py
        
        if [ $? -eq 0 ]; then
            echo "Plots generated successfully!"
            echo "Check the following files:"
            echo "  - comparison_plots.png"
            echo "  - overall_performance_comparison.png"
            echo "  - improvement_analysis.png"
        else
            echo "Error generating plots"
        fi
    else
        echo "No testing report found"
    fi
else
    echo "Testing failed. Please check the errors above."
fi

echo "=================================================="
echo "All testing operations completed."
echo "See testing_report.json for detailed results."
