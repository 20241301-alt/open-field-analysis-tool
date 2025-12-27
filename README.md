# open-field-analysis-tool

# DeepLabCut Trajectory Analysis Tool

This repository contains a Python-based analysis tool for processing trajectory data exported from DeepLabCut. It calculates travel distance, velocity, and time spent in specific central zones (30% and 50% of the arena).

## Features
- **GUI-based interaction**: Easily select data folders and define analysis regions.
- **Manual Calibration**: Define the arena size by clicking 4 corners on the video frame.
- **Data Filtering**: Automated likelihood-based filtering and linear interpolation for missing data points.
- **Zone Analysis**: Calculates stay time and distance within 30% and 50% center areas.
- **Visualization**: Generates trajectory plots for each analyzed file.

## Requirements
To run this script, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `opencv-python`
- `openpyxl`

You can install them via pip:
```bash
pip install pandas numpy matplotlib opencv-python openpyxl
