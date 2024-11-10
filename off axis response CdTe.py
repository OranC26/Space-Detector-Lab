# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:52:47 2024

@author: oranc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pandas import DataFrame

# Function to read spectrum data from a file
def read_spectrum(file_path):
    Counts = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('<<DATA>>'):
                Count = file.readlines()
                for element in Count:
                    if element.strip() == '<<END>>':
                        break
                    data = [float(x) for x in element.strip().split(',')]
                    Counts.extend(data)  # Use extend to ensure 1D data
    return np.array(Counts)


# Function to read background data from a file
def read_background(file_path):
    Background = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('<<DATA>>'):
                Background1 = f.readlines()
                for element in Background1:
                    if element.strip() == '<<END>>':
                        break
                    b_data = [float(x) for x in element.strip().split(',')]
                    Background.extend(b_data)  # Use extend to ensure 1D data
    return np.array(Background)

# Function to calibrate and subtract background from counts
def process_spectrum(spectrum_file, background_file, calibration_slope):
    Counts = read_spectrum(spectrum_file)
    Background = read_background(background_file)
    New_counts = np.subtract(Counts, Background)  # Subtract background
    New_counts[New_counts < 0] = 1e-10
    Channels = np.array(list(range(len(Counts))))  # Adjust for number of channels
    Energies = Channels * calibration_slope  # Convert channels to energy
    return Energies, New_counts

# Example usage with your files
background_file = 'CDTE_Background_0deg_900.mca'
spectrum_files = ['Am241_0deg_600sec.mca', 'Am241_30deg_600sec.mca','Am241_60deg_600sec.mca','Am241_90deg_600sec.mca']  # Add your spectrum file names here
calibration_slope = 59.5 / 1200  # Calibration slope based on your values

# Create a plot
plt.figure(figsize=(10, 6))

# Define angle labels corresponding to the spectrum files
angle_labels = ['0 degrees', '30 degrees', '60 degrees', '90 degrees']

# Loop through each spectrum file, process it, and plot with angle labels
for idx, spectrum_file in enumerate(spectrum_files):
    Energies, New_counts = process_spectrum(spectrum_file, background_file, calibration_slope)
    plt.plot(Energies, New_counts, label=angle_labels[idx])  # Use angle labels
# Plot formatting
plt.xlabel('Energies (KeV)')
plt.ylabel('Counts (10 min)')
plt.title('CdTe off-axis response (Am-241)')
plt.xlim(55, 60)  # Adjust x limits as needed
plt.ylim(0, )  # Adjust y limits for better visibility
plt.legend()
plt.grid(True)
plt.show()

 
