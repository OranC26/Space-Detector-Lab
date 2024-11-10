# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:26:17 2024

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
            if line.startswith('0 1023'):
                Count = file.readlines()
                for element in Count:
                    if element.strip() == '$ROI:':
                        break
                    data = [float(x) for x in element.strip().split(',')]
                    Counts.extend(data)  # Use extend to ensure 1D data
    return np.array(Counts)

# Function to read background data from a file
def read_background(file_path):
    Background = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('0 1023'):
                Background1 = f.readlines()
                for element in Background1:
                    if element.strip() == '$ROI:':
                        break
                    b_data = [float(x) for x in element.strip().split(',')]
                    Background.extend(b_data)  # Use extend to ensure 1D data
    return np.array(Background)

# Function to calibrate and subtract background from counts
def process_spectrum(spectrum_file, background_file, calibration_slope):
    Counts = read_spectrum(spectrum_file)
    Background = read_background(background_file)
    New_counts = np.subtract(Counts, Background)  # Subtract background
    Channels = np.array(list(range(len(Counts))))  # Adjust for number of channels
    Energies = Channels * calibration_slope  # Convert channels to energy
    return Energies, New_counts

# Example usage with your files
background_file = 'background_BGO.SPE'
spectrum_files = ['AM241_Nal_0deg.SPE', 'AM241_Nal_30deg.Spe','AM241_Nal_60deg.SPE','AM241_Nal_90deg.SPE']  # Add your spectrum file names here
calibration_slope = 36.24 / 27  # Calibration slope based on your values

# Create a plot
plt.figure(figsize=(10, 6))

# Define angle labels corresponding to the spectrum files
angle_labels = ['0 degrees', '30 degrees', '60 degrees', '90 degrees']

# Loop through each spectrum file, process it, and plot with angle labels
for idx, spectrum_file in enumerate(spectrum_files):
    Energies, New_counts = process_spectrum(spectrum_file, background_file, calibration_slope)
    plt.plot(Energies, New_counts, label=angle_labels[idx])  # Use angle labels
# Plot formatting
max_counts = max(New_counts)
print(max_counts)

plt.xlabel('Energies (KeV)')
plt.ylabel('Counts (5 min)')
plt.title('Na(I)tl off-axis response (Am-241)')
plt.xlim(0, 100)  # Adjust x limits as needed
plt.ylim(0, np.max(New_counts)*1.1)  # Adjust y limits for better visibility
plt.legend()
plt.grid(True)
plt.show()

 


