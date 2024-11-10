# Space-Detector-Lab
# Python for space detector lab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os

""" Define the detectors data as dataframes"""

BGO_dict = {
    'Sample': ['Cs-137', 'Am-241', 'Ba-133', 'Ba-133(2)', 'Co-60', 'Co-60(2)'],
    'Photopeak Channel': [336, 27, 40, 180, 600, 710],
    'Energy': [661.65, 36.24, 80.99, 356.01, 1173.22, 1332.49],
    'FWHM': [80.336, 13.58, 40.326, 78.725, 147.57, 152.47],
    'Resolution': [ 0.1214, 0.5159, 0.5189, 0.2152, 0.1069, 0.1168],
}

Nal_dict = {
    'Sample': ['Cs-137', 'Am-241', 'Ba-133', 'Ba-133(2)', 'Co-60'],
    'Photopeak Channel': [285, 29, 41, 180, 600],
    'Energy': [661.65, 59.54, 79.61, 356.01, 1332.49],
    'FWHM': [97.92, 22.48, 37.57, 72.28, 553.73],
    'Resolution': [0.1483, 0.3763, 0.4703, 0.208, 0.1727],
}

#NOTE for th CdTe detector due to undiscernible CO60 peaks TWO peaks from AM were chosen and 2 peaks from barium 
CdTe_dict = {'Sample': ['Am-241(1)' ,'Am-241(2)','Am-241(3)','Ba-133','Ba-133(2)','Cs-137'],
             'Photopeak Channel':[350, 510, 1200 ,618, 1600,639 ],
             'Energy': [ 17.8, 26.3, 59.5 , 31 , 80.99, 661.65],
             'FWHM': [ 0.488, 0.506 , 0.661 , 0.727 , 1.68, 72.05],
             'Resolution':[ 0.027 , 0.01923 , 0.0111, 0.0234, 0.0207, 0.1089] }


# Create DataFrames for each detector type
BGO_df = pd.DataFrame(BGO_dict)
Nal_df = pd.DataFrame(Nal_dict)
CdTe_df = pd.DataFrame(CdTe_dict)

# calculated efficiency data for each detector
Energies_I = np.array([59, 661.65, 80.99, 356.01])   # In keV
efficiency_bgo = np.array([0.0002187, 0.000095, 0.000234, 0.000133]) #Am, Cs, Ba, Ba 
efficiency_NaI = np.array([0.000125, 0.0000322, 0.000208, 0.0000919])
efficiency_CdTe = np.array([4.25e-7, 2.9e-7, 1.67e-6, 5.01e-7 ])

intrinsic_bgo = np.array([0.0059, 0.00257, 0.00632, 0.003614])
intrinsic_NaI = np.array([0.00338, 0.000872, 0.00564, 0.00248 ])
intrinsic_CdTe = np.array([0.000019, 0.0000132, 0.00007599, 0.00002279 ])


# Create a dictionary to store the data and fit results
detectors_I = {
    'BGO': {'efficiency': intrinsic_bgo, 'color': 'red', 'marker': '+'},
    'NaI(Tl)': {'efficiency': intrinsic_NaI, 'color': 'black', 'marker': '^'},
    'CdTe': {'efficiency': intrinsic_CdTe, 'color': 'green', 'marker': 'o'}
}

# Prepare a dictionary to store the data and fit results
detectors = {
    'BGO': {'efficiency': efficiency_bgo, 'color': 'red', 'marker': '+'},
    'NaI(Tl)': {'efficiency': efficiency_NaI, 'color': 'black', 'marker': '^'},
    'CdTe': {'efficiency': efficiency_CdTe, 'color': 'green', 'marker': 'o'}
}


#Define a function that allows the user to select a sample 
def select_sample_and_roi(detector_df):
    """Prompt the user to select a sample and retrieve the corresponding ROI."""
    samples = detector_df['Sample'].tolist()
    print(f"Available samples: {samples}")
    
    # User selects a sample
    sample = input("Please enter the sample you want to analyze: ")
    
    # Check if the sample exists in the dataframe
    if sample not in samples:
        print("Invalid sample selected. Exiting.")
        return None, None

    # Get the corresponding ROI
    roi_row = detector_df[detector_df['Sample'] == sample]
    
    # Adjust ROI based on detector type
    if detector_df is CdTe_df:
        # For CdTe, use ±10 keV as the ROI
        roi = (roi_row['Energy'].values[0] - 2, roi_row['Energy'].values[0] + 2)
    else:
        # Default ROI for other detectors
        roi = (roi_row['Energy'].values[0] - 40, roi_row['Energy'].values[0] + 40)  # Example ROI range
    
    print(f"Selected sample: {sample}, ROI: {roi}")
    
    return sample, roi


def read_background_BGO(file):
    """Read the background data for BGO and NaI detectors."""
    background_list = []
    for line in file:
        if line.startswith('0 1023'):
            background_data = file.readlines()
            for element in background_data:
                if element.strip() == '$ROI:':
                    break
                b_data = [float(x) for x in element.strip().split(',')]
                background_list.extend(b_data)
    return background_list

def read_background_CdTe(file):
    """Read the background data for CdTe detectors."""
    background_list = []
    for line in file:
        if line.startswith('<<DATA>>'):
            background_data = file.readlines()
            for element in background_data:
                if element.strip() == '<<END>>':
                    break
                b_data = [float(x) for x in element.strip().split(',')]
                background_list.extend(b_data)
    return background_list

def read_counts_BGO(file):
    """Read the counts data for BGO and NaI detectors."""
    counts_list = []
    for line in file:
        if line.startswith('0 1023'):
            counts_data = file.readlines()
            for element in counts_data:
                if element.strip() == '$ROI:':
                    break
                data = [float(x) for x in element.strip().split(',')]
                counts_list.extend(data)
    return counts_list

def read_counts_CdTe(file):
    """Read the counts data for CdTe detectors."""
    counts_list = []
    for line in file:
        if line.startswith('<<DATA>>'):
            counts_data = file.readlines()
            for element in counts_data:
                if element.strip() == '<<END>>':
                    break
                data = [float(x) for x in element.strip().split(',')]
                counts_list.extend(data)
    return counts_list

def process_spectrum(filename, detector_type):
    """Process the spectrum data from the specified filename."""
    counts_list = []
    background_list = []

    # Set the number of channels based on detector type
    if detector_type.lower() in ['bgo', 'nai']:
        num_channels = 1024
    elif detector_type.lower() == 'cdte':
        num_channels = 2048
    else:
        print("Invalid detector type specified.")
        return None

    # Read in the background for the Detector
    try:
        if detector_type.lower() == 'bgo':
            background_filename = 'background_BGO.SPE'
            with open(background_filename, 'r') as f:
                background_list = read_background_BGO(f)
                
        elif detector_type.lower() == 'nai':
            background_filename = 'background_NaTL.SPE'
            with open(background_filename, 'r') as f:
                background_list = read_background_BGO(f)
                
        elif detector_type.lower() == 'cdte':
            background_filename = 'CDTE_Background_0deg_900.mca'
            with open(background_filename, 'r') as f:
                background_list = read_background_CdTe(f)
                
    except Exception as e:
        print(f"Error reading background data: {e}")
        return None

    # Read in the Spectrum Data
    try:
        if detector_type.lower() in ['bgo', 'nai']:
            with open(filename, 'r') as file:
                counts_list = read_counts_BGO(file)
        elif detector_type.lower() == 'cdte':
            with open(filename, 'r') as file:
                counts_list = read_counts_CdTe(file)
    except Exception as e:
        print(f"Error reading spectrum data: {e}")
        return None

    # Convert to numpy arrays
    Counts = np.array(counts_list)
    Background = np.array(background_list)

    # Ensure the counts and background arrays are of the correct size
    Counts = np.pad(Counts, (0, max(0, num_channels - len(Counts))), mode='constant')
    Background = np.pad(Background, (0, max(0, num_channels - len(Background))), mode='constant')

    # Adjust background scaling for CdTe
     
    if detector_type.lower() == 'cdte':
        Background *= 1.3333  # Scale the background for CdTe,

    # Subtract the background from the Counts
    New_counts = np.subtract(Counts[:num_channels], Background[:num_channels])

    # Replace negative values with a very small positive number for log scale
    New_counts[New_counts < 0] = 1e-10  # Replace negative values or zeros

    return New_counts

def calibrated_spectrum(ax, Energies, New_counts, xlabel='Energies (keV)', ylabel='Counts', title='Calibrated Spectrum', **kwargs):
    """Plot the calibrated spectrum."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return ax.scatter(Energies, New_counts, **kwargs)

def plot_model(ax, model, xrange, ps, npoints=1000, **kwargs):
    """Plots a model on an Axes smoothly over a specified range."""
    _Energies = np.linspace(*xrange, npoints)
    _New_counts = model(_Energies, *ps)
    return ax.plot(_Energies, _New_counts, **kwargs)

def Gaussian(x, mu, sig, A):
    """Gaussian model function."""
    return (A / (np.sqrt(2 * np.pi) * sig)) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))

#Model for fitting Resolution v Energy
def Resolution_model(E,a,b,c):
    return a*E**-2 + b*E**-1 + c

# Function to calculate uncertainties in resolution
def calculate_resolution_uncertainties(energies, popt, pcov):
    # Unpack fitted parameters and their uncertainties
    a, b, c = popt
    sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(pcov))
    
    # Calculate resolution values and uncertainties for each energy point
    resolutions = Resolution_model(energies, a, b, c)
    uncertainties = np.sqrt((energies**-2 * sigma_a)**2 + (energies**-1 * sigma_b)**2 + (1 * sigma_c)**2)
    
    return resolutions, uncertainties

def simple_model_fit(model, Energies, New_counts, roi, **kwargs):
    """Least squares estimate of model parameters."""
    # Select relevant channels & counts
    _Energies, _New_counts = filter_in_interval(Energies, New_counts, *roi)
    
    # Fit the model to the data
    popt, pcov = curve_fit(model, _Energies, _New_counts, **kwargs)
    return popt, pcov

def filter_in_interval(x, y, xmin, xmax):
    """Selects only elements of x and y where xmin <= x < xmax."""
    _mask = (xmin <= x) & (x < xmax)
    return x[_mask], y[_mask]

def format_result(params, popt, pcov):
    """Display parameter best estimates and uncertainties."""
    # extract the uncertainties from the covariance matrix
    perr = np.sqrt(np.diag(pcov))
    
    # format parameters with best estimates and uncertainties
    
    _lines = (f"{p} = {o} ± {e}" for p, o, e in zip(params, popt, perr))
    return "\n".join(_lines)

def plot_efficiency(Energies_I, detectors, intrinsic=False):
    plt.figure(figsize=(8, 6))
    for name, properties in detectors.items():
        efficiency = properties['efficiency']
        log_E = np.log(Energies_I)
        log_efficiency = np.log(efficiency)

        coefficients = np.polyfit(log_E, log_efficiency, 2)
        a, b, c = coefficients[2], coefficients[1], coefficients[0]

        log_E_fit = np.linspace(min(log_E), max(log_E), 100)
        log_efficiency_fit = a + b * log_E_fit + c * log_E_fit**2
        efficiency_fit = np.exp(log_efficiency_fit)

        plt.scatter(Energies_I, efficiency, color=properties['color'], marker=properties['marker'], label=f'{name}')
        plt.plot(np.exp(log_E_fit), efficiency_fit, color=properties['color'], linestyle='--')

    plt.yscale('log')
    plt.title(f"{'Intrinsic ' if intrinsic else ''}Peak Efficiency vs Energy")
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intrinsic Efficiency' if intrinsic else 'Efficiency')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# Refactored function for plotting Resolution vs Energy and calculating uncertainties
def plot_resolution():
    # List of detectors with their corresponding dataframes and labels
    detectors = [
        ('BGO', BGO_df),
        ('NaI(Tl)', Nal_df),
        ('CdTe', CdTe_df)
    ]
    
    # Prepare for plotting
    plt.figure(figsize=(8, 6))
    
    for label, df in detectors:
        # Extract energy and resolution for the detector
        Photopeak_Energy = df['Energy']
        Resolutions = df['Resolution']
        
        # Perform curve fitting
        popt, pcov = curve_fit(Resolution_model, Photopeak_Energy, Resolutions)
        E_fit = np.linspace(min(Photopeak_Energy), max(Photopeak_Energy), 500)
        Resolution_fit = Resolution_model(E_fit, *popt)
        
        # Plot the data and the fitted curve
        plt.scatter(Photopeak_Energy, Resolutions, label=f'{label} Data', marker='+')
        plt.plot(E_fit, Resolution_fit, linestyle='--')

        # Calculate and add resolution uncertainties
        resolutions, uncertainties = calculate_resolution_uncertainties(Photopeak_Energy, popt, pcov)
        df['Resolution Uncertainty'] = uncertainties

        # Print the results
        print(f"{label} Detector Resolutions and Uncertainties:")
        print(df[['Energy', 'Resolution', 'Resolution Uncertainty']])
        print("\n")
    
    # Finalize plot settings
    plt.title('Resolution vs Energy')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Resolution (keV)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_calibration():
    #Plot the calibration Curve for CdTe
    plt.figure(figsize=(8,6))
    plt.scatter(CdTe_df['Photopeak Channel'], CdTe_df['Energy'], color='g', marker='o')
    z = np.polyfit(CdTe_df['Photopeak Channel'], CdTe_df['Energy'], 1)
    p = np.poly1d(z)
    plt.plot(CdTe_df['Photopeak Channel'], p(CdTe_df['Photopeak Channel']), 'r--' )
    plt.xlabel('Photopeak Channel')
    plt.ylabel('Energy (Kev)')
    plt.title('Calibration Curve for CdTe detector')
    plt.grid(True)
    plt.show()

    #plot the Calibration Curve for NaTL
    plt.figure(figsize=(8,6))
    plt.scatter(Nal_df['Photopeak Channel'], Nal_df['Energy'], color='b', marker='o')
    z = np.polyfit(Nal_df['Photopeak Channel'], Nal_df['Energy'], 1)
    p = np.poly1d(z)
    plt.plot(Nal_df['Photopeak Channel'], p(Nal_df['Photopeak Channel']), 'r--' )
    plt.xlabel('Photopeak Channel')
    plt.ylabel('Energy (Kev)')
    plt.title('Calibration Curve for NaTL detector')
    plt.grid(True)
    plt.show()


    """Plot the Calibration curve BGO, For this we plot Channel v Energy for each sample"""
    plt.figure(figsize=(8,6))
    plt.scatter(BGO_df['Photopeak Channel'], BGO_df['Energy'], color='b', marker='o' )
    #Add a trendline
    z = np.polyfit(BGO_df['Photopeak Channel'], BGO_df['Energy'], 1)
    p = np.poly1d(z)
    plt.plot(BGO_df['Photopeak Channel'], p(BGO_df['Photopeak Channel']), 'r--' )
    plt.xlabel('Photopeak Channel')
    plt.ylabel('Energy (Kev)')
    plt.title('Calibration Curve for BGO detector')
    plt.grid(True)
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze gamma-ray spectrum data')
    parser.add_argument('filename', type=str, help="Input spectrum file to analyze")
    parser.add_argument('detector', type=str, help="Detector type (BGO, NaI, or CdTe)")
    parser.add_argument('--plot_efficiency', action='store_true', help="Display efficiency plots")
    parser.add_argument('--plot_resolution', action='store_true', help="Display Resolution Energy plot")
    parser.add_argument('--plot_calibration', action='store_true', help='Display calibration curves for each detector')
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.filename):
        print(f"File {args.filename} not found.")
        return

    # Select the appropriate detector dataframe based on user input
    if args.detector.lower() == 'bgo':
        detector_df = BGO_df
    elif args.detector.lower() == 'nai':
        detector_df = Nal_df
    elif args.detector.lower() == 'cdte':
        detector_df = CdTe_df
    else:
        print("Invalid detector type specified. Please use 'BGO', 'NaI', or 'CdTe'.")
        return

    # Select sample and ROI
    sample, roi = select_sample_and_roi(detector_df)
    if roi is None:
        return
    
    # Process the spectrum data
    New_counts = process_spectrum(args.filename, args.detector)
    if New_counts is None:
        return

    # Assign channels to energy using calibration slope
    if args.detector.lower() in ['bgo', 'nai']:
        Channels = np.array(list(range(1024)))  # 1024 channels for BGO and NaI
    elif args.detector.lower() == 'cdte':
        Channels = np.array(list(range(2048)))  # 2048 channels for CdTe

    measured_photopeak_channel = detector_df[detector_df['Sample'] == sample]['Photopeak Channel'].values[0]
    expected_photopeak = detector_df[detector_df['Sample'] == sample]['Energy'].values[0]
    calibration_slope = expected_photopeak / measured_photopeak_channel
    Energies = Channels * calibration_slope

    # Check for matching sizes
    if len(Energies) != len(New_counts):
        print("Error: Energies and New_counts arrays are of different lengths.")
        return

    # Perform Gaussian fitting if desired
    Gaussian_params = ('mu', 'sig', 'A')
    ROI = roi  # Use the ROI obtained earlier

    _Energies, _New_counts = filter_in_interval(Energies, New_counts, *ROI)
    p0 = (expected_photopeak, 10, max(_New_counts))  # Initial estimates

    try:
        popt, pcov = simple_model_fit(Gaussian, Energies, New_counts, ROI, p0=p0)
    except Exception as e:
        print(f"Error during fitting: {e}")
        return

    # Display fitted parameters
    print("> the final fitted estimates:")
    print(format_result(Gaussian_params, popt, pcov))

    # Plot the Gaussian fit
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    calibrated_spectrum(ax, Energies, New_counts, s=10, marker='+')
    plot_model(ax, Gaussian, (min(Energies), max(Energies)), popt, color='red', label='Gaussian Fit')
    plt.xlabel('Energies (keV)')
    plt.ylabel('Counts')
    plt.title(f'Fitted Gaussian for {sample} ({args.detector})')
    plt.legend()
    plt.grid(True)
    plt.show()


    """User input to display Efficiency"""
    # Conditionally display the efficiency plots if the user requested it
    if args.plot_efficiency:
        # Efficiency calculation and plotting code
        plot_efficiency(Energies_I, detectors)
        plot_efficiency(Energies_I, detectors_I, intrinsic=True)
        
    """User input to display Resolution"""
    if args.plot_resolution:
        plot_resolution()
        
    """User input to display the Calibration Curves"""
    if args.plot_calibration:
        plot_calibration()

        
if __name__ == "__main__":
    main()
