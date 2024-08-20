This repository contains the dataset and associated code used in our study on the dissociation front of methane hydrates. The dataset has been made publicly available through Mendeley Data, and the code provides a detailed analysis of the hydrate dissociation process.

Dataset Overview
The dataset, named DLHDF (DL model for hydrate dissociation front), is structured into four CSV files, each detailing different aspects of the simulation data:

input.csv: Contains the model input data across 31,000 simulation cases, with each case featuring 59 properties of the Thermo-Hydro-Chemical-Mechanical (THCM) system. The data includes hydrate saturation at each element, recorded every 0.5 days over a 30-day period, resulting in 60 time points per simulation. The file has 60 columns for the 59 parameters and a time point tag, and 1,860,000 rows representing the simulation cases and time points.

output20.csv: Records the location of the 20% dissociation front for each simulation case and time point, based on the first element reaching the defined degree of dissociation among the 35 elements in the numerical model.

output50.csv: Similar to output20.csv, but for the 50% dissociation front.

output80.csv: Contains data for the 80% dissociation front, following the same structure as the other output files.

Dataset Access
The dataset can be freely accessed and downloaded from Mendeley Data at the following link:
DLHDF Dataset on Mendeley Data

Code Description
The repository includes three main code scripts, each designed to analyze the dissociation front at different thresholds:

dissociation_front20.py: Analyzes the data for the 20% dissociation front.

dissociation_front50.py: Analyzes the data for the 50% dissociation front.

dissociation_front80.py: Analyzes the data for the 80% dissociation front.

Each script processes the corresponding CSV file and provides insights into the hydrate dissociation dynamics at the specified threshold.
