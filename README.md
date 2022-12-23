# Finding The Minimal Energy Configuration Of Charge Particle Within A Circle with Simulated Annealing 


<p align="center">
  <img src="/Images/forces12.png" />
  <figcaption>Visualisation of the forces in the optimal state for 12 particles.</figcaption>
</p>

## Contributors:

* Paul Hosek
* Marcel van de Lagemaat

## Program Overview
This work looks at the optimal configuration of charge particles in a circle

## Requirements
* Python 3.9+
* math
* numpy
* matplotlib
* pandas
* csv

## Running the code

All results are aggregated into a single Jupyter Notebook named MarcelvandeLagemaat_10886699_PaulHosek_12637033_1.ipynb.
This notebook imports all relevant files and generates the plots and data found in the report.
The authors hope this will simplify review of the codebase at later points in time.

## Repository structure


| File Name           | Description                                                                                                                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Final Notebook | Named "MarcelvandeLagemaat_10886699_PaulHosek_12637033_1.ipynb". This notebook aggregates all the rest of the respository.|
|charges.py| Contains the charges class. This includes all code relevant to simulate various scenarios.|
|plotting.py| Contains the functions to produce the plots in the report.|
|magic.py| Contains the optimal configurations for magic numbers used in the final notebook|
|markov.py| Contains the results for the Markov chain length convergence used in the final notebook.|
|Legacy (Directory)| Contains functions no longer in use. This includes drafting spaces and functions that have been revised since.|
|Images (Directory)| Contains all plot and visualisations relevant for the report and the final notebook|
|logged_data (Directory| Contains all data logged to produce plots. Data is sorted into relevant categories named accordingly.|



<p align="center">
  <img src="/Images/final/wavy_variance_evenspacing.svg" />
  <figcaption>An example convergence chart.</figcaption>
</p>
