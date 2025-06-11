# GLASS Challenge
![image](https://github.com/user-attachments/assets/9681004a-76a8-42d5-b5d9-987fb0496d25)
This repository contains the code to generate a series of GLASS simulations and process them using Heracles. 

## Goal

The goal of this repository is to provide a set of simulations with a known ground truth that can be used to test the performance of different algorithms in the Euclid processing pipeline and to study the impact of different noise and masking scenarios on the performance of these algorithms.

### Challenges

- Unmixing: study the performance of different E/B unmixing algorithms as a result of masked observations.
- Covariance: study the level of agreement between the simulation ensemble covariance (truth), Jackknife covariance (DICES) and theoretical covariance (Spaceborne).
- Bandpowers: study the agreement between the harmonic space measurement of bandpowers and the configuration space measurement of bandpowers. 

## Theory

The GLASS simulations are generated from a known cosmology given by the CAMB default parameters setting:
- h = 0.7
- Oc = 0.25
- Ob = 0.05
Galaxy distributions are then generated using the Snail distributions within GLASS.
The first two distributions correspond to the lenses samples while the last two correspond to the sources samples.

![image](https://github.com/user-attachments/assets/41541ede-6c0c-49d8-86ab-d079fbce5886)

The resulting theoretical angular power spectra can be seen below.

![image](https://github.com/user-attachments/assets/fbf8ca5d-5789-4692-910f-caf079ca014d)
![image](https://github.com/user-attachments/assets/53e58a99-1480-4cf1-b6cb-07597f9bf4ca)
![image](https://github.com/user-attachments/assets/42488821-3a4e-4825-acc3-946aa9800ca5)

## GLASS Settings

We use GLASS to simulate lognormal fields from the CAMB theoretical power spectra. 
To do: exampnd on this.


## Masking

We use the Euclid RR2 footprint shown below:

![image](https://github.com/user-attachments/assets/b8a1bf3d-4bd7-4aa9-af7c-c6cb296ce66c)

