# GLASS Challenge

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

Galaxy distributions are generated using the Snail distributions within GLASS.

## Masking

