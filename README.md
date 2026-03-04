# Cov Matrix MC

Monte Carlo framework for studying covariance matrix estimation and
feasible GLS estimators in **high‑dimensional panel models with
multi‑way effects**.

This repository implements a simulation environment used to evaluate the
performance of **OLS, GLS, and feasible GLS estimators** when the true
error covariance matrix contains **multiple structured components**.

The project is written in **Julia** and is designed for reproducible
econometric Monte‑Carlo experiments.

------------------------------------------------------------------------

# Motivation

In multi‑way panel datasets indexed by

-   \(i\) --- spacial unit 1
-   \(j\) --- spacial unit 2
-   \(t\) --- time

The error structure often contains **multiple sources of correlation**.
Standard estimators such as OLS or cluster‑robust estimators may perform
poorly when the covariance matrix has a structured block form.

This project studies the properties of estimators when the covariance
matrix takes the form

\[ `\Omega `{=tex}=
S\_`\alpha `{=tex}`\Omega`{=tex}*`\alpha `{=tex}S*`\alpha`{=tex}' +
S\_`\gamma `{=tex}`\Omega`{=tex}*`\gamma `{=tex}S*`\gamma`{=tex}' +
S\_`\lambda `{=tex}`\Omega`{=tex}*`\lambda `{=tex}S*`\lambda`{=tex}' +
`\sigma`{=tex}\_u\^2 I \]

where

-   (S\_`\alpha`{=tex}, S\_`\gamma`{=tex}, S\_`\lambda`{=tex}) map
    observations to effect dimensions
-   (`\Omega`{=tex}*`\alpha`{=tex}, `\Omega`{=tex}*`\gamma`{=tex},
    `\Omega`{=tex}\_`\lambda`{=tex}) describe covariance across units
-   (u\_{ijt}) is idiosyncratic noise

The objective is to understand how well different estimators approximate
this covariance structure and recover the slope parameter (
`\beta `{=tex}).

------------------------------------------------------------------------

# Model

The data generating process follows

\[ y\_{ijt} = `\beta `{=tex}x\_{ijt} + `\alpha`{=tex}\_i +
`\gamma`{=tex}\_j + `\lambda`{=tex}*t + u*{ijt} \]

where

  Component              Description
  ---------------------- ----------------------------
  (x\_{ijt})             observed regressor
  (`\alpha`{=tex}\_i)    effect along dimension (i)
  (`\gamma`{=tex}\_j)    effect along dimension (j)
  (`\lambda`{=tex}\_t)   time effect
  (u\_{ijt})             idiosyncratic error

The stacked error vector therefore has a **structured covariance
matrix**.

------------------------------------------------------------------------

# Estimators Studied

The Monte Carlo experiments compare several estimators.

### OLS with Fixed Effects

Standard within estimator removing multi‑way fixed effects.

Used as the **baseline estimator**.

------------------------------------------------------------------------

### Diagonal GLS

Assumes

\[`\Omega `{=tex}= `\sigma`{=tex}\^2 I\]

This estimator ignores correlation across observations.

------------------------------------------------------------------------

### FGLS1

One‑step feasible GLS procedure

1.  Estimate residuals using OLS
2.  Estimate covariance components
3.  Construct ( `\hat{\Omega}`{=tex} )
4.  Run GLS

------------------------------------------------------------------------

### FGLS2

Extended feasible GLS estimator using additional moment conditions and
differencing schemes to estimate covariance components.

------------------------------------------------------------------------

### Oracle GLS

GLS estimator using the **true covariance matrix from the DGP**.

This provides the **efficiency benchmark**.

------------------------------------------------------------------------

# Repository Structure

    Cov Matrix MC
    │
    ├── main.jl
    ├── params.jl
    ├── io.jl
    ├── dgp.jl
    ├── beta_estimators.jl
    ├── omega_estimators.jl
    ├── estimation.jl
    ├── rc_functions.jl
    ├── diagnostics.jl
    ├── plotting.jl
    ├── mc_driver.jl
    └── smoke_test.jl

### main.jl

Entry point of the project.

Initializes parameters and runs the Monte Carlo experiment.

------------------------------------------------------------------------

### params.jl

Defines all simulation parameters including

-   panel sizes (N_1, N_2, T)
-   covariance structure
-   number of Monte Carlo replications
-   model parameters

------------------------------------------------------------------------

### dgp.jl

Implements the **data generating process**.

Responsibilities:

-   draw random effects
-   generate regressors
-   generate outcome variable
-   construct covariance components

------------------------------------------------------------------------

### beta_estimators.jl

Contains estimators for the slope parameter ( `\beta `{=tex}).

Includes

-   OLS
-   fixed‑effects OLS
-   GLS routines
-   variance estimators

------------------------------------------------------------------------

### omega_estimators.jl

Implements covariance matrix estimators used for feasible GLS.

Includes

-   diagonal estimators
-   block covariance estimators
-   cluster outer‑product estimators

------------------------------------------------------------------------

### estimation.jl

Combines beta and covariance estimation to implement **feasible GLS
estimators**.

------------------------------------------------------------------------

### mc_driver.jl

Controls the Monte Carlo simulation loop.

Typical steps:

1.  simulate data
2.  estimate parameters
3.  store results
4.  repeat across replications

------------------------------------------------------------------------

### diagnostics.jl

Tools for evaluating estimator performance

Examples

-   bias diagnostics
-   empirical variance calculations
-   covariance decomposition

------------------------------------------------------------------------

### plotting.jl

Utilities for generating figures from simulation results.

------------------------------------------------------------------------

### io.jl

Handles reading and writing of simulation results.

Uses **JLD2** for storing Monte Carlo outputs.

------------------------------------------------------------------------

### smoke_test.jl

Minimal script used to verify that the full pipeline runs correctly.

Useful after modifying core code.

------------------------------------------------------------------------

# Running the Simulation

Run the Monte Carlo experiment

    julia main.jl

Quick pipeline test

    julia smoke_test.jl

------------------------------------------------------------------------

# Dependencies

Main Julia packages

    DataFrames
    LinearAlgebra
    Statistics
    Random
    JLD2

Install packages

    using Pkg
    Pkg.add(["DataFrames","JLD2"])

------------------------------------------------------------------------

# Output

The simulation produces

-   estimated coefficients
-   empirical estimator variance
-   bias comparisons
-   covariance estimation diagnostics

Results are stored as **JLD2 files** and can be analyzed using the
plotting utilities.

------------------------------------------------------------------------

# Research Context

This codebase supports research on

-   covariance estimation in high‑dimensional panels
-   feasible GLS estimators
-   multi‑way fixed effects models

and is used for Monte Carlo experiments evaluating estimator properties
under different panel dimensions and covariance structures.

------------------------------------------------------------------------

# Author

**Ramzi Chariag**\
PhD Candidate in Economics\
Central European University
