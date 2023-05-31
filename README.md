# Hyperelasticity_code
This project is based on the topic of **"Hyperelastic material model discovery"** using reaction force and displacement data. For this project we have used FEniCS module to generate the data set for two different benchmark hyperelastic material model namely _NeoHookean_ and _Isihara_ models using a 2-D geometry of a plate with bi-axial displacement boundary conditions. After generating the data we can add noise to the displacement data using a Normal distribution with standard deviation of '1e-4 for low noise' and '1e-3 for high noise', and then denoising the data using Kernel Ridge Regression (KRR). Using the six data sets (3 cases each i.e. no noise, low noise and high noise level for two benchmark models) we can train our code to predict actual material model used to generate the data in the first place using the Variational Bayesian (VB) algorithm and then compare the time taken and predicted material models by VB and MCMC (Markov Chain Monte Carlo) algorithms. Below are the steps to follow:

Let us consider a benchmark material model named as NeoHookean material model for illustration-

**Step 1: Generate the data using the file named as 'NeoHookean_J2.py' present in 'data_generation' folder.**

_This will generate a data set which contain reaction force (at every sides of the plate) and displacement data (of every nodes) along with the connectivity matrix and initial location of each nodes. The plate is stretched unsymmetrically in two directions ('d/2' in x-direction and 'd' in y-direction) where 'd' is the percentage stretched at a particular loadstep. For this we have considered 6 loasteps (d = 10,20,30,...,60) for data generations and then use those data further. The initial location and displacement of each nodes are present in the file named as 'Coor_and_disp.csv' and reaction forces are stored in 'reaction.csv' file in the folder named as 'FEM_data/NeoHookean_J2' for each loadstep._


**Step 2: Add noise and then denoise the generated displacement data.**

_This will generate the denoised data in which first noise is added to displacement data and then denoised using KRR which then saved to the same file where the actual displacement data is present for each loadstep._

Let us consider a case with "no noise" (without noise) conditions such that the actual reaction force and actual displacement data set is used for the simulation. 

**Step 3: Estimate the material model using the data**

_Open the 'main_new_2.m' file enter material as 'NeoHookean_J2' and noiselevel as 'no' and then run the fill will give the estimated material model for the data set we have used i.e. NeoHookean material model._

Step 3 can be used for material model estimation using low and high noise conditions for NeoHookean model. Just change noiselevel as 'low' or 'high' respectively, in the 'main_new.m' file.

Similarly, it can be done for Isihara material model.
