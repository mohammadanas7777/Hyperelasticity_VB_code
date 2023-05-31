clc; clear; close all
%% This file runs a MCS for Duffing oscillator
% addpath('..\myBVS_functions');
addpath('SB2_Release_200');
addpath('myBVS_functions');

type = 'Duffing';

%% Data generation
fs      = 1000;             % Sampling frequency
Tf      = 4;                % Total time duration
tspan   = (0:1/fs:Tf-1/fs)';
Nt      = length(tspan);
noisePercent = 5/100;       % Percentage of noise added

% Define inputs to the SDOF system
inputType = 'random';  % Random vs sinusoidal
% inputType = 'sine';  % Random vs sinusoidal
inpRnd = strcmpi(inputType, 'random');
if(inpRnd)
    inpAmp  = 100;
    fcut    = 100; % Hz
    [bb,aa] = butter(6,fcut/(fs/2), 'low');
else
    inpAmp  = 100;
    inpFreq = 10;
end

rndNum = 870122; %randi(10000000); %53927088;
rng(rndNum);
modelParam      = struct();
modelParam.k    = abs(1e3 + 0e2*randn);  % Stiffness of the system (in N/m)
modelParam.c    = abs(2 + 0*randn);    % Damping of the system (in Ns/m)
modelParam.m    = 1;        % Mass of the system (in kg)

% Nonlinear terms
modelParam.pend = 0;        % Sinusoid disp nonlinearity
modelParam.cs   = 0;        % Signum vel nonlinearity
modelParam.c2   = 0;        % Squared vel nonlinearity
modelParam.g1   = abs(1e5 + 0*randn);        % Cubic disp nonlinearity

simParam        = struct();
simParam.fs     = fs;
simParam.tspan  = tspan;    % Time series  
simParam.inpRnd = inpRnd;   % if true (then random exc)
simParam.noisePercent = noisePercent;
if(inpRnd)
    utmp        = inpAmp*randn(Nt,1);
    simParam.u  = filtfilt(bb,aa,utmp);
else
    simParam.inpAmp     = inpAmp;   % Sinusoidal exc
    simParam.inpFreq    = inpFreq;
end


DS = get_test_train_data3(modelParam, simParam);
DS.rndNum = rndNum;
    
% Get data ready
Dtr   = DS.Dtrain;
Dtest = DS.Dtest;
scalD = DS.stdDtr;

ytr    = DS.ytrain;
ytest  = DS.ytest;
wtr    = DS.wtrain;
wtr    = wtr./scalD;

%% Equation discovery algorithms
% Fit RVM with Tipping's RVM code...

tic
[wmu_RVM, sigma_RVM, model_RVM] = run_my_RVM(ytr, Dtr, 10000);
wmu_RVM = wmu_RVM./scalD;
tr = toc;

% RVM with gamma diagnostic > 0.9
wmu_RVMg   = zeros(size(wmu_RVM));
goodidx    = model_RVM.parameters.Relevant(model_RVM.diagnostic.Gamma>0.9);
wmu_RVMg(goodidx) = wmu_RVM(goodidx);

nSamples = 5000;
nBurnin  = 1000;
nchains  = 5;

% Using BASAD
% disp('Running BASAD...');
% out_basad = BVS_basad2(Dtr, ytr, nSamples, nBurnin, nchains, false);
% wmu_BASAD = zeros(size(wmu_RVM));
% wmu_BASAD(out_basad.modelIdx) = out_basad.Wsel;
% wmu_BASAD = wmu_BASAD./scalD;
% GRstatsb  = out_basad.Rw;
% disp('BASAD done');


% Using DISSP independent slab
% disp('Running Independent Spike and Slab...');
% out_disspI = BVS_Walli_I(Dtr, ytr, nSamples, nBurnin, nchains, false);
% wmu_DISSPi = zeros(size(wmu_RVM));
% wmu_DISSPi(out_disspI.modelIdx) = out_disspI.Wsel;
% wmu_DISSPi = wmu_DISSPi./scalD;
% GRstatsi   = out_disspI.Rw;
% disp('Independent Spike and Slab done');

% Using DISSP Zellner's g-slab
% disp('Running Correlated Spike and Slab...');
% out_disspG = BVS_Walli_G(Dtr, ytr, nSamples, nBurnin, nchains, false);
% wmu_DISSPg = zeros(size(wmu_RVM));
% wmu_DISSPg(out_disspG.modelIdx) = out_disspG.Wsel;
% wmu_DISSPg = wmu_DISSPg./scalD;
% GRstatsg   = out_disspG.Rw;
% disp('Correlated Spike and Slab done');


% Using DISSPi normal slab
disp('Running independent normal-dist-slab Spike and Slab...');
out_disspIns = BVS_Walli_I_normalslab(Dtr, ytr, nSamples, nBurnin, nchains, false);
wmu_DISSPins = zeros(size(wmu_RVM));
wmu_DISSPins(out_disspIns.modelIdx) = out_disspIns.Wsel;
wmu_DISSPins = wmu_DISSPins./scalD;
GRstatsins   = out_disspIns.Rw;
disp('independent normal-dist-slab Spike and Slab done');

% Using DISSPg normal slab
disp('Running correlated normal-dist-slab Spike and Slab...');
out_disspGns = BVS_Walli_G_normalslab(Dtr, ytr, nSamples, nBurnin, nchains, false);
wmu_DISSPgns = zeros(size(wmu_RVM));
wmu_DISSPgns(out_disspGns.modelIdx) = out_disspGns.Wsel;
wmu_DISSPgns = wmu_DISSPgns./scalD;
GRstatsgns   = out_disspGns.Rw;
disp('correlated normal-dist-slab Spike and Slab done');

% Use independent spike and slab VB John Ormerod
disp('Running independent VB Spike and Slab...');
initz = zeros(size(wmu_RVM));
initz(model_RVM.parameters.Relevant) = model_RVM.diagnostic.Gamma;
tol = 1e-6;
out_disspI_vb = BVS_disspI_vb_John(Dtr, ytr, initz, tol, false);
wmu_DISSPi_vb = zeros(size(wmu_RVM));
wmu_DISSPi_vb(out_disspI_vb.modelIdx) = out_disspI_vb.Wsel;
wmu_DISSPi_vb = wmu_DISSPi_vb./scalD;
disp('independent VB Spike and Slab done');

% Use DISSP independent slab VB Rajdip Student's t-slab
out_disspI_vb2 = BVS_disspI_vb_Stud(Dtr, ytr, initz, tol, false);
wmu_DISSPi_vb2 = zeros(size(wmu_RVM));
wmu_DISSPi_vb2(out_disspI_vb2.modelIdx) = out_disspI_vb2.Wsel;
wmu_DISSPi_vb2 = wmu_DISSPi_vb2./scalD;

% Post inclusion probability means
z_BASAD    = mean(out_basad.ZZ);
z_DISSPi   = mean(out_disspI.ZZ);
z_DISSPg   = mean(out_disspG.ZZ);
z_DISSPins = mean(out_disspIns.ZZ);
z_DISSPgns = mean(out_disspGns.ZZ);

% Train set coefficient error
werr_RVM        = norm(wmu_RVM  - wtr)/norm(wtr);
werr_RVMg       = norm(wmu_RVMg - wtr)/norm(wtr);
werr_BASAD      = norm(wmu_BASAD - wtr)/norm(wtr);
werr_DISSPi     = norm(wmu_DISSPi - wtr)/norm(wtr);
werr_DISSPg     = norm(wmu_DISSPg - wtr)/norm(wtr);
werr_DISSPgns   = norm(wmu_DISSPgns - wtr)/norm(wtr);
werr_DISSPins   = norm(wmu_DISSPins - wtr)/norm(wtr);
werr_DISSPi_vb  = norm(wmu_DISSPi_vb  - wtr)/norm(wtr);
werr_DISSPi_vb2 = norm(wmu_DISSPi_vb2 - wtr)/norm(wtr);
    
vars   = DS.ss'; 
Tab_w  = table(vars, wtr, wmu_RVM, wmu_RVMg, wmu_BASAD, wmu_DISSPi, wmu_DISSPg,...
               wmu_DISSPins, wmu_DISSPgns, wmu_DISSPi_vb, wmu_DISSPi_vb2);
