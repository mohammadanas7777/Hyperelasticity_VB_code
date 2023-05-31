%%%%%%%%%%%%%%%% EUCLID_SS_Gibbs %%%%%%%%%%%%%%%%
clc; clear ;close all;

% Number of nodes per element
NEN = 3; 
% Number of degrees of freedom per node
NDIM = 2;                     
L = 1;
h = 1;
% For NeoHookean use 'NeoHookean_J2' in fem_material.
fem_material = 'NeoHookean_J2';  

loadsteps  =  [10,20,30,40,50,60];
balance = 100;          %Hyperparameter lambda_r used in line 131 while forming LHS and RHS 

% it was giving an error when we consider noise beyond around 0.00015 = 1.5 x 0.0001 on using SS without RVM.
noise = (0.000);       % Standard deviation value i.e. 0.0001, and 0.001 
noiselevel = 'no';
% percent = 1;         % Percentage of noise 

filename  =  'FEM_data/';
filename  =  [filename,fem_material,'/'];
for ii = 1:length(loadsteps)

    filenameadd  =  [filename,num2str(loadsteps(ii)),'/'];
    df = readmatrix([filenameadd,'coordinates_and_disp.csv']);

    % Total Number of nodes
    NN = length(df);         
    nodes = df(:,1);

    bcs_nodes = [nodes,zeros(NN,2)];
    % Boundary condition matrix with dof 1 as negative x-direction,
    % dof 2 as positive x-direction, dof 3 as negative y-direction,
    % dof 4 as positive y-direction

    % Displacements of each nodes
    if noiselevel == "no"
        u_node = df(:,4:5);  
    elseif noiselevel == "low"
        u_node = df(:,6:7);
    elseif noiselevel == "high"
        u_node = df(:,8:9);
    end

    % Coordinates of each nodes
    x_node_x = df(:,2);
    noise_x = noise*randn(size(u_node,1),1);
    idx = x_node_x==0;
    bcs_nodes(idx,2)=1;
    noise_x(idx) = 0;
    idx = x_node_x==L;
    bcs_nodes(idx,2)=2;
    noise_x(idx) = 0;

    x_node_y = df(:,3);
    noise_y = noise*randn(size(u_node,1),1);
    idx = x_node_y==0;
    bcs_nodes(idx,3)=3;
    noise_y(idx) = 0;
    idx = x_node_y==h;
    bcs_nodes(idx,3)=4;
    noise_y(idx) = 0;

    noise_mat = [noise_x,noise_y];
    u_node = u_node + noise_mat;

    u_nodes = [nodes,u_node];
    x_nodes = [nodes,df(:,2:3)];

    % Connectivity Matrix
    df = readmatrix([filenameadd,'connectivity.csv']);
    connectivity = df(:,2:4);      

    % Total number of elements
     NE = length(connectivity);     
    [gradNa] = formgradientofShapefunction(NE,connectivity,x_nodes);

    % Displacement matrix according to connectivity matrix
    [u] = displacements(u_nodes,connectivity);  

    % Coordinate matrix according to connectivity matrix
    [x] = locations(x_nodes,connectivity);      
    [qpweights] = gaussweights(x,NE);
%     headers = {'gradNa_node1_x' ,'gradNa_node1_y','gradNa_node2_x','gradNa_node2_y','gradNa_node3_x','gradNa_node3_y','qpweights'};
%     gradNa_and_qpweights = [gradNa,qpweights];
%     csvwrite_with_headers('integrator.csv',gradNa_and_qpweights,headers)
    [F] = calculating_F(NE,NDIM,NEN,gradNa,u);
    [J] = computeJacobian(NE,NDIM,F);
    [C] = computeCauchyGreenTensor(NE,NDIM,F);
    [I1,I2,I3] = computeStrainInvariants(NE,NDIM,C);
    [dI1dF,dI2dF,dI3dF] = computeStrainInvariantDerivatives(F,J);

    % For 43 features
    NF = 43;
    % Polynomial terms degree.
    N  =  7;
    % Volumetric terms degree.
    M  =  7;
    % dQ/dI(i)
    [Q] = computeFeatures(J,I1,I3,N,M,NF);
    [dQdI1,dQdI2,dQdI3] = differentiateFeaturesWithInvariants(NF,NE,I1,I3); 

    
%     % For 26 features
%     NF = 24;
%     % Polynomial terms degree.
%     N  =  5;
%     % Volumetric terms degree.
%     M  =  1;
%     % dQ/dI(i)
%     [Q] = computeFeatures(J,I1,I3,N,M,NF);
%     [dQdI1,dQdI2,dQdI3]=differentiateFeaturesWithInvariants2(NF,NE,I1,I3);

    [matrix_global] = computeglobaldofmatrix(NE,NEN,connectivity,NDIM);    

    % Degrees of freedom matrix
    [LHS_weak] = computeweakLHS(NN,NDIM,Q,NE,NEN,gradNa,dQdI1,dQdI3,dI1dF,dI3dF,qpweights,matrix_global);
    
    % Reactions at boundaries
    df = readmatrix([filenameadd,'reactions.csv']);
    forces = df(:,1);

    [LHS_free,LHS_fix,RHS_free,RHS_fix] = extendLHSRHS_ssnew(NN,bcs_nodes,LHS_weak,forces);   
%     idx=randi([1,size(LHS_free,1)],1,100);
%     LHS_free = LHS_free(idx,:);
%     RHS_free = RHS_free(idx,:);

    if ii == 1
    LHS_fr = zeros(1,NF);
    LHS_fx = zeros(1,NF);
    RHS_fr = zeros(1,1);
    RHS_fx = zeros(1,1);
    end

    LHS_fr = [LHS_fr ; LHS_free];
    LHS_fx = [LHS_fx ; LHS_fix];
    RHS_fr = [RHS_fr ; RHS_free];
    RHS_fx = [RHS_fx ; RHS_fix];

end

LHS_fx = balance* LHS_fx;
RHS_fx = balance* RHS_fx;
LHS = [LHS_fr ; LHS_fx];
RHS = [RHS_fr ; RHS_fx];

% theta  =  linsolve(LHS,RHS);
scalD = (std(LHS))';
addpath('github_repo') 
addpath('SS_Gibbs')
addpath('SS_Gibbs/myBVS_functions')
addpath('SS_Gibbs/SB2_Release_200')
 
tic
[wmu_RVM, sigma_RVM, model_RVM] = run_my_RVM(RHS, LHS, 10000);
wmu_RVM = wmu_RVM./scalD;
trRVM = toc;
% RVM with gamma diagnostic > 0.9
wmu_RVMg   = zeros(size(wmu_RVM));
goodidx    = model_RVM.parameters.Relevant(model_RVM.diagnostic.Gamma>0.9);
wmu_RVMg(goodidx) = wmu_RVM(goodidx);
theta_RVM = wmu_RVMg;

idx = find(theta_RVM>0 & theta_RVM<50);
% model_RVM.diagnostic.Gamma = model_RVM.diagnostic.Gamma(idx)
LHS_RVM = LHS(:,idx);
theta_RVM = theta_RVM(idx);
scalD_RVM = (std(LHS_RVM))';


% All the hyperparameters of different distributions are set according to Joshi paper
% Using DISSP independent slab
nSamples = 5000;
nBurnin  = 1000;
nchains  = 4;
disp('Running Independent Spike and Slab...');
tic
out_disspI = BVS_Walli_I(LHS_RVM, RHS, nSamples, nBurnin, nchains, false);
wmu_DISSPi = zeros(size(theta_RVM));
wmu_DISSPi(out_disspI.modelIdx) = out_disspI.Wsel;
wmu_DISSPi = wmu_DISSPi./scalD_RVM;
GRstatsi   = out_disspI.Rw;
trMCMC = toc;
disp('Independent Spike and Slab done');
theta_disspi = wmu_DISSPi;
theta_DISSPi = zeros(NF,1);
theta_DISSPi(idx) = theta_disspi;

% disp('Running Independent Spike and Slab...');
% out_disspI = BVS_Walli_I(LHS, RHS, nSamples, nBurnin, nchains, false);
% wmu_DISSPi = zeros(43,1);
% wmu_DISSPi(out_disspI.modelIdx) = out_disspI.Wsel;
% wmu_DISSPi = wmu_DISSPi./scalD;
% GRstatsi   = out_disspI.Rw;
% disp('Independent Spike and Slab done');
% theta_disspi = wmu_DISSPi;
% theta_DISSPi = zeros(NF,1);
% theta_DISSPi(idx) = theta_disspi;

% % Use independent spike and slab VB John Ormerod
% disp('Running independent VB Spike and Slab...');
% initz = zeros(size(theta_RVM));
% initz(model_RVM.parameters.Relevant) = model_RVM.diagnostic.Gamma;
% initz = initz(idx);
% tol = 1e-6;
% out_disspI_vb = BVS_disspI_vb_John(LHS_RVM, RHS, initz, tol, false);
% wmu_DISSPi_vb = zeros(size(theta_RVM));
% wmu_DISSPi_vb(out_disspI_vb.modelIdx) = out_disspI_vb.Wsel;
% wmu_DISSPi_vb = wmu_DISSPi_vb./scalD_RVM;
% disp('independent VB Spike and Slab done');
% theta_vb = wmu_DISSPi_vb;
% theta_VB = zeros(NF,1);
% theta_VB(idx) = theta_vb;
% theta_VB = max(theta_VB,0);
% theta_VB = min(theta_VB,50);
% % posidx = any(theta_VB<0 & theta_VB>50);
% % theta_VB(posidx) = 0;
tic
v=1;
while v==1
% Use independent spike and slab VB John Ormerod
disp('Running independent VB Spike and Slab...');
initz = zeros(size(theta_RVM));
initz(model_RVM.parameters.Relevant) = model_RVM.diagnostic.Gamma;
initz = initz(idx);
tol = 1e-6;
out_disspI_vb = BVS_disspI_vb_John(LHS_RVM, RHS, initz, tol, false);
wmu_DISSPi_vb = zeros(size(theta_RVM));
wmu_DISSPi_vb(out_disspI_vb.modelIdx) = out_disspI_vb.Wsel;
wmu_DISSPi_vb = wmu_DISSPi_vb./scalD_RVM;
% wmu_DISSPi_vb = wmu_DISSPi_vb./scalD;
% theta_VB_cov = out_disspI_vb.Wcov;

% theta_DISSPi_vb = out_disspI_vb.theta_mean;
% theta_DISSPi_vb = theta_DISSPi_vb./(ones(size(theta_DISSPi_vb,1),1)*scalD(out_disspI_vb.modelIdx)');
disp('independent VB Spike and Slab done');
if any(wmu_DISSPi_vb<0)
    v = 1;
    idx2 = find(wmu_DISSPi_vb>0.0);
    idx = idx(idx2);
    theta_RVM = theta_RVM(idx2);
    LHS_RVM = LHS(:,idx);
    scalD_RVM = (std(LHS_RVM))';
else 
    v = 0;
end
end
trVB = toc;
theta_VB = zeros(NF,1);
theta_VB(idx) = wmu_DISSPi_vb;

% If NF = 26 is considered then change the index number 36 to 21 in theta_truth vector.

theta_truth = zeros(NF,1);
if fem_material=="NeoHookean_J2"
    theta_truth(2,1)=0.5;
    theta_truth(36,1)=1.5;
elseif fem_material=="Isihara"
    theta_truth(2,1)=0.5;
    theta_truth(1,1)=1.0;
    theta_truth(5,1)=1.0;
    theta_truth(36,1)=1.5;
elseif fem_material=="HainesWilson"
    theta_truth(2,1)=0.5;
    theta_truth(1,1)=1.0;
    theta_truth(4,1)=0.7;
    theta_truth(9,1)=0.2;
    theta_truth(36,1)=1.5;
end
figure;
% set(gcf, 'Units', 'pixels', 'Position', [10, 150, 1500, 300]);
% 
% XM = linspace(1,43,43)-0.25;
% XV = linspace(1,43,43)+0.25;
% stem(theta_truth,'filled',LineWidth=2.5)
% hold on
% stem(XM,theta_DISSPi,':rdiamond','filled',LineWidth=2.5)
% hold on 
% stem(XV,theta_VB,'--','filled',LineWidth=2.5)
% xlabel("Feature number",FontSize=35)
% ylabel("Coefficient,\theta_i",FontSize=35)
% title('Feature parameter value vs Feature number',FontSize=45)
% set(gca,'XTick',(1:1:NF),'FontSize',22)
% legend({'\theta_{truth}','\theta_{MCMC}','\theta_{VB}'},'Location','northeast','FontSize',20)
% export_fig('IH_theta_no.pdf', '-pdf');


set(gcf, 'Units', 'pixels', 'Position', [10, 150, 1200, 600]);
lambda = linspace(0,1,100);
W_truth = zeros(NF,1);
W_MCMC = zeros(NF,1);
W_VB = zeros(NF,1);
for i =1:100
    F_g = [1+lambda(i),0,0;0,1,0;0,0,1];
    C_g = F_g'*F_g;
    I1_g= trace(C_g);
    I2_g= (0.5).*(trace(C_g)^2-trace(C_g^2));
    I3_g= det(C_g);
    J_g = det(F_g);
    I1_bar_g = J_g^(-2/3).*I1_g;
    I2_bar_g = J_g^(-4/3).*I2_g;
    Q_g = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF);
    W_truth(i) = Q_g'*theta_truth;
    W_MCMC(i) = Q_g'*theta_DISSPi;
    W_VB(i) = Q_g'*theta_VB;
end
subplot(2,3,1)
plot(lambda,W_truth,LineWidth=2)
hold on
plot(lambda,W_MCMC,'.',LineWidth=2)
hold on
plot(lambda,W_VB,'--',LineWidth=2)
xlabel("\lambda",FontSize=15)
set(gca,'FontSize',18)
ylabel("strain energy, W(\lambda)",FontSize=15)
title('Uniaxial tension',Fontsize=20)
legend({'W_{truth}','W_{MCMC}','W_{VB}'},'Location','northwest','FontSize',10)

lambda = linspace(0,1,100);
W_truth = zeros(NF,1);
W_MCMC = zeros(NF,1);
W_VB = zeros(NF,1);
for i =1:100
    F_g = [1/(1+lambda(i)),0,0;0,1,0;0,0,1];
    C_g = F_g'*F_g;
    I1_g= trace(C_g);
    I2_g= (0.5).*(trace(C_g)^2-trace(C_g^2));
    I3_g= det(C_g);
    J_g = det(F_g);
    I1_bar_g = J_g^(-2/3).*I1_g;
    I2_bar_g = J_g^(-4/3).*I2_g;
    Q_g = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF);
    W_truth(i) = Q_g'*theta_truth;
    W_MCMC(i) = Q_g'*theta_DISSPi;
    W_VB(i) = Q_g'*theta_VB;
end
subplot(2,3,2)
plot(lambda,W_truth,LineWidth=2)
hold on
plot(lambda,W_MCMC,'.',LineWidth=2)
hold on
plot(lambda,W_VB,'--',LineWidth=2)
xlabel("\lambda",FontSize=15)
set(gca,'FontSize',18)
ylabel("strain energy, W(\lambda)",FontSize=15)
title('Uniaxial compression',Fontsize=20)
legend({'W_{truth}','W_{MCMC}','W_{VB}'},'Location','northwest','FontSize',10)

lambda = linspace(0,1,100);
W_truth = zeros(NF,1);
W_MCMC = zeros(NF,1);
W_VB = zeros(NF,1);
for i =1:100
    F_g = [1+lambda(i),0,0;0,1+lambda(i),0;0,0,1];
    C_g = F_g'*F_g;
    I1_g= trace(C_g);
    I2_g= (0.5).*(trace(C_g)^2-trace(C_g^2));
    I3_g= det(C_g);
    J_g = det(F_g);
    I1_bar_g = J_g^(-2/3).*I1_g;
    I2_bar_g = J_g^(-4/3).*I2_g;
    Q_g = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF);
    W_truth(i) = Q_g'*theta_truth;
    W_MCMC(i) = Q_g'*theta_DISSPi;
    W_VB(i) = Q_g'*theta_VB;
end
subplot(2,3,3)
plot(lambda,W_truth,LineWidth=2)
hold on
plot(lambda,W_MCMC,'.',LineWidth=2)
hold on
plot(lambda,W_VB,'--',LineWidth=2)
xlabel("\lambda",FontSize=15)
set(gca,'FontSize',18)
ylabel("strain energy, W(\lambda)",FontSize=15)
title('Biaxial tension',Fontsize=20)
legend({'W_{truth}','W_{MCMC}','W_{VB}'},'Location','northwest','FontSize',10)

lambda = linspace(0,1,100);
W_truth = zeros(NF,1);
W_MCMC = zeros(NF,1);
W_VB = zeros(NF,1);
for i =1:100
    F_g = [1/(1+lambda(i)),0,0;0,1/(1+lambda(i)),0;0,0,1];
    C_g = F_g'*F_g;
    I1_g= trace(C_g);
    I2_g= (0.5).*(trace(C_g)^2-trace(C_g^2));
    I3_g= det(C_g);
    J_g = det(F_g);
    I1_bar_g = J_g^(-2/3).*I1_g;
    I2_bar_g = J_g^(-4/3).*I2_g;
    Q_g = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF);
    W_truth(i) = Q_g'*theta_truth;
    W_MCMC(i) = Q_g'*theta_DISSPi;
    W_VB(i) = Q_g'*theta_VB;
end
subplot(2,3,4)
plot(lambda,W_truth,LineWidth=2)
hold on
plot(lambda,W_MCMC,'.',LineWidth=2)
hold on
plot(lambda,W_VB,'--',LineWidth=2)
xlabel("\lambda",FontSize=15)
set(gca,'FontSize',18)
ylabel("strain energy, W(\lambda)",FontSize=15)
title('Biaxial compression',Fontsize=20)
legend({'W_{truth}','W_{MCMC}','W_{VB}'},'Location','northwest','FontSize',10)

lambda = linspace(0,1,100);
W_truth = zeros(NF,1);
W_MCMC = zeros(NF,1);
for i =1:100
    F_g = [1,lambda(i),0;0,1,0;0,0,1];
    C_g = F_g'*F_g;
    I1_g= trace(C_g);
    I2_g= (0.5).*(trace(C_g)^2-trace(C_g^2));
    I3_g= det(C_g);
    J_g = det(F_g);
    I1_bar_g = J_g^(-2/3).*I1_g;
    I2_bar_g = J_g^(-4/3).*I2_g;
    Q_g = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF);
    W_truth(i) = Q_g'*theta_truth;
    W_MCMC(i) = Q_g'*theta_DISSPi;
    W_VB(i) = Q_g'*theta_VB;
end
subplot(2,3,5)
plot(lambda,W_truth,LineWidth=2)
hold on
plot(lambda,W_MCMC,'.',LineWidth=2)
hold on
plot(lambda,W_VB,'--',LineWidth=2)
xlabel("\lambda",FontSize=15)
set(gca,'FontSize',18)
ylabel("strain energy, W(\lambda)",FontSize=15)
title('Simple shear',Fontsize=20)
legend({'W_{truth}','W_{MCMC}','W_{VB}'},'Location','northwest','FontSize',10)

lambda = linspace(0,1,100);
W_truth = zeros(NF,1);
W_MCMC = zeros(NF,1);
for i =1:100
    F_g = [1+lambda(i),0,0;0,1/(1+lambda(i)),0;0,0,1];
    C_g = F_g'*F_g;
    I1_g= trace(C_g);
    I2_g= (0.5).*(trace(C_g)^2-trace(C_g^2));
    I3_g= det(C_g);
    J_g = det(F_g);
    I1_bar_g = J_g^(-2/3).*I1_g;
    I2_bar_g = J_g^(-4/3).*I2_g;
    Q_g = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF);
    W_truth(i) = Q_g'*theta_truth;
    W_MCMC(i) = Q_g'*theta_DISSPi;
    W_VB(i) = Q_g'*theta_VB;
end
subplot(2,3,6)
plot(lambda,W_truth,LineWidth=2)
hold on
plot(lambda,W_MCMC,'.',LineWidth=2)
hold on
plot(lambda,W_VB,'--',LineWidth=2)
xlabel("\lambda",FontSize=15)
set(gca,'FontSize',18)
ylabel("strain energy, W(\lambda)",FontSize=15)
title('Pure shear',Fontsize=20)
legend({'W_{truth}','W_{MCMC}','W_{VB}'},'Location','northwest','FontSize',10)
export_fig('NH_SE_no.pdf', '-pdf');