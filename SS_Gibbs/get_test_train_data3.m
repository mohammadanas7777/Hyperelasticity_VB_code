function DS = get_test_train_data3(modelParam, simParam)
% Generate training and test data

tspan = simParam.tspan;    % Time-span

x0 = [0;0];                   % Initial conditions

if(simParam.inpRnd)
    u       = simParam.u;
    xState  = model_1DOF_randominp2(modelParam, tspan, x0, u);
else
    inpAmp  = simParam.inpAmp;   % Input amplitude
    inpFreq = simParam.inpFreq;   % Input frequency 
    u  = inpAmp*sin(2*pi*inpFreq*tspan);  
    xState  = model_1DOF_sineinp2(modelParam, tspan, x0, inpAmp, inpFreq);
end

% Plot phase diagrams
plot_phaseD = 0;
if(plot_phaseD)
    plot(xState(:,1), xState(:,2),'color',[0.11,0.72,0.54]);
    xlabel('$x_1$', 'interpreter', 'latex');
    ylabel('$x_2$', 'interpreter', 'latex');
end

% Adding noise to measured input
un = u + simParam.noisePercent*std(u)*randn(size(u));

% xn = xState;
xnoiseSD = simParam.noisePercent*std(xState);
xn = xState + randn(size(xState))*diag(xnoiseSD);

% Central Diff to get acceleration from velocity state 
% acctrue = gradient(xState(:,2),1/simParam.fs);   % true acc
acctrue = get_acc(modelParam, u, xState);
yacc = acctrue + simParam.noisePercent*std(acctrue)*randn(size(acctrue));

y = yacc;

% Segregating test and train data
NN = length(y);

N1 = ceil(NN/2);    % Split data into half and half

idxtrain = 1:N1;
idxtest  = N1+1:NN;
ytrain   = y(idxtrain);
ytest    = y(idxtest);

xtrain   = xn(idxtrain,:);
xtest    = xn(idxtest,:);

utrain   = un(idxtrain);
utest    = un(idxtest);

%% Pre-scaling 
prescale = false;
if(prescale)
    % Scale states (the RHS)
    stdXntr   = std(xtrain)/10;
    xtrain    = xtrain*diag(1./stdXntr);
    stdutr    = std(utrain);
    utrain    = utrain/stdutr;     
else
    stdXntr = [1, 1];
    stdutr  = 1;
end

%% Generate training dictionary with symbolic forms
syms x1 x2;
s = x1 + x2;
[P1tr, ss1] =  generate_polynomial_dictionary(xtrain, s^1);
[P2tr, ss2] =  generate_polynomial_dictionary(xtrain, s^2);
[P3tr, ss3] =  generate_polynomial_dictionary(xtrain, s^3);
[P4tr, ss4] =  generate_polynomial_dictionary(xtrain, s^4);
[P5tr, ss5] =  generate_polynomial_dictionary(xtrain, s^5);
[P6tr, ss6] =  generate_polynomial_dictionary(xtrain, s^6);

ss = horzcat(ss1,ss2,ss3,ss4,ss5,ss6);
ss{end+1} = 'sign(x1)';
ss{end+1} = 'sign(x2)';
%%%%%% NEWLY ADDED %%%%%%
ss{end+1} = 'abs(x1)';
ss{end+1} = 'abs(x2)';
ss{end+1} = 'x1*abs(x1)';
ss{end+1} = 'x2*abs(x1)';
ss{end+1} = 'x1*abs(x2)';
ss{end+1} = 'x2*abs(x2)';

ss{end+1} = '-u';

% Assemble dictionary into a matrix
% Dtrain = [P1tr P2tr P3tr P4tr P5tr P6tr sign(xtrain) -utrain];

signx1x2 = sign(xtrain);
absx1x2  = abs(xtrain);
x1_absx1 = xtrain(:,1).*abs(xtrain(:,1));
x2_absx1 = xtrain(:,2).*abs(xtrain(:,1));
x1_absx2 = xtrain(:,1).*abs(xtrain(:,2));
x2_absx2 = xtrain(:,2).*abs(xtrain(:,2));

Dtrain = [P1tr P2tr P3tr P4tr P5tr P6tr signx1x2 absx1x2 ...
          x1_absx1 x2_absx1 x1_absx2 x2_absx2 -utrain];

% Make training data
stdDtr   = std(Dtrain);
Ds       = Dtrain*diag(1./stdDtr);

% True coefficients
w_tr  = zeros(size(Dtrain,2),1);

w_tr(1) = - modelParam.k/modelParam.m * stdXntr(1);
w_tr(2) = - modelParam.c/modelParam.m * stdXntr(2);
% w_tr(5) = - modelParam.c2/modelParam.m * stdXntr(2)^2; % Squared velocity
w_tr(8) = - modelParam.g1/modelParam.m * stdXntr(1)^3; % Duffing cubic displ
w_tr(29)= - modelParam.cs/modelParam.m; % Signum velocity (Coulomb damping)
w_tr(35) = - modelParam.c2/modelParam.m * stdXntr(2)^2; % Quadratic viscous damping
w_tr(end) = -1/modelParam.m*stdutr;             % Input

w_tr = w_tr.*(stdDtr');

fprintf('Norm of residual: %0.2g \n',norm(ytrain - Ds*w_tr));

%% Generate test disctionary and coeffs
xtest     = xtest*diag(1./stdXntr);
utest     = utest/stdutr; 

%% Generate test dictionary with symbolic forms
[P1ts, ~] =  generate_polynomial_dictionary(xtest, s^1);
[P2ts, ~] =  generate_polynomial_dictionary(xtest, s^2);
[P3ts, ~] =  generate_polynomial_dictionary(xtest, s^3);
[P4ts, ~] =  generate_polynomial_dictionary(xtest, s^4);
[P5ts, ~] =  generate_polynomial_dictionary(xtest, s^5);
[P6ts, ~] =  generate_polynomial_dictionary(xtest, s^6);

% Assemble dictionary into a matrix
signx1x2 = sign(xtest);
absx1x2  = abs(xtest);
x1_absx1 = xtest(:,1).*abs(xtest(:,1));
x2_absx1 = xtest(:,2).*abs(xtest(:,1));
x1_absx2 = xtest(:,1).*abs(xtest(:,2));
x2_absx2 = xtest(:,2).*abs(xtest(:,2));

Dtest = [P1ts P2ts P3ts P4ts P5ts P6ts signx1x2 absx1x2 ...
          x1_absx1 x2_absx1 x1_absx2 x2_absx2 -utest];

% wval = w_tr./stdDtr';

% norm(ytest - Dtest*wval)


fprintf('Condition number of Ds: %g \n', cond(Ds))

% scramble = false;
% if(scramble)
%     d = size(Ds,2);
%     order  = randperm(d);
%     Ds     = Ds(:,order);
%     w_tr = w_tr(order);
% end


% Output data structure
DS.ytrain = ytrain;
DS.ytest  = ytest;
DS.Dtrain = Ds;
DS.Dtest  = Dtest;
DS.wtrain = w_tr;
DS.stdDtr = stdDtr';
DS.ss     = ss;
% scalevar.stdu  = stdu;
% scalevar.stdy  = stdy;

end

function yacc = get_acc(modelParam, u, xn)

m    = modelParam.m;     % Mass
k    = modelParam.k;     % Linear stiffness
c    = modelParam.c;     % Linear damping
pend = modelParam.pend;  % Pendulum
cs   = modelParam.cs;    % Signum vel nonlinearity
c2   = modelParam.c2;    % Squared vel nonlinearity
g1   = modelParam.g1;    % Cubic dis nonlinearity

x1   = xn(:,1);
x2   = xn(:,2);

yacc = m \ (u - k*x1 - c*x2 - g1*x1.^3 - cs*sign(x2) - c2*x2.*abs(x2) - pend*sin(x1));

end

%%% FORWARD SIMULATION FOR RANDOMLY EXCITED SYSTEMS %%%
function xx = model_1DOF_randominp2(modelParam, tspan, x0, u)
% This function is used to solve a SDOF system with random excitation
% Uses fourth order Runge-Kutta (ode4) to solve the ODE

tspan = tspan(:);

t0     = tspan(1);
h      = tspan(3) - tspan(2);
tfinal = tspan(end);
uaug   = [u(:); u(end)];

xx = ode4( @(t,x) general_1DOF_randomExt(t, x, modelParam, uaug, h), t0, h, tfinal, x0);

[m,n] = size(xx);
if(m<n)
    xx = xx';
end

end

function xt = general_1DOF_randomExt(t, xprev, modelParam, u, dt)
% Description: Single dof oscillator subjected to external excitation

% Linear terms
k       = modelParam.k;     % Stiffness
c       = modelParam.c;     % Damping
mass    = modelParam.m;     % Mass

% Nonlinear terms
if(~isfield(modelParam,'pend'));    modelParam.pend = 0; end
if(~isfield(modelParam,'cs'));      modelParam.cs = 0;   end
if(~isfield(modelParam,'c2'));      modelParam.c2 = 0;   end
if(~isfield(modelParam,'g1'));      modelParam.g1 = 0;   end

pend    = modelParam.pend;  % Sinusoid dis nonlinearity
cs      = modelParam.cs;    % Signum vel nonlinearity
c2      = modelParam.c2;    % Squared vel nonlinearity
g1      = modelParam.g1;    % Cubic dis nonlinearity (Duffing)


invmass = 1/mass;

dis     = xprev(1);
vel     = xprev(2);

% Interpolate the forcing function
ut      = interpforce(t, dt, u);

% Equations of motion
xt(1)   = xprev(2);
xt(2)   = - invmass*(pend*sin(dis) + cs*sign(vel) + c2*vel*abs(vel) + c*vel ...
              + k*dis + g1*dis^3);
          
xt(2)   = invmass*ut + xt(2);

xt = xt';
end

function ut = interpforce(t,dt,u)
% Keith's function

% Find where we are in the X array.
ct = t/dt;  % Number of 'deltat's into time array

it = floor(ct);     % Index of last stored X value (-1).
index = it + 1;
rt = ct - it;       % Distance into current deltat interval.

% Interpolate to current value of X.
ut = u(index) + rt*(u(index+1) - u(index));

end

%%% FORWARD SIMULATION FOR SINE EXCITED SYSTEMS %%%
function xx = model_1DOF_sineinp2(modelParam, tspan, x0, inpAmp, inpFreq)
% This function is used to solve a SDOF system with sinusoidal excitation
% Uses fourth order Runge-Kutta (ode4) to solve the ODE

tspan = tspan(:);
if(nargin<4)
    inpAmp = 0;
    inpFreq = 0;
end

t0     = tspan(1);
h      = tspan(3) - tspan(2);
tfinal = tspan(end);

xx = ode4( @(t,x) general_1DOF_sineExt(t, x, modelParam, inpAmp, inpFreq), t0, h, tfinal, x0);

[m,n] = size(xx);
if(m<n)
    xx = xx';
end

end

function xt = general_1DOF_sineExt(t, xprev, modelParam, A, finp)
% Description: Single dof oscillator subjected to sinusoidal excitation
%   Parameter structure setup function for a generalized oscillator model, 
%   described by the equation
%  
%   d^2x(t)/dt^2 - mu(1-ep x^2(t)) dx(t)/dt + omega^2 x(t) = Asin(2*pi*finp*t),
% 

% Linear terms
k       = modelParam.k;
c       = modelParam.c;
mass    = modelParam.m;

% Nonlinear terms
if(~isfield(modelParam,'pend'));    modelParam.pend = 0; end
if(~isfield(modelParam,'cs'));      modelParam.cs = 0;   end
if(~isfield(modelParam,'c2'));      modelParam.c2 = 0;   end
if(~isfield(modelParam,'g1'));      modelParam.g1 = 0;   end

pend    = modelParam.pend;  % Sinusoid dis nonlinearity
cs      = modelParam.cs;    % Signum vel nonlinearity
c2      = modelParam.c2;    % Squared vel nonlinearity
g1      = modelParam.g1;    % Cubic dis nonlinearity


invmass = 1/mass;

dis     = xprev(1);
vel     = xprev(2);

ut      = A*sin(2*pi*finp*t);

% Equations of motion
xt(1)   = xprev(2);
xt(2)   = - invmass*(pend*sin(dis) + cs*sign(vel) + c2*vel*abs(vel) + c*vel ...
              + k*dis + g1*dis^3);
          
xt(2)   = invmass*ut + xt(2);

xt = xt';
end

function yout = ode4(F,t0,h,tfinal,y0)
   % ODE4  Classical Runge-Kutta ODE solver.
   %   yout = ODE4(F,t0,h,tfinal,y0) uses the classical
   %   Runge-Kutta method with fixed step size h on the interval
   %      t0 <= t <= tfinal
   %   to solve
   %      dy/dt = F(t,y)
   %   with y(t0) = y0.

   %   Copyright 2014 - 2015 The MathWorks, Inc.
   
      y = y0;
      yout = y;
      for t = t0 : h : tfinal-h
         s1 = F(t,y);
         s2 = F(t+h/2, y+h*s1/2);
         s3 = F(t+h/2, y+h*s2/2);
         s4 = F(t+h, y+h*s3);
         y = y + h*(s1 + 2*s2 + 2*s3 + s4)/6;
         yout = [yout, y]; %#ok<AGROW>
      end
end

%%% Generate dictionaries from symbolic entries  %%%

function [D, sym_list] =  generate_polynomial_dictionary(X, expression)

% Find the symbolic variables in the expression
sv = symvar(char(expression));
for i=1:numel(sv)
    syms(sv{i})
end

% Expand the expression
a = expand(expression);
achar = char(a);

% Separate the individual terms in the expansion
achar = strrep(achar,'+', ',');

sym_list = eval(['[',char(achar), ']' ]);
% cl = coeffs(a);
s    = cell(1, numel(sym_list));
stmp = cell(1, numel(sym_list));
for i = 1:numel(sym_list)
    s{i}    = sym_list(i);%/cl(i);
    stmp{i} = char(s{i});
end
sym_list = stmp;


achar = strrep(achar, '*', '.*');
achar = strrep(achar, '^', '.^');
for k = 1:numel(sv)
    eval(sprintf('%s=X(:,%d);',sv{k},k));
end
D = eval(['[',achar,']']);


end