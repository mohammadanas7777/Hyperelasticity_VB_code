function[dI1dF,dI2dF,dI3dF]=computeStrainInvariantDerivatives(F,J)
F11 = F(:,1);
F12 = F(:,2);
F21 = F(:,3);
F22 = F(:,4);
%dI1/dF:
dI1dF = 2.0*F;
%dI2/dF:
dI2dF11 = 2.0*F11 - 2.0*F12.*F21.*F22 + 2.0*F11.*(F22.^2);
dI2dF12 = 2.0*F12 + 2.0*F12.*(F21.^2) - 2.0*F11.*F21.*F22;
dI2dF21 = 2.0*F21 + 2.0*(F12.^2).*F21 - 2.0*F11.*F12.*F22;
dI2dF22 = 2.0*F22 - 2.0*F11.*F12.*F21 + 2.0*(F11.^2).*F22;
dI2dF = [dI2dF11,dI2dF12,dI2dF21,dI2dF22];        
%dI3/dF:
dI3dF11 = 2.0*F22.*J;
dI3dF12 = -2.0*F21.*J;
dI3dF21 = -2.0*F12.*J;
dI3dF22 = 2.0*F11.*J;
dI3dF = [dI3dF11,dI3dF12,dI3dF21,dI3dF22]; 
end