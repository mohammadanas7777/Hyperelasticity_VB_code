import numpy as np
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

alpha = np.arange(1e-10,1,1)
gamma = np.arange(1e-6,1,1)
loadsteps = list(np.arange(10,11,10))
L_x = 1.0
L_y = 1.0
    
for m in range(0,len(loadsteps)):  
    print('.')
    print('.')
    print('.')
    df = pd.read_csv("/Users/mohammad.anas7777gmail.com/Downloads/EUCLID_SS_Gibbs/Plate_without_hole/FEM_data/HainesWilson/"+str(loadsteps[m])+"/coordinates_and_disp.csv") 
    print('loadstep = ',loadsteps[m])
    x_coor = np.array([df["x_coor"]])
    y_coor = np.array([df["y_coor"]])
    u_x_orig = (np.array(df["u_x"]))
    u_y_orig = (np.array(df["u_y"]))
    coor = np.concatenate((x_coor.T, y_coor.T), axis=1)
    NN = len(df)
    idx_x0 = x_coor==0
    idx_xL = x_coor==L_x
    idx_y0 = y_coor==0
    idx_yL = y_coor==L_y
    
    noise_low = 0.0001
    u_x_noise_low = (noise_low * np.array(torch.randn(NN)))
    u_x_noise_low[idx_x0[0]] = 0
    u_x_noise_low[idx_xL[0]] = 0
    u_y_noise_low = (noise_low * np.array(torch.randn(NN)))
    u_y_noise_low[idx_y0[0]] = 0
    u_y_noise_low[idx_yL[0]] = 0
    ux_low = u_x_orig + u_x_noise_low
    uy_low = u_y_orig + u_y_noise_low
    
    noise_high = 0.001
    u_x_noise_high = (noise_high * np.array(torch.randn(NN)))
    u_y_noise_high = (noise_high * np.array(torch.randn(NN)))
    ux_high = u_x_orig + u_x_noise_high
    uy_high = u_y_orig + u_y_noise_high
    
    mse_x_low_prev = 10
    mse_y_low_prev = 10
    mse_x_high_prev = 10
    mse_y_high_prev = 10
    
    for i in range(0,len(alpha)):
        for j in range(0,len(gamma)):
            ux_low_pred = []
            krr_x_low = KernelRidge(alpha = alpha[i],kernel='rbf',gamma=gamma[j])
            krr_x_low.fit(coor,ux_low)
            ux_low_pred=(krr_x_low.predict(coor))
            mse_x_low = mean_squared_error(ux_low,ux_low_pred)
            # u_x_low = u_x_pred
            if mse_x_low < mse_x_low_prev:
                alpha_min_x_low = alpha[i]
                gamma_min_x_low = gamma[j]
                mse_x_low_prev = mse_x_low
                u_x_low = ux_low_pred
    
            uy_low_pred = []
            krr_y_low = KernelRidge(alpha = alpha[i],kernel='rbf',gamma=gamma[j])
            krr_y_low.fit(coor,uy_low)
            uy_low_pred=(krr_y_low.predict(coor))
            mse_y_low = mean_squared_error(uy_low,uy_low_pred)
            # u_y_low = u_y_pred
            if mse_y_low < mse_y_low_prev:
                alpha_min_y = alpha[i]
                gamma_min_y = gamma[j]
                mse_y_low_prev = mse_y_low
                u_y_low = uy_low_pred
                
            ux_high_pred = []
            krr_x_high = KernelRidge(alpha = alpha[i],kernel='rbf',gamma=gamma[j])
            krr_x_high.fit(coor,ux_high)
            ux_high_pred=(krr_x_high.predict(coor))
            mse_x_high = mean_squared_error(ux_high,ux_high_pred)
            # u_x_low = u_x_pred
            if mse_x_high < mse_x_high_prev:
                alpha_min_x_high = alpha[i]
                gamma_min_x_high = gamma[j]
                mse_x_high_prev = mse_x_high
                u_x_high = ux_high_pred
    
            uy_high_pred = []
            krr_y_high = KernelRidge(alpha = alpha[i],kernel='rbf',gamma=gamma[j])
            krr_y_high.fit(coor,uy_high)
            uy_high_pred=(krr_y_high.predict(coor))
            mse_y_high = mean_squared_error(uy_high,uy_high_pred)
            # u_y_low = u_y_pred
            if mse_y_high < mse_y_high_prev:
                alpha_min_y = alpha[i]
                gamma_min_y = gamma[j]
                mse_y_high_prev = mse_y_high
                u_y_high = uy_high_pred
    
    print('saving denoised data')
    Y1 = u_x_noise_high
    Y2 = u_x_high-u_x_orig
    node_no = np.arange(0,NN)
    plt.plot(node_no,Y1)        
    plt.plot(node_no,Y2)
    df2 = pd.DataFrame()      
    df2.insert(0, "Node_no.", np.arange(1,NN+1), True)  
    df2.insert(1, "x_coor", x_coor[0], True)
    df2.insert(2, "y_coor", y_coor[0], True)
    df2.insert(3, "u_x", u_x_orig, True)
    df2.insert(4, "u_y", u_y_orig, True)    
    df2.insert(5, "u_x_low", u_x_low, True)
    df2.insert(6, "u_y_low", u_y_low, True)
    df2.insert(7, "u_x_high", u_x_high, True)
    df2.insert(8, "u_y_high", u_y_high, True)
    df2.columns = ['Node_no.','x_coor','y_coor','u_x','u_y','u_x_low','u_y_low','u_x_high','u_y_high']
    df2.to_csv("/Users/mohammad.anas7777gmail.com/Downloads/EUCLID_SS_Gibbs/Plate_without_hole/FEM_data/HainesWilson/"+str(loadsteps[m])+"/coordinates_and_disp.csv",
                  columns=['Node_no.','x_coor','y_coor','u_x','u_y','u_x_low','u_y_low','u_x_high','u_y_high'],index = False)