from dolfin import *
import fenics as fe
from fenics import *
import matplotlib.pyplot as plt
from ufl import nabla_grad
from ufl import nabla_div
import numpy as np
import pygmsh
import gmsh
import meshio
import pandas as pd
import os
import sympy as sp

loadsteps = list(np.arange(0,61,10))
# loadsteps = [10]

## Displacement in x direction in steps
u_x_right = (np.array(loadsteps)/2) * 0.01   
epsilon_11 = np.zeros(len(u_x_right))
sigma_11 = np.zeros(len(epsilon_11))
Sig11_anly = np.zeros(len(epsilon_11))


## Displacement in y direction in steps
u_y_top = (np.array(loadsteps)) * 0.01    
epsilon_22 = np.zeros(len(u_y_top))
sigma_22 = np.zeros(len(epsilon_22))
Sig22_anly = np.zeros(len(epsilon_22))

# for j in range(0,len(u_x_right)):
for j in range(0,len(loadsteps)):
    print("loadstep = ",loadsteps[j])
        
    # Check whether the specified path exists or not
    isExist = os.path.exists("../../FEM_data/NeoHookean_J2/"+str(loadsteps[j]))
    
    if not isExist:
    # Create a new directory because it does not exist
       os.makedirs("../../FEM_data/NeoHookean_J2/"+str(loadsteps[j]))
       print("The new directory is created!")
    
    ## Dimensions
    L_x = 1.0
    L_y = 1.0
    
    ## Meshing
    gmsh.initialize()
    
    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_min = 0.0
        geom.characteristic_length_max = 0.03
    
        rectangle = geom.add_rectangle([0, 0, 0], L_x, L_y)
        mesh = geom.generate_mesh()
    
    triangle_cells = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            # Collect the individual meshes
            if triangle_cells is None:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack((triangle_cells, cell.data))
    
    triangle_data = None
    triangle_mesh = meshio.Mesh(points=mesh.points, cells={"triangle": triangle_cells})
    nodes = mesh.points[:,0:2]
    cells = triangle_cells
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh,"triangle", 2, 2)
    editor.init_vertices(len(nodes))
    editor.init_cells(len(cells))
    
    ## Saving connectivity matrix in csv
    df = pd.DataFrame(cells)+1
    df.insert(0, "Element_no.", np.arange(1,len(cells)+1), True)
    df.columns = ['Element_no.','node_1','node_2','node_3']
    df.to_csv("../../FEM_data/NeoHookean_J2/"+str(loadsteps[j])+"/connectivity.csv",
              columns=['Element_no.','node_1','node_2','node_3'],index = False)
    
    # for i,n in enumerate(nodes):
    #      print(i, n)
    
    [editor.add_vertex(i,n) for i,n in enumerate(nodes)]
    [editor.add_cell(i,n) for i,n in enumerate(cells)]
    editor.close()
    
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    Du = TrialFunction(V)
    du = TestFunction(V)
    u  = Function(V)
    
    I = Identity(len(u))
    
    ## Defining Boundaries
    def bottom(x, on_boundary):
        return (on_boundary and fe.near(x[1], 0.0))
    
    def top(x, on_boundary):
        return (on_boundary and fe.near(x[1], L_y))
    
    def left(x, on_boundary):
        return (on_boundary and fe.near(x[0], 0.0))
    
    def right(x, on_boundary):
        return (on_boundary and fe.near(x[0], L_x))
    
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    AutoSubDomain(left).mark(boundary_subdomains, 1)
    AutoSubDomain(right).mark(boundary_subdomains, 2)
    AutoSubDomain(bottom).mark(boundary_subdomains, 3)
    AutoSubDomain(top).mark(boundary_subdomains, 4)
    
    dss = ds(subdomain_data=boundary_subdomains)
    
    ## Dirichlet Boundary conditions
    
    ## For simply support ends
    ## left end restricted to move in x direction but free to move in y direction
    ## right end is given u_x_right displacement in x direction and free to move in y direction
    
    bc_l = DirichletBC(V.sub(0), 0.0, left)
    left_bdry_dofs = np.fromiter(bc_l.get_boundary_values().keys(), dtype = int)
    bc_r = DirichletBC(V.sub(0), u_x_right[j], right)
    right_bdry_dofs = np.fromiter(bc_r.get_boundary_values().keys(), dtype = int)
    bc_b = DirichletBC(V.sub(1), 0.0, bottom)
    bottom_bdry_dofs = np.fromiter(bc_b.get_boundary_values().keys(), dtype = int)
    bc_t = DirichletBC(V.sub(1), u_y_top[j], top)
    top_bdry_dofs = np.fromiter(bc_t.get_boundary_values().keys(), dtype = int)
    
    ## For clamped ends
    ## left end restricted to move in both x direction and y direction
    ## right end is given u_x_right displacement in x direction and restricted to move in y direction
    
    # left_bc = Constant((0.0, 0.0))
    # bc_l = DirichletBC(V, left_bc, left)
    # left_bdry_dofs = np.fromiter(bc_l.get_boundary_values().keys(), dtype = int)
    # right_bc = Constant((u_x_right[i], 0.0))
    # bc_r = DirichletBC(V, right_bc, right)
    # right_bdry_dofs = np.fromiter(bc_r.get_boundary_values().keys(), dtype = int)
    # bottom_bc = Constant((0.0, 0.0))
    # bc_b = DirichletBC(V.sub(1), bottom_bc, bottom)
    # bottom_bdry_dofs = np.fromiter(bc_b.get_boundary_values().keys(), dtype = int)
    # top_bc = Constant((u_y_top[j], 0.0))
    # bc_t = DirichletBC(V.sub(1), top_bc, top)
    # top_bdry_dofs = np.fromiter(bc_t.get_boundary_values().keys(), dtype = int)
    
    bcs = [bc_l,bc_r,bc_b,bc_t]

    # Kinematics
    B = Constant((0, 0))
    T = Constant((0, 0))
    F = I + grad(u)             # Deformation gradient
    J = det(F)
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    I1 = tr(C) + 1  
    I2 = 0.5*((tr(C)+1)**2 - (tr(C*C)+1))
    I3  = J**2
    I1_bar = J**(-2/3)*I1
    I2_bar = J**(-4/3)*I2

    # Stored strain energy density
    W = 0.5*(I1_bar - 3) + 1.5*((J-1)**2)

    # Total potential energy
    f_int = W*dx
    f_ext = dot(B, u)*dx + dot(T, u)*dss
    R = f_int - f_ext
    F = derivative(R, u, du)
    
    # Compute first variation of Pi (directional derivative about u in the direction of du)
    Jacobian = derivative(F, u, Du)
    solve(F == 0, u, bcs, J=Jacobian)
    
    u_x = np.zeros(len(nodes))
    u_y = np.zeros(len(nodes))
    for n in range(0,len(nodes)):
        u_x[n] = u(nodes[n,0],nodes[n,1])[0]
        u_y[n] = u(nodes[n,0],nodes[n,1])[1]
        
    ## Saving coordinates and displacement data in csv
    df = pd.DataFrame(nodes)
    df.insert(0, "Node_no.", np.arange(1,len(nodes)+1), True)
    df.insert(3, "u_x", u_x, True)
    df.insert(4, "u_y", u_y, True)
    df.columns = ['Node_no.','x_coor','y_coor','u_x','u_y']
    df.to_csv("../../FEM_data/NeoHookean_J2/"+str(loadsteps[j])+"/coordinates_and_disp.csv",
              columns=['Node_no.','x_coor','y_coor','u_x','u_y'],index = False)

    f_ext_unknown = assemble(F)
    
    ## Sorting x and y degrees of freedom
    x_dofs = V.sub(0).dofmap().dofs()
    y_dofs = V.sub(1).dofmap().dofs()
    
    ## Sorting the reactions in x and y direction residuals of boundaries
    
    ## x component of residuals of nodes present in at left boundary
    Rxlb = [0 for element in range(len(left_bdry_dofs))]
    k = 0
    for i in left_bdry_dofs:
        Rxlb[k] = f_ext_unknown[i]
        k +=1
        
    ## x component of residuals of nodes present in right boundary
    Rxrb = [0 for element in range(len(right_bdry_dofs))]
    k = 0
    for i in right_bdry_dofs:
        Rxrb[k] = f_ext_unknown[i]
        k +=1
        
    ## y component of residuals of nodes present in at bottom boundary
    Rybb = [0 for element in range(len(bottom_bdry_dofs))]
    k = 0
    for i in bottom_bdry_dofs:
        Rybb[k] = f_ext_unknown[i]
        k +=1
    
    ## y component of residuals of nodes present in top boundary
    Rytb = [0 for element in range(len(top_bdry_dofs))]
    k = 0
    for i in top_bdry_dofs:
        Rytb[k] = f_ext_unknown[i]
        k +=1
    
    ## Sunmming all the +ve and -ve x and y direction reactions present in different boundary   
    print("Sum of x component of residuals of nodes present in left boundary",sum(Rxlb))
    print("Sum of x component of residuals of nodes present in right boundary",sum(Rxrb))
    print("Sum of y component of residuals of nodes present in bottom boundary",sum(Rybb))
    print("Sum of y component of residuals of nodes present in top boundary",sum(Rytb))
    
    reactions = [sum(Rxlb),sum(Rxrb),sum(Rybb),sum(Rytb)]
    
    ## Saving coordinates and displacement data in csv
    df = pd.DataFrame(reactions)
    df.columns = ['reactions']
    df.to_csv("../../FEM_data/NeoHookean_J2/"+str(loadsteps[j])+"/reactions.csv",
              columns=['reactions'],index = False)
    
    ## Plotting the final state
    fe.plot(u, mode="displacement")
    plt.show()
    
    ## Saving the displacement data in VTK file
    file = File("NeoHookean_J2.pvd")
    file << u
    
    ## Calculating stress and strain at right end 
    # sigma_11[j] = sum(Fx_pos)/(len(right_bdry_dofs)-1)
    # epsilon_11[j] = (u_x_right[j]*100)/(len(bottom_bdry_dofs)-1)
    sigma_11[j] = sum(Rxrb)/(L_y+u_y_top[j])
    epsilon_11[j] = u_x_right[j]/L_x
    # sigma_22[j] = sum(Fy_pos)/(len(top_bdry_dofs)-1)
    # epsilon_22[j] = (u_y_top[j]*100)/(len(left_bdry_dofs)-1)
    sigma_22[j] = sum(Rytb)/(L_x+u_x_right[j])
    epsilon_22[j] = u_y_top[j]/L_y
    
    ## Analytical Solution
     
    lambda_x = u_x_right[j] + L_x
    lambda_y = u_y_top[j] + L_y
    
    Fa = [[lambda_x, 0], 
          [0, lambda_y]]
    
    Ja = np.linalg.det(Fa)
    I1a = lambda_x**2 + lambda_y**2 + 1
    I1bara = I1a/Ja**(2/3)
    I2a = 0.5*(I1a**2 - (lambda_x**4+lambda_y**4+1)) 
    I2bara = I2a/Ja**(4/3)
   
    # creating a Variable
    i1bar = sp.Symbol('i1bar')
    i2bar = sp.Symbol('i2bar')
    ja = sp.Symbol('ja')
    
    #Define function
    Wa = 0.5*(i1bar-3) + 1.5*((ja-1)**2)
     
    #Calculating Derivative
    dWdI1bar = Wa.diff(i1bar)
    dWdI2bar = Wa.diff(i2bar)
    dWdJ = Wa.diff(ja)
    
    # Calculating stresses
    Sig11a = (((2/Ja**(5/3))*(dWdI1bar+I1bara*dWdI2bar)*(lambda_x**2))-(2/(3*Ja))*(I1bara*dWdI1bar+2*I2bara*dWdI2bar)-((2/Ja**(7/3))*dWdI2bar*lambda_x**4)+dWdJ) 
    Sig11a = Sig11a.subs([(i1bar,I1bara),(i2bar,I2bara),(ja,Ja)])
    Sig11_anly[j] = Sig11a
    
    Sig22a = (((2/Ja**(5/3))*(dWdI1bar+I1bara*dWdI2bar)*(lambda_y**2))-(2/(3*Ja))*(I1bara*dWdI1bar+2*I2bara*dWdI2bar)-((2/Ja**(7/3))*dWdI2bar*lambda_y**4)+dWdJ) 
    Sig22a = Sig22a.subs([(i1bar,I1bara),(i2bar,I2bara),(ja,Ja)])
    Sig22_anly[j] = Sig22a

percentage_error_P11 = ((sigma_11-Sig11_anly)/Sig11_anly)*100
percentage_error_P22 = ((sigma_22-Sig22_anly)/Sig22_anly)*100

## Sigma_11 vs Epsilon_11
plt.plot(
    epsilon_11,
    sigma_11,
    "o-", label = 'FEM solution',
)
plt.plot(
    epsilon_11,
    Sig11_anly,
    "*--",label = 'Analytical solution',
)

plt.legend(['FEM','Analytical'])

plt.xlabel(r"$\epsilon_{11}(x = L)\ \longrightarrow$",fontsize=13)
plt.ylabel(r"$\overline{\sigma}_{11}(x = L)\ \longrightarrow$",fontsize=13)
plt.title(r"$\overline{\sigma}_{11}(x = L)\ $ vs $\epsilon_{11}(x = L)$",fontsize=15)

# ## Sigma_22 vs Epsilon_22
# plt.plot(
#     epsilon_22,
#     sigma_22,
#     "o-", label = 'FEM solution'
# )
# plt.plot(
#     epsilon_22,
#     Sig22_anly,
#     "*--", label = 'Analytical solution'
# )
# plt.legend(['FEM','Analytical'])

# plt.xlabel(r"$\epsilon_{22}(y = H)\ \longrightarrow$",fontsize=13)
# plt.ylabel(r"$\overline{\sigma}_{22}(y = H)\ \longrightarrow$",fontsize=13)
# plt.title(r"$\overline{\sigma}_{22}(y = H)\ $ vs $\epsilon_{22}(y = H)$",fontsize=15)

# slope,intercept = np.polyfit(epsilon_11,sigma_11,2)
# slope,intercept = np.polyfit(epsilon_11,S11_anly,2)

# ## Sigma_22 vs Epsilon_22
# plt.plot(
#     epsilon_22,
#     sigma_22,
#     "o-",
# )
# plt.grid()

# plt.xlabel(r"$\epsilon_{22}(y = H)\ \longrightarrow$")
# plt.ylabel(r"$\sigma_{22}(y = H)\ \ $ $\longrightarrow$")
# plt.title("Stress vs Strain")

# slope,intercept = np.polyfit(epsilon_22,sigma_22,2)
# E = slope
# print("Young's Modulus = ", E)
