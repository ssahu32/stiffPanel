"""
This wingbox is a simplified version of the one of the University of Michigan uCRM-9.
We use a couple of pyTACS load generating methods to model various wing loads under cruise.
The script runs the structural analysis, evaluates the wing mass and von misses failure index
and computes sensitivities with respect to wingbox thicknesses and node xyz locations.
"""
# ==============================================================================
# Standard Python modules
# ==============================================================================
from __future__ import print_function
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pprint import pprint
import numpy as np
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import TACS, functions, constitutive, elements, pyTACS, problems

comm = MPI.COMM_WORLD

# There are 8 plate segments and 4 stiffener segments.
# Since the overall panel is symmetric, they can be set equal to the opposite of one another
# to reduce design variables
def symmetryIndex(xInput):
    totalIndex = np.array([[0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 5, 4], 
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    xOutput = np.zeros(len(totalIndex[0]))

    for i in range(0, len(xInput)):
        for j in range(0, len(xOutput)):
            if i == totalIndex[0, j]:
                xOutput[totalIndex[1, j]] = xInput[i]

    return xOutput


def revSymmetryIndex(xInput):
    totalIndex = np.array([[0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 5, 4], 
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    xOutput = np.zeros(6)

    for i in range(0, len(xInput)):
        for j in range(0, len(totalIndex[0])):
            if i == totalIndex[1, j]:
                xOutput[totalIndex[0, j]] += xInput[i]

    return xOutput

# CAPS group assigments are all jumbled up. This maps them properly
# First 8 indexes represent plate segments perpendicular to stiffeners
# Last 4 indexes represent stiffener segments
def designIndex(xInput):
    plateIndex = np.array([[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7], 
    [16,23,40,47,64,71,88,95,103,111,15,22,39,46,63,70,87,94,102,110,14,21,38,45,62,69,86,93,101,109,13,20,37,44,61,68,85,92,100,108,12,19,36,43,60,67,84,91,99,107,11,18,35,42,59,66,83,90,98,106,10,17,34,41,58,65,82,89,97,105,8,9,32,33,56,57,80,81,96,104]])

    stiffenerIndex = np.array([[8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11],
    [72,73,74,75,76,77,78,79,48,49,50,51,52,53,54,55,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7]])

    totalIndex = np.hstack((plateIndex, stiffenerIndex))
    xOutput = np.zeros(len(totalIndex[0]))

    for i in range(0, len(xInput)):
        for j in range(0, len(xOutput)):
            if i == totalIndex[0, j]:
                xOutput[totalIndex[1, j]] = xInput[i]

    return xOutput


def revDesignIndex(xInput):
    plateIndex = np.array([[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7], 
    [16,23,40,47,64,71,88,95,103,111,15,22,39,46,63,70,87,94,102,110,14,21,38,45,62,69,86,93,101,109,13,20,37,44,61,68,85,92,100,108,12,19,36,43,60,67,84,91,99,107,11,18,35,42,59,66,83,90,98,106,10,17,34,41,58,65,82,89,97,105,8,9,32,33,56,57,80,81,96,104]])

    stiffenerIndex = np.array([[8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11],
    [72,73,74,75,76,77,78,79,48,49,50,51,52,53,54,55,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7]])

    totalIndex = np.hstack((plateIndex, stiffenerIndex))
    xOutput = np.zeros( totalIndex[0, len(totalIndex[0]) - 1] + 1)

    for i in range(0, len(xInput)):
        for j in range(0, len(totalIndex[0])):
            if i == totalIndex[1, j]:
                xOutput[totalIndex[0, j]] += xInput[i]
                
    return xOutput

def revdesignVarIndex(xInput):
    designVarIndex = np.array([111,16,110,102,94,87,70,63,46,39,22,103,15,109,101,93,86,69,62,45,38,21,95,14,108,100,92,85,68,61,44,37,20,88,13,107,99,91,84,67,60,43,36,19,71,12,106,98,90,83,66,59,42,35,18,64,11,105,97,89,82,65,58,41,34,17,47,10,104,96,80,81,56,57,32,33,8,40,9,23,79,53,29,5,76,52,28,4,75,51,27,55,3,74,50,26,2,73,49,25,1,72,31,48,24,0,7,78,54,30,6,77])
    xOutput = np.zeros(len(xInput))
    for i in range(0, len(xInput)):
        xOutput[designVarIndex[i]] = xInput[i]
    return xOutput

# Instantiate FEASolver
structOptions = {
    'printtiming':True,
}

bdfFile = os.path.join(os.path.dirname(__file__), 'nastran_CAPS3_coarse.dat')
FEASolver = pyTACS(bdfFile, options=structOptions, comm=comm)

# Material properties
rho = 2780.0        # density kg/m^3
E = 73.1e9          # Young's modulus (Pa)
nu = 0.33           # Poisson's ratio
kcorr = 5.0/6.0     # shear correction factor
ys = 324.0e6        # yield stress

# Shell thickness
tInputArray1 = 0.1*np.ones(6)
# tInputArray1 = np.linspace(1.0, 10.0,num=6)
# tInputArray1 = np.array([1.46e-2, 4.41e-3, 1.61e-3, 7.38e-3, 1.22e-2, 4.95e-3]) # Optimized Result
tInputArray2 = symmetryIndex(tInputArray1)
tOutputArray3= designIndex(tInputArray2)
# tOutputArray3 = np.linspace(1, 112,num=112)
print('tOutputArray3:   ', tOutputArray3)


# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # print('dvNum:          ', dvNum)
    # print('compID:         ', compID)
    # print('compDescript:   ', compDescript)
    # print('elemDescripts:  ', elemDescripts)
    # print('globalDVs:      ', globalDVs)
    # print('kwargs:         ', kwargs)
    elemIndex = kwargs['propID'] - 1
    # t = tOutputArray[compID]
    t = tOutputArray3[elemIndex]

    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # prop = constitutive.MaterialProperties(rho=rho, specific_heat=specific_heat,
    #                                        E=E, nu=nu, ys=ys, cte=cte, kappa=kappa)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

    refAxis = np.array([1.0, 0.0, 0.0])

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    # transform = elements.ShellRefAxisTransform(refAxis)
    transform = None
    for elemDescript in elemDescripts:
        if elemDescript in ['CQUAD4', 'CQUADR']:
            # elem = elements.Quad4ThermalShell(transform, con)
            elem = elements.Quad4Shell(transform, con)
        elif elemDescript in ['CTRIA3', 'CTRIAR']:
            elem = elements.Tri3Shell(transform, con)
        else:
            print("Uh oh, '%s' not recognized" % (elemDescript))
        elemList.append(elem)

    # Add scale for thickness dv
    scale = [100.0]
    return elemList, scale

# Set up elements and TACS assembler
FEASolver.initialize(elemCallBack)
assembler = FEASolver.assembler

# Create the KS Function
ksWeight = 100.0
funcs = [functions.KSFailure(assembler, ksWeight=ksWeight),
         functions.StructuralMass(assembler),
         functions.Compliance(assembler),
         functions.KSDisplacement(assembler, ksWeight=1, direction=[0.0, 0.0, 1.0])]
# funcs = [functions.KSFailure(assembler, ksWeight=ksWeight),
#          functions.StructuralMass(assembler),
#          functions.AverageTemperature(assembler),
#          functions.KSTemperature(assembler, ksWeight=ksWeight)]
# funcs = [functions.Compliance(assembler)]

# Get the design variable values
x = assembler.createDesignVec()
x_array = x.getArray()
assembler.getDesignVars(x)
if comm.rank == 0:
    print('x_DesignVars:      ', x_array)
    print('len(x_DesignVars): ', len(x_array))

# Get the node locations
X = assembler.createNodeVec()
assembler.getNodes(X)
assembler.setNodes(X)

# Create the forces
forces = assembler.createVec()
force_array = forces.getArray()
force_array[2::6] -= 1 # uniform load in z direction
# assembler.applyBCs(forces)
assembler.setBCs(forces)

# Set up and solve the analysis problem
res = assembler.createVec()
ans = assembler.createVec()
u = assembler.createVec()
mat = assembler.createSchurMat()
pc = TACS.Pc(mat)
subspace = 100
restarts = 2
gmres = TACS.KSM(mat, pc, subspace, restarts)

# Assemble the Jacobian and factor
alpha = 1.0
beta = 0.0
gamma = 0.0
assembler.zeroVariables()
assembler.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

# Solve the linear system
gmres.solve(forces, ans)
assembler.setVariables(ans)

# Evaluate the function
fvals1 = assembler.evalFunctions(funcs)

if comm.rank == 0:
    print('fvals1:      ', fvals1)

# Assemble the transpose of the Jacobian matrix
assembler.assembleJacobian(alpha, beta, gamma, res, mat, TACS.TRANSPOSE)
pc.factor()

# Solve for the adjoint variables
adjoint = assembler.createVec()

dfdx = []

for func in funcs:
    res.zeroEntries()
    assembler.addSVSens([func], [res])
    gmres.solve(res, adjoint)

    # Compute the total derivative w.r.t. material design variables
    fdv_sens = assembler.createDesignVec()
    assembler.addDVSens([func], [fdv_sens])
    assembler.addAdjointResProducts([adjoint], [fdv_sens], -1.0)

    # Finalize sensitivity arrays across all procs
    fdv_sens.beginSetValues()
    fdv_sens.endSetValues()

    dfdx.append(fdv_sens)

# Set the complex step
xpert = assembler.createDesignVec()
xpert.setRand()

xnew = assembler.createDesignVec()
xnew.copyValues(x)
if TACS.dtype is complex:
    dh = 1e-30
    xnew.axpy(dh*1j, xpert)
else:
    dh = 1e-6
    xnew.axpy(dh, xpert)

# Set the design variables
assembler.setDesignVars(xnew)

# Compute the perturbed solution
assembler.zeroVariables()
assembler.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()
gmres.solve(forces, u)
assembler.setVariables(u)

# Evaluate the function for perturbed solution
fvals2 = assembler.evalFunctions(funcs)

for i in range(len(funcs)):
    if TACS.dtype is complex:
        fd = fvals2[i].imag/dh
    else:
        fd = (fvals2[i] - fvals1[i])/dh

    result = xpert.dot(dfdx[i])
    if comm.rank == 0:
        print('FD:      ', fd)
        print('Adjoint: ', result)
        print('Rel err: ', (result - fd)/result)


dfdxnp1 = dfdx[3].getArray()
dfdxnp2 = revdesignVarIndex(dfdxnp1)
dfdxnp3 = revDesignIndex(dfdxnp2)
dfdxnp4 = revSymmetryIndex(dfdxnp3)
print('dfdxnp4:     ', dfdxnp4)

# Adjoint Testing

# # dfdxnp = np.zeros(112)
# dfdxnp1 = dfdx[2].getArray()
# dfdxnp1 = revdesignVarIndex(dfdxnp1)
# print('dfdxnp1:  ', dfdxnp1)
# dfdxnp2 = revDesignIndex(dfdxnp1)
# print('dfdxnp2:  ', dfdxnp2)
# dfdxnp3 = revSymmetryIndex(dfdxnp2)
# print('dfdxnp3:  ', dfdxnp3)

# # xpertnp1 = np.ones(112)
# xpertnp1 = xpert.getArray()
# xpertnp1 = revdesignVarIndex(xpertnp1)
# print('xpertnp1: ', xpertnp1)
# xpertnp2 = revDesignIndex(xpertnp1)
# print('xpertnp2: ', xpertnp2)

# fdnp = np.dot(xpertnp2, dfdxnp2)
# print('fdnp: ', fdnp)
# fdnpcorrect = np.dot(xpertnp1, dfdxnp1)
# print('fdnpcorrect: ', fdnpcorrect)

# Get the design variable values
x2 = assembler.createDesignVec()
x_array2 = x2.getArray()
assembler.getDesignVars(x2)

print('reordering', assembler.getReordering())

# Output for visualization
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS |
        TACS.OUTPUT_STRESSES |
        TACS.OUTPUT_EXTRAS)
f5 = TACS.ToFH5(assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)
f5.writeToFile('output.f5')