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


# ==============================================================================
# Pyoptsparse modules
# ==============================================================================
from pyoptsparse import SLSQP, Optimization
import time

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

# rst begin objfunc
def objfunc(xdict):

    tInputArray1 = xdict["xvars"]

    # Instantiate FEASolver
    structOptions = {
        'printtiming':False,
    }

    bdfFile = os.path.join(os.path.dirname(__file__), 'nastran_CAPS3_coarse.dat')
    FEASolver = pyTACS(bdfFile, options=structOptions, comm=comm)

    # Material properties
    rho = 2780.0        # density kg/m^3
    E = 73.1e9          # Young's modulus (Pa)
    nu = 0.33           # Poisson's ratio
    kcorr = 5.0/6.0     # shear correction factor
    ys = 324.0e6        # yield stress
    
    tInputArray2 = symmetryIndex(tInputArray1)
    tOutputArray3= designIndex(tInputArray2)

    def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
        elemIndex = kwargs['propID'] - 1
        # t = tOutputArray[compID]
        t = tOutputArray3[elemIndex]

        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

        elemList = []
        transform = None
        for elemDescript in elemDescripts:
            if elemDescript in ['CQUAD4', 'CQUADR']:
                # elem = elements.Quad4ThermalShell(transform, con)
                elem = elements.Quad4Shell(transform, con)
            else:
                print("Uh oh, '%s' not recognized" % (elemDescript))
            elemList.append(elem)
        scale = [100.0]
        return elemList, scale
    
    # Set up elements and TACS assembler
    FEASolver.initialize(elemCallBack)
    assembler = FEASolver.assembler

    # Create the KS Function
    ksWeight = 100.0
    tacsFuncs = [functions.KSFailure(assembler, ksWeight=ksWeight),
         functions.StructuralMass(assembler),
         functions.Compliance(assembler)]
    # funcs = [functions.KSFailure(assembler, ksWeight=ksWeight),
    #      functions.StructuralMass(assembler),
    #      functions.AverageTemperature(assembler),
    #      functions.KSTemperature(assembler, ksWeight=ksWeight)]

    # Get the design variable values
    x = assembler.createDesignVec()
    x_array = x.getArray()
    assembler.getDesignVars(x)
    # if comm.rank == 0:
    # print('x_DesignVars:      ', x_array)

    # Get the node locations
    X = assembler.createNodeVec()
    assembler.getNodes(X)
    assembler.setNodes(X)

    # Create the forces
    forces = assembler.createVec()
    force_array = forces.getArray()
    # force_array[2::7] += 1.0 # uniform load in z direction
    force_array[2::6] += 1 # Uniform z loading
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
    fvals = assembler.evalFunctions(tacsFuncs)

    if comm.rank == 0:
        print('Design Variables:  ', tInputArray1)
        print('KSFailure:         ', fvals[0])
        print('Structural Mass:   ', fvals[1])
        print('Compliance:        ', fvals[2])
    
    # Objective 
    funcs = {}
    funcs["obj"] = fvals[1] # Objective is mass minimization
    conval = fvals[2] # Constraint is Compliance
    funcs["con"] = conval
    fail = False

    return funcs, fail



def sens(xdict, funcs):
    
    tInputArray1 = xdict["xvars"]

    # Instantiate FEASolver
    structOptions = {
        'printtiming':False,
    }

    bdfFile = os.path.join(os.path.dirname(__file__), 'nastran_CAPS3_coarse.dat')
    FEASolver = pyTACS(bdfFile, options=structOptions, comm=comm)

    # Material properties
    rho = 2780.0        # density kg/m^3
    E = 73.1e9          # Young's modulus (Pa)
    nu = 0.33           # Poisson's ratio
    kcorr = 5.0/6.0     # shear correction factor
    ys = 324.0e6        # yield stress

    tInputArray2 = symmetryIndex(tInputArray1)
    tOutputArray3= designIndex(tInputArray2)

    def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
        elemIndex = kwargs['propID'] - 1
        # t = tOutputArray[compID]
        t = tOutputArray3[elemIndex]

        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

        elemList = []
        transform = None
        for elemDescript in elemDescripts:
            if elemDescript in ['CQUAD4', 'CQUADR']:
                # elem = elements.Quad4ThermalShell(transform, con)
                elem = elements.Quad4Shell(transform, con)
            else:
                print("Uh oh, '%s' not recognized" % (elemDescript))
            elemList.append(elem)
        scale = [100.0]
        return elemList, scale
    
    # Set up elements and TACS assembler
    FEASolver.initialize(elemCallBack)
    assembler = FEASolver.assembler

    # Create the KS Function
    ksWeight = 100.0
    tacsFuncs = [functions.KSFailure(assembler, ksWeight=ksWeight),
         functions.StructuralMass(assembler),
         functions.Compliance(assembler)]
    # funcs = [functions.KSFailure(assembler, ksWeight=ksWeight),
    #      functions.StructuralMass(assembler),
    #      functions.AverageTemperature(assembler),
    #      functions.KSTemperature(assembler, ksWeight=ksWeight)]

    # Get the design variable values
    x = assembler.createDesignVec()
    x_array = x.getArray()
    assembler.getDesignVars(x)
    # if comm.rank == 0:
    # print('x_DesignVars:      ', x_array)

    # Get the node locations
    X = assembler.createNodeVec()
    assembler.getNodes(X)
    assembler.setNodes(X)

    # Create the forces
    forces = assembler.createVec()
    force_array = forces.getArray()
    # force_array[2::7] += 1.0 # uniform load in z direction
    force_array[2::6] += 1 # Uniform z loading
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
    fvals = assembler.evalFunctions(tacsFuncs)

    if comm.rank == 0:
        print('Design Variables:  ', tInputArray1)
        print('KSFailure:         ', fvals[0])
        print('Structural Mass:   ', fvals[1])
        print('Compliance:        ', fvals[2])

    # Begin adjoint formulation
    start = time.time()

    # Assemble the transpose of the Jacobian matrix
    assembler.assembleJacobian(alpha, beta, gamma, res, mat, TACS.TRANSPOSE)
    pc.factor()

    # Solve for the adjoint variables
    adjoint = assembler.createVec()

    dfdx = []

    for tacsFunc in tacsFuncs:
        res.zeroEntries()
        assembler.addSVSens([tacsFunc], [res])
        gmres.solve(res, adjoint)

        # Compute the total derivative w.r.t. material design variables
        fdv_sens = assembler.createDesignVec()
        assembler.addDVSens([tacsFunc], [fdv_sens])
        assembler.addAdjointResProducts([adjoint], [fdv_sens], -1.0)

        # Finalize sensitivity arrays across all procs
        fdv_sens.beginSetValues()
        fdv_sens.endSetValues()

        dfdx.append(fdv_sens)

    end = time.time()
    if comm.rank == 0:
        print('Adjoint calculation time: ', end - start)
    start = time.time()
    
    dfdxObj1 = dfdx[1].getArray() # Mass minimization
    dfdxObj2 = revdesignVarIndex(dfdxObj1)
    dfdxObj3 = revDesignIndex(dfdxObj2)
    dfdxObj4 = revSymmetryIndex(dfdxObj3)

    dfdxCon1 = dfdx[2].getArray() # Compliance constraint
    dfdxCon2 = revdesignVarIndex(dfdxCon1)
    dfdxCon3 = revDesignIndex(dfdxCon2)
    dfdxCon4 = revSymmetryIndex(dfdxCon3)

    end = time.time()
    if comm.rank == 0:
        print('Objective Gradient:       ', dfdxObj4)
        print('Constraint Gradient:      ', dfdxCon4)
        print('Adjoint organizing time:  ', end - start)
    
    # Objective 
    funcsSens = {}
    funcsSens = {
        "obj": {
            "xvars": [dfdxObj4]
        },
        "con": {
            "xvars": [dfdxCon4]
        },
    }

    fail = False

    return funcsSens, fail

# rst begin optProb
# Optimization Object
optProb = Optimization("Stiffened Panel Optimization", objfunc)

# rst begin addVar
# Design Variables
# optProb.addVarGroup("xvars", 3, "c", lower=[0, 0, 0], upper=[42, 42, 42], value=10)
optProb.addVarGroup("xvars", 6, "c", lower=0.001*np.ones(6), upper=0.1*np.ones(6), value=0.01)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 1, lower=6.8e-1, upper=6.8e-1)
# Default values at x = 0.001 for KSFailure, Struct Mass, and Compliance
# [1.74024250e-02 3.33600000e+01 6.81517601e-01]

# rst begin addObj
# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
print(optProb)

# rst begin OPT
# Optimizer
optOptions = {"IPRINT": -1}
opt = SLSQP(options=optOptions)

# rst begin solve
# Solve
# sol = opt(optProb, sens="FD", storeHistory='opt_history.hst')
sol = opt(optProb, sens=sens, storeHistory='opt_history.hst')

# rst begin check
# Check Solution
if comm.rank == 0:
    print(sol)