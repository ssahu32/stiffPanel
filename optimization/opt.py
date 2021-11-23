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

comm = MPI.COMM_WORLD

# rst begin objfunc
def objfunc(xdict):

    tInputArray1 = xdict["xvars"]

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
    # Shell thickness
    # t = 0.005            # m
    # tInputArray1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    tInputArray2 = symmetryIndex(tInputArray1)
    # tInputArray = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
    #     0.1, 0.1, 0.1, 0.1])
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
    force_array[2::7] += 1e-4 # Uniform z loading
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

    print('Panel Segment Thicknesses:      ', tInputArray1[0:4])
    print('Stiffener Segment Thicknesses:  ', tInputArray1[4:6])
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


# rst begin optProb
# Optimization Object
optProb = Optimization("Stiffened Panel Optimization", objfunc)

# rst begin addVar
# Design Variables
# optProb.addVarGroup("xvars", 3, "c", lower=[0, 0, 0], upper=[42, 42, 42], value=10)
optProb.addVarGroup("xvars", 6, "c", lower=0.001*np.ones(6), upper=1*np.ones(6), value=0.01)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 1, lower=3e-7, upper=6e-7)
# Default values at x = 0.1 for KSFailure, Struct Mass, and Compliance
# [8.93888317e-03 3.33600000e+02 3.88132085e-03]

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
sol = opt(optProb, sens="FD")

# rst begin check
# Check Solution
print(sol)