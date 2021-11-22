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


tacs_comm = MPI.COMM_WORLD



# # rst begin objfunc
# def objfunc(xdict):
#     x = xdict["xvars"]
#     funcs = {}
#     funcs["obj"] = -x[0] * x[1] * x[2]
#     conval = [0] * 2
#     conval[0] = x[0] + 2.0 * x[1] + 2.0 * x[2] - 72.0
#     conval[1] = -x[0] - 2.0 * x[1] - 2.0 * x[2]
#     funcs["con"] = conval
#     fail = False

#     return funcs, fail




# rst begin objfunc
def objfunc(xdict):

    # Instantiate FEASolver
    structOptions = {
        'printtiming':False,
    }

    bdfFile = os.path.join(os.path.dirname(__file__), 'nastran_CAPS3_coarse.dat')
    FEASolver = pyTACS(bdfFile, options=structOptions, comm=tacs_comm)

    # Material properties
    rho = 2780.0        # density kg/m^3
    E = 73.1e9          # Young's modulus (Pa)
    nu = 0.33           # Poisson's ratio
    kcorr = 5.0/6.0     # shear correction factor
    ys = 324.0e6        # yield stress
    # Shell thickness
    t = 0.005            # m

    def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
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
    FEASolver.createTACSAssembler(elemCallBack)
    tacs = FEASolver.assembler

    # Create the KS Function
    ksWeight = 100.0
    # funcs = [functions.KSFailure(tacs, ksWeight=ksWeight)]
    funcs = [functions.StructuralMass(tacs)]
    # funcs = [functions.Compliance(tacs)]

    # Get the design variable values
    x = tacs.createDesignVec()
    x_array = x.getArray()
    tacs.getDesignVars(x)

    # Get the node locations
    X = tacs.createNodeVec()
    tacs.getNodes(X)
    tacs.setNodes(X)

    # Create the forces
    forces = tacs.createVec()
    force_array = forces.getArray() 
    force_array[2::6] += 100.0 # uniform load in z direction
    # force_array[3::7] += 1.0 # Heat Flux
    tacs.applyBCs(forces)
    # tacs.setBCs(forces)

    # Set up and solve the analysis problem
    res = tacs.createVec()
    ans = tacs.createVec()
    u = tacs.createVec()
    mat = tacs.createSchurMat()
    pc = TACS.Pc(mat)
    subspace = 100
    restarts = 2
    gmres = TACS.KSM(mat, pc, subspace, restarts)

    # Assemble the Jacobian and factor
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)
    pc.factor()

    # Solve the linear system
    gmres.solve(forces, ans)
    tacs.setVariables(ans)


    xPO = xdict["xvars"]
    funcs = {}
    funcs["obj"] = -xPO[0] * xPO[1] * xPO[2]
    conval = [0] * 2
    conval[0] = xPO[0] + 2.0 * xPO[1] + 2.0 * xPO[2] - 72.0
    conval[1] = -xPO[0] - 2.0 * xPO[1] - 2.0 * xPO[2]
    funcs["con"] = conval
    fail = False

    return funcs, fail


# rst begin optProb
# Optimization Object
optProb = Optimization("TP037 Constraint Problem", objfunc)

# rst begin addVar
# Design Variables
optProb.addVarGroup("xvars", 3, "c", lower=[0, 0, 0], upper=[42, 42, 42], value=10)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 2, lower=None, upper=0.0)

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