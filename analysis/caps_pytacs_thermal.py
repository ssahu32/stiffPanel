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

# Instantiate FEASolver
structOptions = {
    'printtiming':True,
}

bdfFile = os.path.join(os.path.dirname(__file__), 'nastran_CAPS3_coarse_thermal.dat')
FEASolver = pyTACS(bdfFile, options=structOptions, comm=comm)

# Material properties
rho = 2780.0        # density kg/m^3
E = 73.1e9          # Young's modulus (Pa)
nu = 0.33           # Poisson's ratio
kcorr = 5.0/6.0     # shear correction factor
ys = 324.0e6        # yield stress
specific_heat = 920.096
cte = 24.0e-6
kappa = 230.0

# Shell thickness
t = 0.005            # m
# tarray = np.array([0.01, 0.05])
tMin = 0.002        # m
tMax = 0.05         # m

# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):

    # print('dvNum:          ', dvNum)
    # print('compID:         ', compID)
    # print('compDescript:   ', compDescript)
    # print('elemDescripts:  ', elemDescripts)
    # print('globalDVs:      ', globalDVs)
    # print('kwargs:         ', kwargs)

    # t = tarray[dvNum]
    # Setup (isotropic) property and constitutive objects
    # prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    prop = constitutive.MaterialProperties(rho=rho, specific_heat=specific_heat,
                                           E=E, nu=nu, ys=ys, cte=cte, kappa=kappa)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

    # # Define reference axis for local shell stresses
    # if 'SKIN' in compDescript: # USKIN + LSKIN
    #     sweep = 35.0 / 180.0 * np.pi
    #     refAxis = np.array([np.sin(sweep), np.cos(sweep), 0])
    # else: # RIBS + SPARS + ENGINE_MOUNT
    #     refAxis = np.array([0.0, 0.0, 1.0])

    refAxis = np.array([1.0, 0.0, 0.0])

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    # transform = elements.ShellRefAxisTransform(refAxis)
    transform = None
    for elemDescript in elemDescripts:
        if elemDescript in ['CQUAD4', 'CQUADR']:
            elem = elements.Quad4ThermalShell(transform, con)
            # elem = elements.Quad4Shell(transform, con)
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
         functions.AverageTemperature(assembler),
         functions.KSTemperature(assembler, ksWeight=ksWeight)]
# funcs = [functions.Compliance(assembler)]

# Get the design variable values
x = assembler.createDesignVec()
x_array = x.getArray()
assembler.getDesignVars(x)
if comm.rank == 0:
    print('x_DesignVars:      ', x_array)

# Get the node locations
X = assembler.createNodeVec()
assembler.getNodes(X)
assembler.setNodes(X)

# Create the forces
forces = assembler.createVec()
force_array = forces.getArray()
# force_array[2::7] += 1.0 # uniform load in z direction
force_array[6::7] += 1e-3 # Heat Flux
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

# Output for visualization
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS |
        TACS.OUTPUT_STRESSES |
        TACS.OUTPUT_EXTRAS)
f5 = TACS.ToFH5(assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)
f5.writeToFile('outputThermal.f5')