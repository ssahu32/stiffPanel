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
from pyoptsparse import SLSQP, Optimization, ParOpt

comm = MPI.COMM_WORLD

# # There are 8 plate segments and 4 stiffener segments.
# # Since the overall panel is symmetric, they can be set equal to the opposite of one another
# # to reduce design variables
# def symmetryIndex(xInput):
#     totalIndex = np.array([[0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 5, 4], 
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

#     xOutput = np.zeros(len(totalIndex[0]))

#     for i in range(0, len(xInput)):
#         for j in range(0, len(xOutput)):
#             if i == totalIndex[0, j]:
#                 xOutput[totalIndex[1, j]] = xInput[i]

#     return xOutput


# def revSymmetryIndex(xInput):
#     totalIndex = np.array([[0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 5, 4], 
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

#     xOutput = np.zeros(6)

#     for i in range(0, len(xInput)):
#         for j in range(0, len(totalIndex[0])):
#             if i == totalIndex[1, j]:
#                 xOutput[totalIndex[0, j]] += xInput[i]

#     return xOutput

# # CAPS groups are all jumbled up. This maps them properly
# # First 8 indexes represent plate segments perpendicular to stiffeners
# # Last 4 indexes represent stiffener segments
# def designIndex(xInput):
#     plateIndex = np.array([[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7], 
#     [16,23,40,47,64,71,88,95,103,111,15,22,39,46,63,70,87,94,102,110,14,21,38,45,62,69,86,93,101,109,13,20,37,44,61,68,85,92,100,108,12,19,36,43,60,67,84,91,99,107,11,18,35,42,59,66,83,90,98,106,10,17,34,41,58,65,82,89,97,105,8,9,32,33,56,57,80,81,96,104]])

#     stiffenerIndex = np.array([[8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11],
#     [72,73,74,75,76,77,78,79,48,49,50,51,52,53,54,55,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7]])

#     totalIndex = np.hstack((plateIndex, stiffenerIndex))
#     xOutput = np.zeros(len(totalIndex[0]))

#     for i in range(0, len(xInput)):
#         for j in range(0, len(xOutput)):
#             if i == totalIndex[0, j]:
#                 xOutput[totalIndex[1, j]] = xInput[i]

#     return xOutput


# def revDesignIndex(xInput):
#     plateIndex = np.array([[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7], 
#     [16,23,40,47,64,71,88,95,103,111,15,22,39,46,63,70,87,94,102,110,14,21,38,45,62,69,86,93,101,109,13,20,37,44,61,68,85,92,100,108,12,19,36,43,60,67,84,91,99,107,11,18,35,42,59,66,83,90,98,106,10,17,34,41,58,65,82,89,97,105,8,9,32,33,56,57,80,81,96,104]])

#     stiffenerIndex = np.array([[8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11],
#     [72,73,74,75,76,77,78,79,48,49,50,51,52,53,54,55,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7]])

#     totalIndex = np.hstack((plateIndex, stiffenerIndex))
#     xOutput = np.zeros( totalIndex[0, len(totalIndex[0]) - 1] + 1)

#     for i in range(0, len(xInput)):
#         for j in range(0, len(totalIndex[0])):
#             if i == totalIndex[1, j]:
#                 xOutput[totalIndex[0, j]] += xInput[i]
                
#     return xOutput

# def revdesignVarIndex(xInput):
#     designVarIndex = np.array([111,16,110,102,94,87,70,63,46,39,22,103,15,109,101,93,86,69,62,45,38,21,95,14,108,100,92,85,68,61,44,37,20,88,13,107,99,91,84,67,60,43,36,19,71,12,106,98,90,83,66,59,42,35,18,64,11,105,97,89,82,65,58,41,34,17,47,10,104,96,80,81,56,57,32,33,8,40,9,23,79,53,29,5,76,52,28,4,75,51,27,55,3,74,50,26,2,73,49,25,1,72,31,48,24,0,7,78,54,30,6,77])
#     xOutput = np.zeros(len(xInput))
#     for i in range(0, len(xInput)):
#         xOutput[designVarIndex[i]] = xInput[i]
#     return xOutput


#####################################################################################################################################################################

class thermalProblem:
    def __init__(self):
            # Instantiate FEASolver
        structOptions = {
            'printtiming':False,
        }

        bdfFile = os.path.join(os.path.dirname(__file__), 'stiffPanel7.dat')
        self.FEASolver = pyTACS(bdfFile, options=structOptions, comm=comm)

        # Material properties
        rho = 2780.0        # density kg/m^3
        E = 73.1e9          # Young's modulus (Pa)
        nu = 0.33           # Poisson's ratio
        kcorr = 5.0/6.0     # shear correction factor
        ys = 324.0e6        # yield stress
        specific_heat = 920.096
        cte = 24.0e-6
        kappa = 230.0
        
        # tInputArray2 = symmetryIndex(tInputArray1)
        # tOutputArray3= designIndex(tInputArray2)

        self.mass_fixed = 3.34e2
        self.ks_scale = 0.008

        t = 0.1 # Initial value

        def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
            # elemIndex = kwargs['propID'] - 1
            # t = tOutputArray[compID]
            # t = tOutputArray3[elemIndex]

            prop = constitutive.MaterialProperties(rho=rho, specific_heat=specific_heat,
                                                    E=E, nu=nu, ys=ys, cte=cte, kappa=kappa)
            con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

            elemList = []
            transform = None
            for elemDescript in elemDescripts:
                if elemDescript in ['CQUAD4', 'CQUADR']:
                    elem = elements.Quad4ThermalShell(transform, con)
                else:
                    print("Uh oh, '%s' not recognized" % (elemDescript))
                elemList.append(elem)
            scale = [100.0]
            return elemList, scale
        
        # Set up elements and TACS assembler
        self.FEASolver.initialize(elemCallBack)
        self.assembler = self.FEASolver.assembler

        # Create the KS Function
        ksWeight = 100.0
        self.tacsFuncs = [functions.KSFailure(self.assembler, ksWeight=ksWeight),
            functions.StructuralMass(self.assembler),
            functions.AverageTemperature(self.assembler),
            functions.KSTemperature(self.assembler, ksWeight=ksWeight),
            functions.Compliance(self.assembler),
            functions.KSDisplacement(self.assembler, ksWeight=100, direction=[0.0, 0.0, -1.0])]

        self.tacsFuncsScale = np.array([
            1.0/self.ks_scale, 
            1.0/self.mass_fixed, 
            1.0, 1.0, 1.0, 1.0])

        # Create the forces
        self.forces = self.assembler.createVec()
        force_array = self.forces.getArray()
        force_array[6::7] += 1e-3 # Heat flux
        # self.assembler.applyBCs(self.forces)
        self.assembler.setBCs(self.forces)

        # Set up and solve the analysis problem
        self.res = self.assembler.createVec()
        self.ans = self.assembler.createVec()
        self.mat = self.assembler.createSchurMat()
        self.pc = TACS.Pc(self.mat)
        subspace = 100
        restarts = 2
        self.gmres = TACS.KSM(self.mat, self.pc, subspace, restarts)

        return


    def objfunc(self, xdict):

        tacs_comm = self.assembler.getMPIComm()

        xvalues = xdict["xvars"]
        
        # tInputArray2 = symmetryIndex(tInputArray1)
        # tOutputArray3= designIndex(tInputArray2)

        # Get the design variable values
        x = self.assembler.createDesignVec()
        x_array = x.getArray()
        if tacs_comm.rank == 0:       
            x_array[:] = xvalues
        
        self.assembler.setDesignVars(x)

        # Assemble the Jacobian and factor
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.assembler.zeroVariables()
        self.assembler.assembleJacobian(alpha, beta, gamma, self.res, self.mat)
        self.pc.factor()

        # Solve the linear system
        self.gmres.solve(self.forces, self.ans)
        self.assembler.setVariables(self.ans)

        # Evaluate the function
        fvals = self.assembler.evalFunctions(self.tacsFuncs)
        fvals *= self.tacsFuncsScale

        if comm.rank == 0:
            print('\n---------- FUNCTION SOLVE ----------')
            print('Design Variables:  ', xvalues)
            print('KSFailure:         ', fvals[0])
            print('Structural Mass:   ', fvals[1])
            print('Average Temp:      ', fvals[2])
            print('KSTemperature:     ', fvals[3])
        
        # Function Assignment 
        funcs = {}
        funcs["obj"] = fvals[0] # Objective
        conval = fvals[1] # Constraint
        funcs["con"] = conval
        fail = False

        return funcs, fail


    def sens(self, xdict, funcs):

        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        # Assemble the transpose of the Jacobian matrix
        self.assembler.assembleJacobian(alpha, beta, gamma, self.res, self.mat, TACS.TRANSPOSE)
        self.pc.factor()

        # Solve for the adjoint variables
        adjoint = self.assembler.createVec()

        dfdx = []

        for index, tacsFunc in enumerate(self.tacsFuncs):
            self.res.zeroEntries()
            self.assembler.addSVSens([tacsFunc], [self.res])
            self.gmres.solve(self.res, adjoint)

            # Compute the total derivative w.r.t. material design variables
            fdv_sens = self.assembler.createDesignVec()
            self.assembler.addDVSens([tacsFunc], [fdv_sens])
            self.assembler.addAdjointResProducts([adjoint], [fdv_sens], -1.0)

            # Finalize sensitivity arrays across all procs
            fdv_sens.beginSetValues()
            fdv_sens.endSetValues()

            fdv_sens.scale(self.tacsFuncsScale[index])
            dfdx.append(fdv_sens)

        # Gradient Assignment    
        dfdxObj1 = dfdx[0].getArray() # Objective
        # dfdxObj2 = revdesignVarIndex(dfdxObj1)
        # dfdxObj3 = revDesignIndex(dfdxObj2)
        # dfdxObj4 = revSymmetryIndex(dfdxObj3)

        dfdxCon1 = dfdx[1].getArray() # Constraint
        # dfdxCon2 = revdesignVarIndex(dfdxCon1)
        # dfdxCon3 = revDesignIndex(dfdxCon2)
        # dfdxCon4 = revSymmetryIndex(dfdxCon3)

        tacs_comm = self.assembler.getMPIComm()

        dfdxObj1 = tacs_comm.bcast(dfdxObj1, root=0)
        dfdxCon1 = tacs_comm.bcast(dfdxCon1, root=0)

        if tacs_comm.rank == 0:
            print(dfdxObj1)
            print(dfdxCon1)

        if comm.rank == 0:
            print('\n---------- GRADIENT SOLVE ----------')
            print('Objective Gradient:       ', dfdxObj1)
            print('Constraint Gradient:      ', dfdxCon1)
            # print('dfdxObj1:      ', dfdxObj1)
            # print('dfdxObj4:      ', dfdxObj4)
        
        # Objective 
        funcsSens = {}
        funcsSens = {
            "obj": {
                "xvars": [dfdxObj1]
            },
            "con": {
                "xvars": [dfdxCon1]
            },
        }

        fail = False

        return funcsSens, fail



    def outputResult(self, xdict):

        # Output for visualization
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_STRAINS |
                TACS.OUTPUT_STRESSES |
                TACS.OUTPUT_EXTRAS)
        f5 = TACS.ToFH5(self.assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)
        f5.writeToFile('optThermal.f5')



#####################################################################################################################################################################

tp = thermalProblem()


# rst begin optProb
# Optimization Object
optProb = Optimization("Stiffened Panel Optimization", tp.objfunc, sens=tp.sens)

# rst begin addVar
# Design Variables
optProb.addVarGroup("xvars", 12, "c", lower=0.01*np.ones(12), upper=1*np.ones(12), value=0.1)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 1, lower=1.0, upper=1.0) # 3.34e2, upper=3.34e2)
# Default values at x = 0.1 for KSFailure, Struct Mass, Average Temperature, KSTemperature
# [8.82910988e-03 3.33600000e+02 1.57902424e-02 1.63050875e-02]

# rst begin addObj
# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
if comm.rank == 0:
    print(optProb)

# rst begin OPT
# Optimizer
optOptions = {"IPRINT": -1}
opt = SLSQP(options=optOptions)
# opt = ParOpt(options="")


# rst begin solve
# Solve
sol = opt(optProb)

# rst begin check
# Check Solution
# if comm.rank == 0:
print(sol)

tp.outputResult(sol.xStar)