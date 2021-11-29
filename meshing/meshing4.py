#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:52:03 2021

@author: sengelstad6
"""

from __future__ import print_function
import unittest
import time

import os
import glob
import shutil

import sys

import pyCAPS

####################################################################

filename = "stiffPanel4"
csmFile = os.path.join("./CSM",filename + ".csm")

myProblem = pyCAPS.Problem('myCAPS', capsFile=csmFile, outLevel=0)

geom = myProblem.geometry

egads = myProblem.analysis.create(aim="egadsTessAIM")

tacsAnalysis = myProblem.analysis.create(aim = "tacsAIM",
                                 name = "tacs")

egads.input.Edge_Point_Min = 5
egads.input.Edge_Point_Max = 10
        
egads.input.Mesh_Elements = "Quad"
        
egads.input.Tess_Params = [.25,.01,15]
        
NumberOfNode = egads.output.NumberOfNode
        
# Link the mesh
tacsAnalysis.input["Mesh"].link(egads.output["Surface_Mesh"])
        
# Set analysis type
tacsAnalysis.input.Analysis_Type = "Static"
        
shell  = {"propertyType" : "Shell"}
#need to add the strength here

propertyDict = {}
for i in range(80):
    propertyDict["plate"+str(i+1)] = shell
for j in range(32):
    propertyDict["stiffener"+str(j+1)] = shell
tacsAnalysis.input.Property = propertyDict

# Set constraints
constraint = {"groupName" : "edge",
              "dofConstraint" : 12346}
        
tacsAnalysis.input.Constraint = {"edgeConstraint": constraint}

tacsAnalysis.preAnalysis()

homeDir = os.getcwd()
csmDir = os.path.join(homeDir,"BDF")
bdfFile = os.path.join(tacsAnalysis.analysisDir, tacsAnalysis.input.Proj_Name + '.bdf')
datFile = os.path.join(tacsAnalysis.analysisDir, tacsAnalysis.input.Proj_Name + '.dat')

bdfFile2 = os.path.join(csmDir, filename + '.bdf')
datFile2 = os.path.join(csmDir, filename + '.dat')

shutil.move(bdfFile,bdfFile2)
shutil.move(datFile,datFile2)
