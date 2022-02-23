import numpy as np
import pandas as pd
from pandana.core.var import Var

from Vars.Vars import *
from Utils.index import KL

kNusScaleFDFHC = 1.2
kNusScaleFDRHC = 1.18
kNusScaleNDFHC = 1.11
kNusScaleNDRHC = 1.15

def kNusEnergy(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)
  cale = kCaloE(tables).copy()
  cale[(isRHC == 1) & (det == detector.kFD)] *= kNusScaleFDRHC
  cale[(isRHC == 1) & (det == detector.kND)] *= kNusScaleNDRHC
  cale[(isRHC == 0) & (det == detector.kFD)] *= kNusScaleFDFHC
  cale[(isRHC == 0) & (det == detector.kND)] *= kNusScaleNDFHC
  return cale
kNusEnergy = Var(kNusEnergy)

kClosestSlcTime = Var(lambda tables: tables['rec.slc']['closestslicetime'])
kClosestSlcMinDist = Var(lambda tables: tables['rec.slc']['closestslicemindist'])
kClosestSlcMinTop = Var(lambda tables: tables['rec.slc']['closestsliceminfromtop'])
