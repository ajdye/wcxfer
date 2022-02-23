import pandas as pd
import numpy as np

from pandana.core import Cut

from Utils.index import KL
from Vars.Vars import *
from Vars.NCVars import *
from Cuts.Cuts import *

# FD 

kNusFDContain = (kDistAllTop > 100) & (kDistAllBottom > 10) & \
    (kDistAllEast > 50) & (kDistAllWest > 50) & \
    (kDistAllFront > 50) & (kDistAllBack > 50)

kNusContPlanes = Cut(lambda tables: tables['rec.slc']['ncontplanes'] > 2)

kNusEventQuality = kHasVtx & kHasPng & \
                  (kHitsPerPlane < 8) & kNusContPlanes

kNusFDPresel = kNueApplyMask & kVeto & kNusEventQuality & kNusFDContain

kNusBackwardCut = ((kDistAllBack < 200) & (kSparsenessAsymm < -0.1)) | (kDistAllBack >= 200)

kNusEnergyCut = (kNusEnergy >= 0.5) & (kNusEnergy <= 20.)

kNusSlcTimeGap = (kClosestSlcTime > -150.) & (kClosestSlcTime < 50.)
kNusSlcDist = (kClosestSlcMinTop < 100) & (kClosestSlcMinDist < 500)
kNusShwPtp = ((kMaxY > 580) & (kPtP > 0.2)) | ((kMaxY > 540) & (kPtP > 0.4))

# Nus Cosrej Cuts use TMVA trained BDT natively
kNusNoPIDFD = (kNusFDPresel & kNusBackwardCut) & (~(kNusSlcTimeGap & kNusSlcDist)) & \
              (~kNusShwPtp) & kNusEnergyCut


# ND 

def kNusNDFiducial(tables):
    df = tables['rec.vtx.elastic'][check]
    return ((df['vtx.x'] > -100) & \
        (df['vtx.x'] < 100) & \
        (df['vtx.y'] > -100) & \
        (df['vtx.y'] < 100) & \
        (df['vtx.z'] > 150) & \
        (df['vtx.z'] < 1000)).groupby(level=KL).first()
kNusNDFiducial = Cut(kNusNDFiducial)

kNusNDContain = (kDistAllTop > 25) & (kDistAllBottom > 25) & \
    (kDistAllEast > 25) & (kDistAllWest > 25) & \
    (kDistAllFront > 25) & (kDistAllBack > 25)

kNusNDPresel = kNusEventQuality & kNusNDFiducial & kNusNDContain
kNusNoPIDND = kNusNDPresel & (kPtP <= 0.8) & kNusEnergyCut
