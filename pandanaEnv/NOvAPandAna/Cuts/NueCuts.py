import pandas as pd
import numpy as np

from pandana.core import Cut

from Utils.index import KL
from Vars.Vars import *
from Vars.NueVars import *
from Cuts.Cuts import *

#Data Quality
def kNueApplyMask(tables):
    mask = tables['rec.hdr']['dibmask']
    fp = tables['rec.slc']['firstplane']
    lp = tables['rec.slc']['lastplane']
    FirstToFront = pd.Series(calcFirstLivePlane(mask.to_numpy(dtype=np.int32), fp.to_numpy(dtype=np.int32)),
                             index=mask.index)
    LastToFront  = pd.Series(calcFirstLivePlane(mask.to_numpy(dtype=np.int32), lp.to_numpy(dtype=np.int32)),
                             index=mask.index)
    FirstToBack  = pd.Series(calcLastLivePlane(mask.to_numpy(dtype=np.int32), fp.to_numpy(dtype=np.int32)),
                             index=mask.index)
    LastToBack   = pd.Series(calcLastLivePlane(mask.to_numpy(dtype=np.int32), lp.to_numpy(dtype=np.int32)),
                             index=mask.index)
    return (FirstToFront == LastToFront) & (FirstToBack == LastToBack) & \
        ((LastToBack - FirstToFront + 1)/64 >= 4)
kNueApplyMask = Cut(kNueApplyMask)

kNueDQ = (kHitsPerPlane < 8) & kHasVtx & kHasPng

kNueBasicPart = kIsFD & kNueDQ & kVeto & kNueApplyMask

# Presel
kNuePresel = (kNueEnergy > 1) & (kNueEnergy < 4) & \
    (kNHit > 30) & (kNHit < 150) & \
    (kLongestProng > 100) & (kLongestProng < 500)

kNueProngContainment = (kDistAllTop > 63) & (kDistAllBottom > 12) & \
    (kDistAllEast > 12) & (kDistAllWest > 12) & \
    (kDistAllFront > 18) & (kDistAllBack > 18)

kNueBackwardCut = ((kDistAllBack < 200) & (kSparsenessAsymm < -0.1)) | (kDistAllBack >= 200)

kNuePtPCut = (kPtP < 0.58) | ((kPtP >= 0.58) & (kPtP < 0.8) & (kMaxY < 590)) | ((kPtP >= 0.8) & (kMaxY < 350))

kNueCorePart = kNuePresel & kNueProngContainment & kNuePtPCut & kNueBackwardCut

kNueCorePresel = kNueCorePart & kNueBasicPart

# PID Selections
kNueCVNFHC = 0.84
kNueCVNRHC = 0.89

def kNueCVNCut(tables):
    df = kCVNe(tables)
    dfRHC = df[kRHC(tables)==1] >= kNueCVNRHC
    dfFHC = df[kRHC(tables)!=1] >= kNueCVNFHC

    return pd.concat([dfRHC, dfFHC])
kNueCVNCut = Cut(kNueCVNCut)

# Full FD Selection
kNueFD = kNueCVNCut & kNueCorePresel

# ND

def kNueNDFiducial(tables):
    df = tables['rec.vtx.elastic']
    df = (df['vtx.x'] > -100) & \
         (df['vtx.x'] < 160) & \
         (df['vtx.y'] > -160) & \
         (df['vtx.y'] < 100) & \
         (df['vtx.z'] > 150) & \
         (df['vtx.z'] < 900)
    return df.groupby(level=KL).first()
kNueNDFiducial = Cut(kNueNDFiducial)

def kNueNDContain(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    df_trans = df[['start.y','stop.y', 'start.x', 'stop.x']]
    df_long = df[['start.z', 'stop.z']]

    return ((df_trans.min(axis=1) > -170) & (df_trans.max(axis=1) < 170) & \
            (df_long.min(axis=1) > 100) & (df_long.max(axis=1) < 1225)).groupby(level=KL).agg(np.all)
kNueNDContain = Cut(kNueNDContain)

kNueNDFrontPlanes = Cut(lambda tables: tables['rec.sel.contain']['nplanestofront'] > 6)

kNueNDNHits = (kNHit >= 20) & (kNHit <= 200)

kNueNDEnergy = (kNueEnergy < 4.5)

kNueNDProngLength = (kLongestProng > 100) & (kLongestProng < 500)

kNueNDPresel = kNueDQ & kNueNDFiducial & kNueNDContain & kNueNDFrontPlanes & \
               kNueNDNHits & kNueNDEnergy & kNueNDProngLength

kNueNDCVNSsb = kNueNDPresel & kNueCVNCut

