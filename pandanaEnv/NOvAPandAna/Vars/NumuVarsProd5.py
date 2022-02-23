import numpy as np
import pandas as pd
from pandana.core.var import Var

from Vars.Vars import *
from Vars.numuE_utils import *
from Utils.index import KL

def kCosNumi(tables):
    df = tables['rec.trk.kalman.tracks'][['dir.x','dir.y', 'dir.z']]
    # Primary kalman track only
    df = df.groupby(level=KL).first()
    CosNumi = pd.Series(np.zeros_like(df.shape[0]), index=df.index)

    # Use separate beam dir for each detector
    det = kDetID(tables)
    CosNumi[det == detector.kND] = df.mul(BeamDirND, axis=1).sum(axis=1)
    CosNumi[det == detector.kFD] = df.mul(BeamDirFD, axis=1).sum(axis=1)
    return CosNumi
kCosNumi = Var(kCosNumi)

def kNumuMuEND(tables):
  det = kDetID(tables)
  hdr_df = tables['rec.hdr'][['ismc']]
  runs = pd.Series(hdr_df.index.get_level_values('run'), index=hdr_df.index)
  isRHC = kRHC(tables)

  ntracks = (tables['rec.trk.kalman']['ntracks'] != 0)
  
  trklenact = tables['rec.energy.numu']['ndtrklenact']/100.
  trklencat = tables['rec.energy.numu']['ndtrklencat']/100.
  trkcalactE = tables['rec.energy.numu']['ndtrkcalactE']
  trkcaltranE = tables['rec.energy.numu']['ndtrkcaltranE']

  df = pd.concat([runs, isRHC, trklenact, trklencat, det[det == detector.kND]], axis=1, join='inner')

  muE = pd.Series(kApplySplineProd5(df['run'], detector.kND, df['isRHC'], 'act', df['ndtrklenact']) + \
                  kApplySplineProd5(df['run'], detector.kND, df['isRHC'], 'cat', df['ndtrklencat']),
                  index=df.index)
  muE[(trkcalactE == 0.) & (trkcaltranE == 0.)] = -5.
  return muE.where(ntracks, -5.)
kNumuMuEND = Var(kNumuMuEND)

def kNumuMuEFD(tables):
  det = kDetID(tables)
  hdr_df = tables['rec.hdr'][['ismc']]
  runs = pd.Series(hdr_df.index.get_level_values('run'), index=hdr_df.index)
  isRHC = kRHC(tables)

  ntracks = (tables['rec.trk.kalman']['ntracks'] != 0)

  trklen = tables['rec.trk.kalman.tracks']['len']/100
  trklen = trklen.groupby(level=KL).first()

  df = pd.concat([runs, isRHC, trklen, det[det == detector.kFD]], axis=1, join='inner')

  muE = pd.Series(kApplySplineProd5(df['run'], detector.kFD, df['isRHC'], 'muon', df['len']),
                  index=df.index)

  return muE.where(ntracks, -5.)
kNumuMuEFD = Var(kNumuMuEFD)

def kNumuMuE(tables):
  dfND = kNumuMuEND(tables)
  dfFD = kNumuMuEFD(tables)
  return pd.concat([dfND, dfFD])
kNumuMuE = Var(kNumuMuE)

def kNumuHadE(tables):
  det = kDetID(tables)
  isRHC = kRHC(tables)

  runs = tables['rec.hdr']['run']
  
  hadvisE = tables['rec.energy.numu']['hadtrkE'] + tables['rec.energy.numu']['hadcalE']
  hadvisE.name = 'hadvisE'
  ntracks = (tables['rec.trk.kalman']['ntracks'] > 0)

  periods = pd.Series(GetPeriod(runs, det), index = det.index)
  
  hadE = pd.Series(kApplySplineProd5(runs, det, isRHC, 'had', hadvisE), 
                   index=det.index)
  return hadE.where(ntracks, -5.)
kNumuHadE = Var(kNumuHadE)

kNumuE = kNumuMuE + kNumuHadE
kCCE = kNumuE
