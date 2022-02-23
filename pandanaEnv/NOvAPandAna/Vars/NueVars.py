import numpy as np
import pandas as pd
from pandana.core.var import Var

from Utils.misc import *
from Utils.index import KL
from Vars.Vars import *

def kLongestProng(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png']['len']
    return df.groupby(level=KL).agg(np.max)    
kLongestProng = Var(kLongestProng)

kHitsPerPlane = Var(lambda tables: tables['rec.sel.nuecosrej']['hitsperplane'])

kPtP = Var(lambda tables: tables['rec.sel.nuecosrej']['partptp'])

def kMaxY(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    df = df[['start.y','stop.y']].max(axis=1)
    return df.groupby(level=KL).agg(np.max)
kMaxY = Var(kMaxY)

kSparsenessAsymm = Var(lambda tables: tables['rec.sel.nuecosrej']['sparsenessasymm'])

kCaloE = Var(lambda tables: tables['rec.slc']['calE'])

def kNueCalibrationCorrFunc(det, ismc, run):
  if (det != detector.kFD): return 1.
  if not ismc: return 0.9949
  if run < 20753: return 0.9949/0.9844
  return 1.
kNueCalibrationCorrFunc = np.vectorize(kNueCalibrationCorrFunc, otypes=[np.float32])

def kNueCalibrationCorr(tables):
    hdr_df = tables['rec.hdr'][['det','ismc']]
    hdr_df['run'] = hdr_df.index.get_level_values('run')

    scale = pd.Series(kNueCalibrationCorrFunc(hdr_df['det'], hdr_df['ismc'], hdr_df['run']),
                      index=hdr_df.index)
    return scale
kNueCalibrationCorr = Var(kNueCalibrationCorr)

def kEMEnergy(tables):
    lng_png = kLongestProng(tables)

    shwlid_df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    prim_png_calE = shwlid_df['calE'].groupby(level=KL).first()
      
    cvn_png_df = tables['rec.vtx.elastic.fuzzyk.png.cvnpart']
    cvn_em_pid_df = cvn_png_df[['photonid',  \
                                'pizeroid',  \
                                'electronid']].sum(axis=1)

    cvn_em_calE = shwlid_df['calE'].where((cvn_em_pid_df >= 0.5), 0).groupby(level=KL).agg(np.sum)
        
    #cvn_em_calE[cvn_em_calE == 0] = prim_png_calE
    cvn_em_calE[lng_png >= 500] = prim_png_calE

    cvn_em_calE *= kNueCalibrationCorr(tables)

    return cvn_em_calE
kEMEnergy = Var(kEMEnergy)

def kHadEnergy(tables):
    EMEnergy = kEMEnergy(tables)

    calE = tables['rec.slc']['calE']*kNueCalibrationCorr(tables)
   
    HadEnergy = calE - EMEnergy
    return HadEnergy.where(HadEnergy > 0, 0)
kHadEnergy = Var(kHadEnergy)

def kNueEnergy(tables):
    EMEnergy = kEMEnergy(tables)
    HadEnergy = kHadEnergy(tables)
    isRHC = kRHC(tables)

    p0 =  0.0
    p1 =  1.00756
    p2 =  1.07093
    p3 =  0.0
    p4 =  1.28608e-02
    p5 =  2.27129e-01
    norm = 0.0501206
    if isRHC.agg(np.all):
      p0 = 0.0
      p1 = 0.980479
      p2 = 1.45170
      p3 = 0.0
      p4 = -5.82609e-03
      p5 = -2.27599e-01
      norm = 0.001766
  
    NueEnergy = 1./(1+norm)*(HadEnergy*HadEnergy*p5 + \
                                     EMEnergy*EMEnergy*p4 +   \
                                     EMEnergy*HadEnergy*p3 +  \
                                     HadEnergy*p2 +              \
                                     EMEnergy*p1 + p0)
    return  NueEnergy.where((HadEnergy >= 0) & (EMEnergy >= 0), -5.)
kNueEnergy = Var(kNueEnergy)
