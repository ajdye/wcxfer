from Utils.index import KL,KLN
from Utils.enums import *
from Weight.xsec_utils import *

from pandana.core.var import Var

import numpy as np
import pandas as pd

def kRescaleMAQE(tables):
  correctionInSigma = (1.04 - 0.99) / 0.25

  genie_plus1 = tables['rec.mc.nu.rwgt.genie']['plus1sigma']
  genie_plus1 = genie_plus1.groupby(level=KLN).nth(genie.fReweightMaCCQE)
  genie_plus1 = 1 + correctionInSigma*(genie_plus1 - 1)

  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kQE) & \
        (df['iscc'] == 1)
  weight = pd.Series(1, sel.index)
  weight = weight.where(~sel, genie_plus1)
  return weight
kRescaleMAQE = Var(kRescaleMAQE)


def kFixNonres1Pi(tables):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kDIS) & \
        (df['W2'] <=  1.7*1.7) & \
        (df['pdg'] >= 0) & \
        (df[['npiplus', 'npiminus', 'npizero']].sum(axis=1) == 1)

  weight = pd.Series(1, sel.index)
  # keeping typo in 2018 version for now. It should actually be 0.43 
  weight = weight.where(~sel, 0.41)
  return weight
kFixNonres1Pi = Var(kFixNonres1Pi)

def kRescaleHighWDIS(tables):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kDIS) & \
        (df['W2'] >= 1.7*1.7) & \
        (df['pdg'] >= 0)
  
  weight = pd.Series(1, sel.index)
  weight = weight.where(~sel, 1.1)
  return weight
kRescaleHighWDIS = Var(kRescaleHighWDIS)

RPACCQECalc = RPAWeightCCQE_2017()
def kRPAWeightCCQE(tables):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kQE) & \
        (df['iscc'] == 1)

  q0 = (df['E']*df['y'])[sel]
  qmag = np.sqrt(df['q2'] + q0**2)[sel]
  isAntiNu = (df['pdg'] < 0)[sel]
  out = pd.Series(RPACCQECalc.GetWeightVectored(q0, qmag, isAntiNu), q0.index)

  weight = pd.Series(1, sel.index)
  weight = weight.where(~sel, out)
  return weight
kRPAWeightCCQE = Var(kRPAWeightCCQE)

RPAQ2Calc = RPAWeightQ2_2017()
def kRPAWeightRES(tables):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kRes) & \
        (df['iscc'] == 1)

  q2 = df['q2'][sel]
  isAntiNu = (df['pdg'] < 0)[sel]
  out = pd.Series(RPAQ2Calc.GetWeightVectored(q2, isAntiNu), q2.index)

  weight = pd.Series(1, sel.index)
  weight = weight.where(~sel, out)
  return weight
kRPAWeightRES = Var(kRPAWeightRES)

MECCalc = EmpiricalMECWgt2018()
def kEmpiricalMECWgt(tables):
  df = tables['rec.mc.nu']
  sel = (df['mode'] == mode.kMEC)
  
  q0 = (df['E']*df['y'])[sel]
  qmag = np.sqrt(df['q2'] + q0**2)[sel]
  isAntiNu = (df['pdg'] < 0)[sel]
  out = pd.Series(MECCalc.GetWeightVectored(q0, qmag, isAntiNu), q0.index)

  weight = pd.Series(1, sel.index)
  weight = weight.where(~sel, out)
  return weight
kEmpiricalMECWgt = Var(kEmpiricalMECWgt)

kPPFXFluxCVWgt_NT = Var(lambda tables: tables['rec.mc.nu.rwgt.ppfx']['cv'])

kXSecCVWgt2018_NT = kRescaleMAQE     * \
                    kRPAWeightCCQE   * \
                    kRPAWeightRES    * \
                    kFixNonres1Pi    * \
                    kRescaleHighWDIS * \
                    kEmpiricalMECWgt

# Only use nu weights for slices with at least one nu
# Protect the cosmics!
def NuWeightToSliceWeight(weight):
  def kSlcWeight(tables):
    nuweight = weight(tables).groupby(KL).first()

    # This is hacky, but we want nnu==0 to have a weight of 1
    # and nnu == 1 to have a weight of nuweight
    df = 1-tables['rec.mc']['nnu']
    return df.where(df==1, nuweight)
  return Var(kSlcWeight)

kCVWgt2018 = NuWeightToSliceWeight(kXSecCVWgt2018_NT * kPPFXFluxCVWgt_NT)
