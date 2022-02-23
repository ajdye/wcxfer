from Utils.index import KL
from Utils.enums import *
from Weight.xsec_utils import *
from Weight.novarwgt import *
from pandana.core.var import Var

import numpy as np
import pandas as pd


class ValenciaMECWgt2020(DoubleGaussMECWgt):
  # parameters of high q3q0 gaussian
  norm = 14.85
  mean_q0 = 0.36
  mean_q3 = 0.86
  sigma_q0 = 0.13
  sigma_q3 = 0.35
  corr = 0.89

  # parameters of low q3q0 gaussian
  norm_2 = 42.0
  mean_q0_2 = 0.034
  mean_q3_2 = 0.45
  sigma_q0_2 = 0.044
  sigma_q3_2 = 0.31
  corr_2 = 0.75

  baseline = -0.08
  
  def __init__(self):
    super().__init__(ValenciaMECWgt2020.norm,
                     ValenciaMECWgt2020.mean_q0,
                     ValenciaMECWgt2020.mean_q3,
                     ValenciaMECWgt2020.sigma_q0,
                     ValenciaMECWgt2020.sigma_q3,
                     ValenciaMECWgt2020.corr,
                     ValenciaMECWgt2020.norm_2,
                     ValenciaMECWgt2020.mean_q0_2,
                     ValenciaMECWgt2020.mean_q3_2,
                     ValenciaMECWgt2020.sigma_q0_2,
                     ValenciaMECWgt2020.sigma_q3_2,
                     ValenciaMECWgt2020.corr_2,
                     ValenciaMECWgt2020.baseline)
  def __call__(self, tables):
    return self.CalcWeight(tables)
kValenciaMECWgt2020 = Var(ValenciaMECWgt2020())


def kNCVWgt2020(tables):
  evt = EventRecordDF(tables)
  # this weight will eventually be calcuated by
  # a BDT using properties of evt.
  # For now just load an EventRecordDF to demonstrate
  # IO load
  return pd.Series(1, index=evt.Enu.index)
kNCVWgt2020 = Var(kNCVWgt2020)

kXSecCVWgt2020 = kNCVWgt2020 * kValenciaMECWgt2020

def kPPFXFluxCVWgt(tables):
  cv = tables['rec.mc.nu.rwgt.ppfx']['cv']
  weight = pd.Series(1, index=cv.index)
  weight[cv <= 90] = cv[cv <= 90]
  return weight
kPPFXFluxCVWgt = Var(kPPFXFluxCVWgt)

kCVWgt2020 = kXSecCVWgt2020 * kPPFXFluxCVWgt

  

