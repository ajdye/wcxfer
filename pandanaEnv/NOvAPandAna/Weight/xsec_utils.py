import pandas as pd
import numpy as np
import h5py
import os

from pandana.utils import *
from Utils.enums import *

nova_xsec_data = os.path.join(os.getcwd(), os.path.dirname(__file__))

class NuWeightDFWrapper():
  def __init__(self, df):
    self.df = df
    self.dim = (int)(len(self.df.columns)-1)/2

    self.xlow = df['xlow'].unique()
    self.xhigh = df['xhigh'].unique()
    if self.dim > 1:
      self.ylow = df['ylow'].unique()
      self.yhigh = df['yhigh'].unique()
  
  def GetNbinsX(self):
    nxlow = len(self.xlow)
    nxhigh = len(self.xhigh)
    assert nxlow == nxhigh
    return nxhigh
  
  def GetNbinsY(self):
    assert self.dim > 1
    nylow = len(self.ylow)
    nyhigh = len(self.yhigh)
    assert nylow == nyhigh
    return nyhigh

  def FindXBin(self, xval):
    if min(self.xlow) > xval:
      return 0
    if max(self.xhigh) <= xval:
      return self.GetNbinsX()+1
    else:
      xbin_arr = np.where((self.xlow <= xval) & (self.xhigh > xval))[0]
      assert len(xbin_arr) == 1
      return xbin_arr[0]+1
  
  def FindYBin(self, yval):
    assert self.dim > 1
    if min(self.ylow) > yval:
      return 0
    if max(self.yhigh) <= yval:
      return self.GetNbinsY()+1
    else:
      ybin_arr = np.where((self.ylow <= yval) & (self.yhigh > yval))[0]
      assert len(ybin_arr) == 1
      return ybin_arr[0]+1

  def GetBinContent(self, binx, biny=-1):
    if biny < 0: 
      assert self.dim == 1
      if binx < 1 or binx > self.GetNbinsX(): return 1.
      return self.df['weight'][binx-1]
    else:
      assert self.dim > 1
      if binx < 1 or biny < 1: return 1.
      if binx > self.GetNbinsX() or biny > self.GetNbinsY(): return 1.
      binidx = (binx-1)*self.GetNbinsY() + (biny-1)
      return self.df['weight'][binidx]
  
  def FindFirstBinAbove(self, threshold, axis):
    assert (axis == 1) or (axis < 3 and self.dim == 2)
    firstbin = self.df[self.df['weight'] > threshold].index[0]
    return (axis-1)*((firstbin % self.GetNbinsY())+1) + (2-axis)*((firstbin/self.GetNbinsX())+1)

  def GetValue(self, val):
    assert self.dim == 1
    return self.GetBinContent(self.FindXBin(val))

  def GetValueInRange(self, vals, maxrange=[-float("inf"), float("inf")], binranges=[[1,-1], [1,-1]]):
    assert self.dim == 2
    if binranges[0][1] < 0: binranges[0][1] = self.GetNbinsX()
    if binranges[1][1] < 0: binranges[1][1] = self.GetNbinsY()
    
    binx = self.FindXBin(vals[0])
    biny = self.FindYBin(vals[1])
    if binx > binranges[0][1]: binx = binranges[0][1]
    elif binx < binranges[0][0]: binx = binranges[0][0]
    if biny > binranges[1][1]: biny = binranges[1][1]
    elif biny < binranges[1][0]: biny = binranges[1][0]

    val = self.GetBinContent(binx, biny)
    if val < maxrange[0]: val = maxrange[0]
    if val > maxrange[1]: val = maxrange[1]
    return val
    
class NuWeightFromFile():
  def __init__(self, fnu, fnubar, forcenu=False):
    fnuh5 = h5py.File(fnu['file'], 'r')
    self.nu = NuWeightDFWrapper(
      pd.DataFrame(fnuh5.get(fnu['group']+'/block0_values')[()],
                   columns=fnuh5.get(fnu['group']+'/block0_items')[()].astype(str)))
    if not forcenu:
      fnubarh5 = h5py.File(fnubar['file'], 'r')
      self.nubar = NuWeightDFWrapper(
        pd.DataFrame(fnubarh5.get(fnubar['group']+'/block0_values')[()],
                     columns=fnubarh5.get(fnubar['group']+'/block0_items')[()].astype(str)))
      assert self.nu.dim == self.nubar.dim
    self.dim = self.nu.dim

    self.GetWeightVectored = np.vectorize(self.GetWeight)

  def GetWeight(self):
    return 1.

class RPAWeightCCQE_2017(NuWeightFromFile):
  def __init__(self):
    fnu = {}
    fnubar = {}
    #fnu['file'] = FindpandanaDir()+"/Data/xs/RPA2017.h5"
    fnu['file'] = os.path.join(nova_xsec_data, "RPA2017.h5")
    fnubar['file'] = fnu['file']
    fnu['group'] = "RPA_CV_nu"
    fnubar['group'] = "RPA_CV_nubar"
    NuWeightFromFile.__init__(self, fnu, fnubar)

  def GetWeight(self, q0, qmag, IsAntiNu):
    df = self.nubar if IsAntiNu else self.nu
    
    val = df.GetValueInRange([qmag, q0],
                             [0., 2.],
                             [[1, df.GetNbinsX()],[1,df.GetNbinsY()]])
    if val == 0.: val = 1.
    return val

class RPAWeightQ2_2017(NuWeightFromFile):
  def __init__(self):
    fnu = {}
    fnubar = {}
    #fnu['file'] = FindpandanaDir()+"/Data/xs/RPA2017.h5"
    fnu['file'] = os.path.join(nova_xsec_data, "RPA2017.h5")
    fnubar['file'] = fnu['file']
    fnu['group'] = "RPA_Q2_CV_nu"
    fnubar['group'] = "RPA_Q2_CV_nubar"
    NuWeightFromFile.__init__(self, fnu, fnubar)

  def GetWeight(self, q2, IsAntiNu):
    df = self.nubar if IsAntiNu else self.nu
    return df.GetValue(q2)

class EmpiricalMECWgt2018(NuWeightFromFile):
  def __init__(self):
    fnu = {}
    fnubar = {}
    #fnu['file'] = FindpandanaDir()+"/Data/xs/rw_empiricalMEC2018.h5"
    fnu['file'] = os.path.join(nova_xsec_data, "rw_empiricalMEC2018.h5")
    fnubar['file'] = fnu['file']
    fnu['group'] = "numu_mec_weights_smoothed"
    fnubar['group'] = "numubar_mec_weights_smoothed"
    NuWeightFromFile.__init__(self, fnu, fnubar)

  def GetWeight(self, q0, qmag, IsAntiNu):
    df = self.nubar if IsAntiNu else self.nu
    
    val = df.GetValueInRange([qmag, q0])
    if val < 0.: val = 0.
    return val

class DoubleGaussMECWgt:
  def __init__(self,
               norm,
               mean_q0,
               mean_q3,
               sigma_q0,
               sigma_q3,
               corr,
               norm_2,
               mean_q0_2,
               mean_q3_2,
               sigma_q0_2,
               sigma_q3_2,
               corr_2,
               baseline):
    self._norm = norm
    self._mean_q0 = mean_q0
    self._mean_q3 = mean_q3
    self._sigma_q0 = sigma_q0
    self._sigma_q3 = sigma_q3
    self._corr = corr
    self._norm_2 = norm_2
    self._mean_q0_2 = mean_q0_2
    self._mean_q3_2 = mean_q3_2
    self._sigma_q0_2 = sigma_q0_2
    self._sigma_q3_2 = sigma_q3_2
    self._corr_2 = corr_2
    self._baseline = baseline


  def CalcWeight(self, tables):
    nu = tables['rec.mc.nu']
    q0 = nu[['E', 'y']].prod(axis=1)
    q3 = (nu['q2'] + (q0.pow(2))).pow(0.5)

    where = nu['mode'] == mode.kMEC
    weight = pd.Series(1, index=nu['E'].index)
    weight[where] = self.SimpleMECDoubleGaussEnh(q0, q3)
    return weight
    
  def SimpleMECDoubleGaussEnh(self, q0, q3):
    z = ( (q0 - self._mean_q0) / self._sigma_q0 ) ** 2 + \
        ( (q3 - self._mean_q3) / self._sigma_q3 ) ** 2 - \
        2 * self._corr * ( q0 - self._mean_q0 ) * ( q3 - self._mean_q3 ) / \
        (self._sigma_q0 * self._sigma_q3)

    z_2 = ( (q0 - self._mean_q0_2) / self._sigma_q0_2 ) ** 2 + \
          ( (q3 - self._mean_q3_2) / self._sigma_q3_2 ) ** 2 - \
          2 * self._corr_2 * ( q0 - self._mean_q0_2 ) * ( q3 - self._mean_q3_2 ) / \
          (self._sigma_q0_2 * self._sigma_q3_2)

    weight = self._baseline + \
             self._norm * np.exp( -0.5 * z / ( 1 - self._corr ** 2 ) ) + \
             self._norm_2 * np.exp( -0.5 * z_2 / ( 1 - self._corr_2 ** 2 ) )
    weight[weight < 0] = 0
    return weight

"""
class BDTReweighter:
  def __init__(self, model_file, multiplier = 1):
    self._multiplier = multiplier

    # somehow init xgboost model with file

  def predict(self, data, pred_margin):
    # call to xgboost api 
    pass

  def GetWeight(self, data, pred_margin):
    return self._multiplier * self.predict(data, pred_margin)


class IHNBDTWgtr:
  def __init__(self, name, bdtWgtrNu, bdtWgtrNubar):
    self._name = name
    self._bdtNu = bdtWgtrNu
    self._bdtNubar = bdtWgtrNubar
    

  def CalcWeight(tables):
    nu = tables['rec.mc.nu']
    prims = tables['rec.mc.nu.prim']
    ##
    ## BDT features
    ## 
    A          = nu['tgtA'  ]
    struckNucl = nu['hitnuc']
    iscc       = nu['iscc'  ]
    nupdg      = nu['pdg'   ]
    Enu        = nu['E'     ]
    Q2         = nu['q2'] + nu[['E', 'y']].prod()
    W          = nu['W2'].pow(0.5)
    y          = nu['y']
    
    # particle multiplicities
    nneutron = (prims['pdg'] == 2112).groupby(KL).apply(sum) # neutron
    nproton  = (prims['pdg'] == 2212).groupby(KL).apply(sum) # proton
    npi0     = (prims['pdg'] == 111 ).groupby(KL).apply(sum) # pi0
    npiminus = (prims['pdg'] == -211).gropuby(KL).apply(sum) # pi-
    npiplus  = (prims['pdg'] == 211 ).groupby(KL).apply(sum) # pi+

    # FS particle energies
    e_neutron = prims[prims['pdg'] == 2112]['p.E'].groupby(KL).apply(sum) # neutron
    e_proton  = prims[prims['pdg'] == 2212]['p.E'].groupby(KL).apply(sum) # proton
    e_pi0     = prims[prims['pdg'] == 111 ]['p.E'].groupby(KL).apply(sum) # pi0
    e_piminus = prims[prims['pdg'] == -211]['p.E'].groupby(KL).apply(sum) # pi-
    e_piplus  = prims[prims['pdg'] == 211 ]['p.E'].groupby(KL).apply(sum) # pi+

    features = pd.concat([A,
                          struckNucl,
                          iscc,
                          nupdg,
                          Enu,
                          Q2,
                          W,
                          y,
                          nneutron,
                          nproton,
                          npi0,
                          npiminus,
                          npiplus,
                          e_neutron,
                          e_proton,
                          e_pi0,
                          e_piminus,
                          e_piplus],
                         axis=1)

    weights_nu    = self._bdtNu   .GetWeight(features[nu['pdg'] > 0], 0)
    weights_nubar = self._bdtNubar.GetWeight(features[nu['pdg'] < 0], 0)

    return pd.concat([weights_nu, weights_nubar])
    
  
"""
