import numpy as np
import pandas as pd
from pandana.core.var import Var
from pandana.core.cut import Cut
from NOvAPandAna.Utils.misc import *
from NOvAPandAna.Utils.index import KL
from NOvAPandAna.Core.NOVALoader import NOVALoader
from NOvAPandAna.Core.NOVASpectrum import NOVASpectrum
from matplotlib import pyplot as plt

vtx = 'rec.vtx.elastic'
vtxk = vtx + '.fuzzyk'
vtxks = vtxk + '.png.shwlid'
kVtx = Var(lambda tables: tables[vtx]['IsValid'])
kVtxX = Var(lambda tables: tables[vtx]['vtx.x'])
kVtxY = Var(lambda tables: tables[vtx]['vtx.y'])
kVtxZ = Var(lambda tables: tables[vtx]['vtx.z'])

kVtxNShwlid = Var(lambda tables: tables[vtxk]['nshwlid'])

kVtxShwlidStopX = Var(lambda tables: tables[vtxks]['stop.x'])
kVtxShwlidStopY = Var(lambda tables: tables[vtxks]['stop.y'])
kVtxShwlidStopZ = Var(lambda tables: tables[vtxks]['stop.z'])

kVtxNPng = Var(lambda tables: tables[vtxk]['npng'])

kVtxMaxPlaneCont = Var(lambda tables: tables[vtxk+'.png']['maxplanecont'])
kVtxMaxPlaneGap = Var(lambda tables: tables[vtxk+'.png']['maxplanegap'])

kPhotonID = Var(lambda tables: tables[vtxk+'.png.cvnpart']['photonid'])
kRemid = Var(lambda tables: tables['rec.sel.remid']['pid'])

deadscale = 0.8747 #0.8720

kProngE = Var(lambda tables: deadscale*1000*tables[vtxk+'.png']['calE'])

kProngL = Var(lambda tables: 1000*tables[vtxk+'.png']['len'])


def kProngdEdx(tables):
    df = (kProngE(tables)/kProngL(tables)) < 0.004
    return df.groupby(KL).all()

kdEdxCut = Cut(kProngdEdx) 

def kMass(tables):
    kDirs = tables[vtxk+'.png'][['dir.x','dir.y','dir.z']]
    kDirs2 = kDirs**2
    norm = kDirs.divide(np.sqrt(kDirs2.sum(axis=1)),axis=0)
    dot = norm.groupby(KL).prod().sum(axis=1)
    prodE = tables[vtxk+'.png']['calE'].groupby(KL).prod()
    return  deadscale*1000 * np.sqrt(2*prodE*(1-dot))
kMass = Var(kMass)

def kNueShwContain(tables):
    a = kVtxNShwlid(tables) != 0
    X = kVtxShwlidStopX(tables).abs() < 180
    Y = kVtxShwlidStopY(tables).abs() < 180
    Z = (kVtxShwlidStopZ(tables) < 1200)  &  (kVtxShwlidStopZ(tables) > 200)
    return (a  &  X  &  Y  &  Z).groupby(KL).all()
kNueShwContain = Cut(kNueShwContain)

def kNueVtxContain(tables):
    X = kVtxX(tables).abs() < 180
    Y = kVtxY(tables).abs() < 180
    Z = (kVtxZ(tables) < 1000)  &  (kVtxZ(tables) > 50)
    return (X  &  Y  &  Z).groupby(KL).all()
kNueVtxContain = Cut(kNueVtxContain)

kTwoProngs = (kVtx == 1)  &  (kVtxNPng == 2)

def kContigPlanesCut(tables):
    df = kVtxMaxPlaneCont(tables) > 4
    return df.groupby(KL).all()
kContigPlanesCut = Cut(kContigPlanesCut)

def kMissingPlanesCut(tables):
    df = kVtxMaxPlaneGap(tables) > 1 
    return df.groupby(KL).all()
kMissingPlanesCut = Cut(kMissingPlanesCut)
def kPhotonIDCut(tables):
    df = (kPhotonID(tables) > 0.75)
    return df.groupby(KL).all()
kPhotonIDCut = Cut(kPhotonIDCut)

kRemidCut = (kRemid > 0)  &  (kRemid < 0.5)

kVtxCut = kVtx == 1
qual = kNueVtxContain  &  kNueShwContain

qual_base = qual  &  kTwoProngs  &  kContigPlanesCut  &  kMissingPlanesCut & kVtxCut

SAsel = qual_base  &  kRemidCut

SA_dEdx = SAsel  &  kdEdxCut

CVN_two = qual_base  &  kPhotonIDCut

CVNASA_dEdx = CVN_two  &  SA_dEdx

tables = NOVALoader("/home/krint/pandanaEnv/testData.h5")

spec = NOVASpectrum(tables,CVNASA_dEdx,kMass)

tables.Go()

print(spec.entries())

plt.style.use("dark_background")
n,bins = spec.histogram(25,(0,500))
fig,ax = plt.subplots()
plt.hist(bins[:-1],bins,weights=n,histtype="step", fill=True)
ax.set_xlabel("Mass (MeV)")
ax.set_title("Pi0 Mass Peak")
ax.tick_params(left = False, bottom = True)
ax.grid(False)
plt.xticks(ticks=[135]) 
plt.show()
