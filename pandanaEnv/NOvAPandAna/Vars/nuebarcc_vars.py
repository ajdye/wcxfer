import numpy as np
import pandas as pd

from pandana import Var
from pandana import VarsToVarND
from Utils.index import KL, KLN

from NOvAPandAna.Vars.nueid import NueID

# select row from sublevel
def select(table, row, level=KL, sort=False):
    if row == 0:
        return table.groupby(level, sort=sort).first()
    return table.groupby(level, sort=sort).nth(row)

"""
Low-level reconstruction variables
"""
kHitsPerPlane = Var(lambda tables: tables['rec.sel.nuecosrej']['hitsperplane'])
kNHit     = Var(lambda tables: tables['rec.slc']['nhit'])
kNPngs    = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk']['npng'])
kNShwLID  = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk']['nshwlid'])
kShwNHitX = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.shwlid']['nhitx'])
kShwNHitY = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.shwlid']['nhity'])
kShwGap   = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.shwlid']['gap'])

# The nuecosrej table is removed in the decaf files. Compute this on the fly.
def kDecafHitsPerPlane(tables):
    pngplanes = tables['rec.vtx.elastic.fuzzyk.png']['nplane']
    pngplanes = select(pngplanes, 0)
    nhit = tables['rec.slc']['nhit']
    return nhit / pngplanes
kDecafHitsPerPlane = Var(kDecafHitsPerPlane)

def kLeadingShowersCosAngle(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png.shwlid'][['dir.x', 'dir.y', 'dir.z']]
    
    group = df.groupby(KL)
    prim = group.first()
    sec = group.nth(1)

    # sec will be missing rows for events with only one prong
    # The result of multiply will be nan for these rows
    # Just replace with 1
    dot = prim.multiply(sec).sum(axis=1, min_count=1).fillna(1)
    return dot
kLeadingShowersCosAngle = Var(kLeadingShowersCosAngle)

# Have to convert the type before subtracting since these are unsigned ints
kShwXYHitAsymm = Var(lambda tables: abs(kShwNHitX(tables).astype('int16') - kShwNHitY(tables).astype('int16')) / (kShwNHitX(tables) + kShwNHitY(tables)))

def kAllShwHits(tables):
    nhit = tables['rec.vtx.elastic.fuzzyk.png.shwlid']['nhit']
    return nhit.groupby(KL).sum()
kAllShwHits = Var(kAllShwHits)

kNPlanesToFront = Var(lambda tables: tables['rec.sel.contain']['nplanestofront'])

kDistAllPngsTop    = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngtop'   ])
kDistAllPngsBottom = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngbottom'])
kDistAllPngsEast   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngeast'  ])
kDistAllPngsWest   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngwest'  ])
kDistAllPngsFront  = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngfront' ])

# Once again we need special decaf version which compute on the fly.
def MakeContainDF(tables, axis, func):
    df = tables['rec.vtx.elastic.fuzzyk.png.shwlid'][['start.'+axis,'stop.'+axis]]
    if func == 'max':
        df = df.max(axis=1)
        df = df.groupby(KL).max()
    elif func == 'min':
        df = df.min(axis=1)
        df = df.groupby(KL).min()
    else:
        print("We shouldn't be here. Returning garbage probably.")
    return df

kDecafDistAllPngsTop    = Var(lambda tables: 193 - MakeContainDF(tables, 'y', 'max'))
kDecafDistAllPngsBottom = Var(lambda tables: 187 + MakeContainDF(tables, 'y', 'min'))
kDecafDistAllPngsEast   = Var(lambda tables: 191 + MakeContainDF(tables, 'x', 'min'))
kDecafDistAllPngsWest   = Var(lambda tables: 192 - MakeContainDF(tables, 'x', 'max'))
kDecafDistAllPngsFront  = Var(lambda tables: MakeContainDF(tables, 'z', 'min'))

def kAllShwMaxZ(tables):
    z = tables['rec.vtx.elastic.fuzzyk.png.shwlid'][['start.z','stop.z']]
    return z.max(axis=1).groupby(KL).max()
kAllShwMaxZ = Var(kAllShwMaxZ)

kRemID = Var(lambda tables: tables['rec.sel.remid']['pid'])        

# Taken from https://cdcvs.fnal.gov/redmine/projects/novaart/repository/entry/trunk/CAFAna/Core/Utilities.cxx
BeamDirFD = np.array([-6.83271078e-05,  6.38772962e-02,  9.97957758e-01])
BeamDirND = np.array([-8.42393199e-04, -6.17395015e-02,  9.98091942e-01])


"""
Reconstructed Analysis Variables
"""
def kRecoElectronCosTheta(tables):
    df = tables['rec.vtx.elastic.fuzzyk.png'][['dir.x', 'dir.y', 'dir.z']]
    elecdir = select(df, 0)
    costheta = elecdir.mul(BeamDirND, axis=1).sum(axis=1)
    costheta.name = 'elec_costheta'
    return costheta
kRecoElectronCosTheta = Var(kRecoElectronCosTheta)

def kRecoElectronE(tables):
    """
    Chi2                      =      4.78147
    NDf                       =           57
    p0                        =   -0.0861793   +/-   0.0460596   
    p1                        =     0.235812   +/-   0.0506663   
    p2                        =   -0.0348964   +/-   0.0102995 
    """
    p0 = -0.0861793
    p1 = 0.235812
    p2 = -0.0348964

    def correct_bias(shwE):
        if shwE <= 6:
            return max(0, shwE - (p0 + shwE * p1 + shwE**2 * p2))
        else:
            return shwE
    correct_bias = np.vectorize(correct_bias, otypes=[np.float32])

    shwE = tables['rec.vtx.elastic.fuzzyk.png.shwlid']['shwE']
    shwE = select(shwE, 0)
    shwE = pd.Series(correct_bias(shwE), shwE.index)
    shwE.name = 'elec_e'
    return shwE
kRecoElectronE = Var(kRecoElectronE)

kSliceCalE = Var(lambda tables: tables['rec.slc']['calE'])

kRecoHadE = kSliceCalE - kRecoElectronE

def kRecoNeutrinoE(tables):
    # TODO need fitting function here
    return kRecoElectronE(tables) + kRecoHadE(tables)
kRecoNeutrinoE = Var(kRecoNeutrinoE)

def kRecoQ2(tables):
    elec_e = kRecoElectronE(tables)
    nu_e   = kRecoNeutrinoE(tables)
    elec_cos = kRecoElectronCosTheta(tables)

    elec_mass_sq = 0.000511**2 # 0.511 MeV
    elec_p = np.sqrt(elec_e**2 - elec_mass_sq)
    
    return 2 * nu_e * (elec_e - elec_p * elec_cos) - elec_mass_sq
kRecoQ2 = Var(kRecoQ2)

kRecoElectronEVsCosTheta = VarsToVarND([kRecoElectronE, kRecoElectronCosTheta])

"""
Truth Analysis Variables
"""

kTrueNeutrinoE = Var(lambda tables: tables['rec.mc.nu']['E'])
kTrueQ2        = Var(lambda tables: tables['rec.mc.nu']['q2'])

def kTrueElectronE(tables):
    prim = tables['rec.mc.nu.prim'][['pdg', 'p.E']]
    E = prim['p.E'][prim['pdg'].abs()==11]
    return E.groupby(KLN).first()
    
def kTrueElectronCosTheta(tables):
    prim = tables['rec.mc.nu.prim'][['pdg', 'p.px', 'p.py', 'p.pz']]
    elec_p = prim[['p.px', 'p.py', 'p.pz']][prim['pdg'].abs() == 11]
    elec_p = elec_p.groupby(KLN).first()
    elec_dir = elec_p.div(np.sqrt((elec_p**2).sum(axis=1)), axis=0)

    costheta = elec_dir.mul(BeamDirND, axis=1).sum(axis=1)
    costheta.name = 'costheta'
    return costheta
kTrueElectronCosTheta = Var(kTrueElectronCosTheta)

kTrueElectronEVsCosTheta = VarsToVarND([kTrueElectronE, kTrueElectronCosTheta])
    

"""
True Particle Cuts
"""
kNeutrinoTruthPDG = Var(lambda tables: tables['rec.mc.nu']['pdg'])
kNeutrinoTruthIsCC = Var(lambda tables: tables['rec.mc.nu']['iscc'])    



def kMuonID(tables):
    muonid = tables['rec.trk.kalman.tracks']['muonid']
    return muonid.groupby(KL).max()
kMuonID = Var(kMuonID)

