
import os.path
import numpy as np
import random
import collections
from pandana import *
#import ROOT

# avogadros constant
N_A = 6.02214086e23

def IsFiducial(vtx, min_extent, max_extent):
    return \
        (vtx < min_extent).all(axis=1) & \
        (vtx > max_extent).all(axis=1)

def DeriveFlux(loader, pdg,
               min_extent=None, max_extent=None, weight=None):

    kTrueNeutrinoVertex = Var(lambda tables: tables['neutrino'][['vtx.x', 'vtx.y', 'vtx.z']])
    kTrueNeutrinoEnergy = Var(lambda tables: tables['neutrino']['E'])

    kIsFiducial = Cut(lambda tables: IsFiducial(kTrueNeutrinoVertex(tables),
                                                min_extent,
                                                max_extent))
    def kIsNCQEOnCarbon(pdg):
        return \
            Cut(lambda tables: tables['neutrino']['pdg'      ] == pdg ) & \
            Cut(lambda tables: tables['neutrino']['iscc'     ] == 0   ) & \
            Cut(lambda tables: tables['neutrino']['inttype'  ] == 1002) & \
            Cut(lambda tables: tables['neutrino']['isvtxcont'] == 1   ) & \
            Cut(lambda tables: tables['neutrino']['tgtZ'     ] == 6   ) & \
            Cut(lambda tables: tables['neutrino']['tgtA'     ] == 12  ) & \
            Cut(lambda tables: tables['neutrino']['ischarm'  ] == 0   ) & \
            Cut(lambda tables: tables['neutrino']['vtx.z'    ] <  1280) & \
            Cut(lambda tables: tables['neutrino']['hitnuc'   ] == 2112)            

    

    # count all carbons: nucleons -> nuclei
    nuclei = 3.05982283905e+31 / 12 ## here's a dummy value so we don't have to run TargetCount
    if min_extent is None or max_extent is None:
        #nuclei = TargetCount(Z=6).NNucleons() / 12 
        truth_cut = kIsNCQEOnCarbon(pdg)
    else:
        #nuclei = TargetCount(min_extent=min_extent,
        #                     max_extent=max_extent,
        #                     npoints=100,#npoints=1000000,
        #                     Z=6).NNucleons() / 12
        truth_cut = kIsNCQEOnCarbon(pdg) & kIsFiducial

    def xsec_weight(tables):
        # GENIE uses GeV internally. We ultimately want a flux in m^-2
        GeV2perm2 = 2.56819e31
        xsec = tables['neutrino']['xsec'] / GeV2perm2
        weight = (xsec*nuclei)
        return weight

    if weight is None:
        total_weight = xsec_weight
    else:
        total_weight = weight * xsec_weight

    return Spectrum(loader, truth_cut, kTrueNeutrinoEnergy, weight=total_weight)
        

    


        
