from pandana import Cut
from NOvAPandAna.Vars.nuebarcc_vars import *
import numpy as np
from collections import OrderedDict



kMC             = Cut(lambda tables: tables['rec.hdr']['ismc'] == 1)
kDQVtxIsValid   = Cut(lambda tables: tables['rec.vtx.elastic']['IsValid'] == 1)
kDQHasProngs    = kNPngs > 0
kDQHasShwLID    = kNShwLID > 0
kDQHitsPerPlane = kHitsPerPlane < 8
kDecafDQHitsPerPlane = kDecafHitsPerPlane < 8
kDQShwFrac      = Cut(lambda tables: (kAllShwHits(tables) / kNHit(tables)) > 0.7)
kDQShwAsymm     = Cut(lambda tables: select(kShwXYHitAsymm(tables) < 0.4, 0))
kDQShwNHit      = Cut(lambda tables: select((kShwNHitX(tables) > 5) & (kShwNHitY(tables) > 5), 0))
kDQShwGap       = Cut(lambda tables: select(kShwGap(tables) <=100, 0))
kDQShwsAngle    = kLeadingShowersCosAngle >= -0.95
kDQ = \
    kDQVtxIsValid   & \
    kDQHasProngs    & \
    kDQHasShwLID    & \
    kDQHitsPerPlane & \
    kDQShwFrac      & \
    kDQShwAsymm     & \
    kDQShwNHit      & \
    kDQShwGap       & \
    kDQShwsAngle

kDecafDQ = \
    kDQVtxIsValid   & \
    kDQHasProngs    & \
    kDQHasShwLID    & \
    kDecafDQHitsPerPlane & \
    kDQShwFrac      & \
    kDQShwAsymm     & \
    kDQShwNHit      & \
    kDQShwGap       & \
    kDQShwsAngle

kFrontPlanes = kNPlanesToFront > 6
kNHits = (kNHit >= 20) & (kNHit <= 200)

vNuebarCCIncFiducialMax = np.array([ 150,  140, 800])
vNuebarCCIncFiducialMin = np.array([-130, -140, 150])
def kNuebarCCIncFiducial(tables):
    df = tables['rec.vtx.elastic']
    vtx = df[['vtx.x', 'vtx.y', 'vtx.z']]
    return \
        (vtx < vNuebarCCIncFiducialMax).all(axis=1) & \
        (vtx > vNuebarCCIncFiducialMin).all(axis=1)
kNuebarCCIncFiducial = Cut(kNuebarCCIncFiducial)

def kNuebarCCIncFiducialST(tables):
    df = tables['rec.mc.nu']
    vtx = df[['vtx.x', 'vtx.y', 'vtx.z']]
    return \
        (vtx < vNuebarCCIncFiducialMax).all(axis=1) & \
        (vtx > vNuebarCCIncFiducialMin).all(axis=1)
kNuebarCCIncFiducialST = Cut(kNuebarCCIncFiducialST)

vNuebarCCIncFiducialLooseMax = np.array([ 165,  165, 1000])
vNuebarCCIncFiducialLooseMin = np.array([-165, -165, 100 ])
def kNuebarCCIncFiducialLoose(tables):
    df = tables['rec.vtx.elastic']
    vtx = df[['vtx.x', 'vtx.y', 'vtx.z']]
    return \
        (vtx < vNuebarCCIncFiducialLooseMax).all(axis=1) & \
        (vtx > vNuebarCCIncFiducialLooseMin).all(axis=1)    
kNuebarCCIncFiducialLoose = Cut(kNuebarCCIncFiducialLoose)


vNuebarCCIncContainmentBounds = \
    {'top': 50,
     'bottom': 30,
     'east': 50,
     'west': 30,
     'front': 150,
     'muon_catcher': 1250}
def kNuebarCCIncContainment(tables):
    muon_catcher = kAllShwMaxZ       (tables) < vNuebarCCIncContainmentBounds['muon_catcher']
    top          = kDistAllPngsTop   (tables) > vNuebarCCIncContainmentBounds['top'         ]
    bottom       = kDistAllPngsBottom(tables) > vNuebarCCIncContainmentBounds['bottom'      ]
    east         = kDistAllPngsEast  (tables) > vNuebarCCIncContainmentBounds['east'        ]
    west         = kDistAllPngsWest  (tables) > vNuebarCCIncContainmentBounds['west'        ]
    front        = kDistAllPngsFront (tables) > vNuebarCCIncContainmentBounds['front'       ]

    return \
        muon_catcher & \
        top          & \
        bottom       & \
        east         & \
        west         & \
        front        
kNuebarCCIncContainment = Cut(kNuebarCCIncContainment)

def kDecafNuebarCCIncContainment(tables):
    muon_catcher = kAllShwMaxZ       (tables) < vNuebarCCIncContainmentBounds['muon_catcher']
    top          = kDecafDistAllPngsTop   (tables) > vNuebarCCIncContainmentBounds['top'         ]
    bottom       = kDecafDistAllPngsBottom(tables) > vNuebarCCIncContainmentBounds['bottom'      ]
    east         = kDecafDistAllPngsEast  (tables) > vNuebarCCIncContainmentBounds['east'        ]
    west         = kDecafDistAllPngsWest  (tables) > vNuebarCCIncContainmentBounds['west'        ]
    front        = kDecafDistAllPngsFront (tables) > vNuebarCCIncContainmentBounds['front'       ]

    return \
        muon_catcher & \
        top          & \
        bottom       & \
        east         & \
        west         & \
        front        
kDecafNuebarCCIncContainment = Cut(kDecafNuebarCCIncContainment)


vNuebarCCIncContainmentLooseBounds = \
    {'top': 30,
     'bottom': 10,
     'east': 30,
     'west': 10,
     'front': 100,
     'muon_catcher': 1250}
def kNuebarCCIncContainmentLoose(tables):
    muon_catcher = kAllShwMaxZ       (tables) < vNuebarCCIncContainmentLooseBounds['muon_catcher']
    top          = kDistAllPngsTop   (tables) > vNuebarCCIncContainmentLooseBounds['top'         ]
    bottom       = kDistAllPngsBottom(tables) > vNuebarCCIncContainmentLooseBounds['bottom'      ]
    east         = kDistAllPngsEast  (tables) > vNuebarCCIncContainmentLooseBounds['east'        ]
    west         = kDistAllPngsWest  (tables) > vNuebarCCIncContainmentLooseBounds['west'        ]
    front        = kDistAllPngsFront (tables) > vNuebarCCIncContainmentLooseBounds['front'       ]
    return \
        muon_catcher & \
        top          & \
        bottom       & \
        east         & \
        west         & \
        front        
kNuebarCCIncContainmentLoose = Cut(kNuebarCCIncContainmentLoose)

# true neutrino cuts
kNC        = Cut(lambda tables: tables['rec.mc.nu']['iscc'] == 0   )
kNueCC     = Cut(lambda tables: tables['rec.mc.nu']['pdg' ] ==  12 ) & ~kNC
kNuebarCC  = Cut(lambda tables: tables['rec.mc.nu']['pdg' ] == -12 ) & ~kNC
kNumuCC    = Cut(lambda tables: tables['rec.mc.nu']['pdg' ] ==  14 ) & ~kNC
kNumubarCC = Cut(lambda tables: tables['rec.mc.nu']['pdg' ] == -14 ) & ~kNC


def kRecoElectronPhaseSpaceCut(tables):
    costheta = kRecoElectronCosTheta(tables)
    elece    = kRecoElectronE       (tables)
    return \
        ((costheta >= 0.97) & (elece >= 1.4) & (elece < 6.0)) | \
        ((costheta >= 0.94) & (costheta < 0.97) & (elece >= 1.4) & (elece < 4.1)) | \
        ((costheta >= 0.90) & (costheta < 0.94) & (elece >= 1.0) & (elece < 2.5)) | \
        ((costheta >= 0.85) & (costheta < 0.90) & (elece >= 1.0) & (elece < 2.0))
kRecoElectronPhaseSpaceCut = Cut(kRecoElectronPhaseSpaceCut)

kRemIDCut = kRemID < 0.6
kMuonIDCut = kMuonID < -0.55
kLoosePreselection = kDQ & kFrontPlanes & kNHits & kNuebarCCIncFiducialLoose & kNuebarCCIncContainmentLoose & kRecoElectronPhaseSpaceCut
kPreselection = kDQ & kFrontPlanes & kNHits & kNuebarCCIncFiducial & kNuebarCCIncContainment & kRecoElectronPhaseSpaceCut & kRemIDCut
kDecafPreselection = kDecafDQ & kFrontPlanes & kNHits & kNuebarCCIncFiducial & kDecafNuebarCCIncContainment & kRecoElectronPhaseSpaceCut & kMuonIDCut

kNueTemplateCut  = (kNueCC | kNuebarCC) & kNuebarCCIncFiducialST
kNumuTemplateCut = (kNumuCC | kNumubarCC)
kNCTemplateCut   = kNC




class CutFlow(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, name, cut):
        
        try:
            super().__setitem__(name, super().__getitem__(self.last()) & cut)
        except StopIteration:
            super().__setitem__(name, cut)
    def last(self):
        return next(reversed(self))


"""
kDQCutFlow = CutFlow()
kDQCutFlow['VtxIsValid'   ] = kDQVtxIsValid   
kDQCutFlow['HasProngs'    ] = kDQHasProngs    
kDQCutFlow['HasShwLID'    ] = kDQHasShwLID
kDQCutFlow['HitsPerPlane' ] = kDQHitsPerPlane 
kDQCutFlow['ShwFrac'      ] = kDQShwFrac      
kDQCutFlow['ShwAsymm'     ] = kDQShwAsymm     
kDQCutFlow['ShwNHit'      ] = kDQShwNHit      
kDQCutFlow['ShwGap'       ] = kDQShwGap       
kDQCutFlow['ShwsAngle'    ] = kDQShwsAngle    

kPreselectionCutFlow = CutFlow()
kPreselectionCutFlow['DQ'         ] = kDQ
kPreselectionCutFlow['FrontPlanes'] = kFrontPlanes
kPreselectionCutFlow['NHits'      ] = kNHits
kPreselectionCutFlow['Fiducial'   ] = kNuebarCCIncFiducial
kPreselectionCutFlow['Containment'] = kNuebarCCIncContainment
kPreselectionCutFlow['PhaseSpace' ] = kRecoElectronPhaseSpaceCut

kLoosePreselectionCutFlow = CutFlow()
kLoosePreselectionCutFlow['DQ'         ] = kDQ
kLoosePreselectionCutFlow['FrontPlanes'] = kFrontPlanes
kLoosePreselectionCutFlow['NHits'      ] = kNHits
kLoosePreselectionCutFlow['Fiducial'   ] = kNuebarCCIncFiducialLoose
kLoosePreselectionCutFlow['Containment'] = kNuebarCCIncContainmentLoose
kLoosePreselectionCutFlow['PhaseSpace' ] = kRecoElectronPhaseSpaceCut
"""


                        
    
        


        
    
    
    
    

