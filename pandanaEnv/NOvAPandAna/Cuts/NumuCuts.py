import pandas as pd
import numpy as np

from pandana.core import Cut

from Utils.misc import *
from Utils.index import KL
from Vars.Vars import *
from Vars.NumuVars import *
from Cuts.Cuts import kIsFD

def kNumuBasicQuality(tables):
    df_numutrkcce=tables['rec.energy.numu']['trkccE']
    df_remid=tables['rec.sel.remid']['pid']
    df_nhit=tables['rec.slc']['nhit']
    df_ncontplanes=tables['rec.slc']['ncontplanes']
    df_cosmicntracks=tables['rec.trk.cosmic']['ntracks']
    return(df_numutrkcce > 0) &\
               (df_remid > 0) &\
               (df_nhit > 20) &\
               (df_ncontplanes > 4) &\
               (df_cosmicntracks > 0)
kNumuBasicQuality = Cut(kNumuBasicQuality)

kNumuQuality = kNumuBasicQuality & (kCCE < 5.)

# FD 

kNumuProngsContainFD = (kDistAllTop > 60) & (kDistAllBottom > 12) & (kDistAllEast > 16) & \
                            (kDistAllWest > 12)  & (kDistAllFront > 18) & (kDistAllBack > 18)

def kNumuOptimizedContainFD(tables):
    ptf = planestofront(tables) > 1
    ptb = planestoback(tables) > 1

    df_containkalfwdcell = tables['rec.sel.contain']['kalfwdcell'] > 6
    df_containkalbakcell = tables['rec.sel.contain']['kalbakcell'] > 6
    df_containcosfwdcell = tables['rec.sel.contain']['cosfwdcell'] > 0 
    df_containcosbakcell = tables['rec.sel.contain']['cosbakcell'] > 7

    return ptf & ptb & df_containkalfwdcell & df_containkalbakcell & \
        df_containcosfwdcell & df_containcosbakcell
kNumuOptimizedContainFD = Cut(kNumuOptimizedContainFD)

kNumuContainFD = kNumuProngsContainFD & kNumuOptimizedContainFD 

kNumuNoPIDFD = kNumuQuality & kNumuContainFD

def kNumuPID(tables):
    return (tables['rec.sel.remid']['pid'] > 0.7) & \
        (tables['rec.sel.cvn2017']['numuid'] > 0.1) & \
        (tables['rec.sel.cvnProd3Train']['numuid'] > 0.7)
kNumuPID = Cut(kNumuPID)

def kNumuCosRej(tables):
    return (tables['rec.sel.cosrej']['anglekal'] > 0.5) & \
        (tables['rec.sel.cosrej']['numucontpid'] > 0.53) & \
        (tables['rec.slc']['nhit'] < 400) & \
        (tables['rec.sel.nuecosrej']['pngptp'] < 0.9)
kNumuCosRej = Cut(kNumuCosRej)

kNumuFD = kNumuPID & kNumuCosRej & kNumuNoPIDFD

# ND
def kNumuContainND(tables):
    shw_df = tables['rec.vtx.elastic.fuzzyk.png.shwlid']
    shw_df_trans = shw_df[['start.y','stop.y', 'start.x', 'stop.x']]
    shw_df_long = shw_df[['start.z', 'stop.z']]
    no_shw = (tables['rec.vtx.elastic.fuzzyk']['nshwlid'] == 0).groupby(level=KL).first()

    shw_contain = ((shw_df_trans.min(axis=1) >= -180.) & (shw_df_trans.max(axis=1) <= 180.) & \
             (shw_df_long.min(axis=1) >= 20.) & (shw_df_long.max(axis=1) <= 1525.)).groupby(level=KL).agg(np.all)
    shw_contain = (shw_contain | no_shw)

    trk_df = tables['rec.trk.kalman.tracks'][['start.z', 'stop.z']]
    trk_df = trk_df.reset_index().set_index(KL)
    kalman_contain = (trk_df['rec.trk.kalman.tracks_idx'] == 0) | ((trk_df['start.z'] <= 1275) & (trk_df['stop.z'] <= 1275))
    kalman_contain = kalman_contain.groupby(level=KL).agg(np.all)

    df_ntracks = tables['rec.trk.kalman']['ntracks']
    df_remid = tables['rec.trk.kalman']['idxremid']
    df_firstplane = tables['rec.slc']['firstplane']
    df_lastplane = tables['rec.slc']['lastplane']
    
    df_startz = trk_df['start.z'].groupby(level=KL).first()
    df_stopz  = trk_df['stop.z'].groupby(level=KL).first()
    
    df_containkalposttrans = tables['rec.sel.contain']['kalyposattrans']
    df_containkalfwdcellnd = tables['rec.sel.contain']['kalfwdcellnd']
    df_containkalbakcellnd = tables['rec.sel.contain']['kalbakcellnd']
    
    return (df_ntracks > df_remid) &\
           (df_firstplane > 1) &\
           (df_lastplane < 212) &\
           (df_containkalfwdcellnd > 5) &\
           (df_containkalbakcellnd > 10) &\
           (df_startz < 1100 ) & (( df_containkalposttrans < 55) | (df_stopz < 1275) ) &\
           shw_contain &\
           kalman_contain
kNumuContainND = Cut(kNumuContainND)

kNumuNCRej = Cut(lambda tables: tables['rec.sel.remid']['pid'] > 0.75)

kNumuNoPIDND = kNumuQuality & kNumuContainND

kNumuCutND = kNumuQuality & kNumuContainND & kNumuPID

#---2020 Cuts---#

def kNumu2020CosRejLoose(tables):
    return((tables['rec.sel.cosrej']['numucontpid2020'] > 0.4) &\
           (tables['rec.sel.cvnloosepreselptp']['numuid'] > 0.))
kNumu2020CosRejLoose = Cut(kNumu2020CosRejLoose)

def kNumu2020CosRej(tables):
    return(tables['rec.sel.cosrej']['numucontpid2020'] > 0.45)
kNumu2020CosRej = Cut(kNumu2020CosRej)

def kNumu2020PID(tables):
    return((tables['rec.sel.remid']['pid'] > 0.3) &\
           (tables['rec.sel.cvnloosepreselptp']['numuid'] > 0.8))
kNumu2020PID = Cut(kNumu2020PID)

def kNumuBaseContainFD2020(tables):
    ptf = planestofront(tables) > 2
    ptb = planestoback(tables) > 3

    df_containkalfwdcell = tables['rec.sel.contain']['kalfwdcell'] > 6
    df_containkalbakcell = tables['rec.sel.contain']['kalbakcell'] > 6
    df_containcosfwdcell = tables['rec.sel.contain']['cosfwdcell'] > 5
    df_containcosbakcell = tables['rec.sel.contain']['cosbakcell'] > 7
    return ptf & ptb & df_containkalfwdcell & df_containkalbakcell & \
           df_containcosfwdcell & df_containcosbakcell
kNumuBaseContainFD2020 = Cut(kNumuBaseContainFD2020)

def kCosRejVeto(tables):
    return (tables['rec.sel.veto']['keep'] == 1)
kCosRejVeto = Cut(kCosRejVeto) & kIsFD

def kCNNVeto(tables):
    return(tables['rec.spill.cosmiccvn']['passSel'] == 1)
kCNNVeto = Cut(kCNNVeto) & kIsFD

k3flavor2020FDVeto = kCosRejVeto & kCNNVeto

kNumuContainFD2020 = kNumuBaseContainFD2020 & kNumuProngsContainFD

kNumuNoPID2020FD = kNumuQuality & kNumuContainFD2020

kNumuNoCosRej2020FD = kNumuQuality & kNumuContainFD2020 & kNumu2020PID

kNumuNoVeto2020FD = kNumuQuality & kNumuContainFD2020 & kNumu2020PID & kNumu2020CosRej

kNumu2020FD = kNumuQuality & kNumuContainFD2020 & kNumu2020PID & kNumu2020CosRej & k3flavor2020FDVeto
