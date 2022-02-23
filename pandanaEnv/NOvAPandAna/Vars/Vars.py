import numpy as np
import pandas as pd
from pandana.core.var import Var

from Utils.index import KL

import numba as nb

kCVNe = Var(lambda tables: tables['rec.sel.cvnProd3Train']['nueid'])
kCVNm = Var(lambda tables: tables['rec.sel.cvnProd3Train']['numuid'])
kCVNnc = Var(lambda tables: tables['rec.sel.cvnProd3Train']['ncid'])

kNHit = Var(lambda tables: tables['rec.slc']['nhit'])

kRHC   = Var(lambda tables: tables['rec.spill']['isRHC'])
kDetID = Var(lambda tables: tables['rec.hdr']['det'])

# Containment vars
@nb.vectorize([nb.int32(nb.int32,nb.int32)], nopython=True, cache=True)
def calcFirstLivePlane(mask, fp):
    fd = fp//64
    dmin = fd

    for i in range(fd, -1, -1):
        temp = mask >> i
        if temp & 1 == 0:
            break
        else:
            dmin = i
    return 64*dmin

def planestofront(tables):
    mask = tables['rec.hdr']['dibmask']
    fp = tables['rec.slc']['firstplane']
    return fp - pd.Series(calcFirstLivePlane(mask.to_numpy(dtype=np.int32), fp.to_numpy(dtype=np.int32)), index=mask.index)
planestofront = Var(planestofront)

@nb.vectorize([nb.int32(nb.int32,nb.int32)], nopython=True, cache=True)
def calcLastLivePlane(mask, lp):
    ld = lp//64
    dmax = ld

    for i in range(ld, 14, 1):
        temp = mask >> i
        if temp & 1 == 0:
            break
        else:
            dmax = i
    return 64*(dmax+1)-1

def planestoback(tables):
    mask = tables['rec.hdr']['dibmask']
    lp = tables['rec.slc']['lastplane']
    return pd.Series(calcLastLivePlane(mask.to_numpy(dtype=np.int32), lp.to_numpy(dtype=np.int32)), index=mask.index) - lp
planestoback = Var(planestoback)

kDistAllTop    = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngtop'])
kDistAllBottom = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngbottom'])
kDistAllWest   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngwest'])
kDistAllEast   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngeast'])
kDistAllBack   = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngback'])
kDistAllFront  = Var(lambda tables: tables['rec.sel.nuecosrej']['distallpngfront'])
