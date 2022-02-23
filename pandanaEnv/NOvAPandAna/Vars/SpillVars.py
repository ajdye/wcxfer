import pandas as pd

from pandana.core.var import Var
from NOvAPandAna.Utils.misc import GetPeriod

kSpillPOT = Var(lambda tables: tables['spill']['spillpot'])

kSpillDet = Var(lambda tables: tables['spill']['det'])
kTrigger = Var(lambda tables: tables['spill']['trigger'])

kEdgeMatch = Var(lambda tables: tables['spill']['dcmedgematchfrac'])
kFracDCMHits = Var(lambda tables: tables['spill']['fracdcm3hits'])
kNMissDCM = Var(lambda tables: tables['spill']['nmissingdcms'])
kNMissDCMLG = Var(lambda tables: tables['spill']['nmissingdcmslg'])

kPosX = Var(lambda tables: tables['spill']['posx'].abs())
kPosY = Var(lambda tables: tables['spill']['posy'].abs())
kWidthX = Var(lambda tables: tables['spill']['widthx'])
kWidthY = Var(lambda tables: tables['spill']['widthy'])

kHornI = Var(lambda tables: tables['spill']['hornI'])

kSpillTime = Var(lambda tables: tables['spill']['spilltimesec'])
kDeltaSpillTime = Var(lambda tables: tables['spill']['deltaspilltimensec'].abs())

def kSpillPeriod(tables):
    det = kSpillDet(tables)
    runs = det.index.get_level_values('run')
    return pd.Series(GetPeriod(runs, det), det.index)
kSpillPeriod = Var(kSpillPeriod)
