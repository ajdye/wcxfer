import numpy as np
import pandas as pd
import numba as nb

from pandana.core.cut import Cut
from NOvAPandAna.Vars.SpillVars import *
from NOvAPandAna.Utils.enums import detector
from NOvAPandAna.Utils.misc import GetPeriod

# Need this in NOvA Loader when not using a cut
kNoSpillCut = kSpillPOT > 0

kSpillIsND = kSpillDet == detector.kND
kSpillIsFD = kSpillDet == detector.kFD
kSpillIsMC = Cut(lambda tables: tables['spill']['ismc'] == 1)

# Good Spills
kEdgeMatch0 = kEdgeMatch == 0
kFracDCMHits0 = kFracDCMHits == 0
kMissDCMCut = kNMissDCM == 0
kMissDCMLGCut = kNMissDCMLG == 0
kIsGoodSpill = Cut(lambda tables: tables['spill']['isgoodspill'] == 1)
kGoodSpill = ~kEdgeMatch0 | ~kFracDCMHits0 | ~kMissDCMLGCut | kIsGoodSpill

# Remove events that were labeled as incomplete by the DAQ
kComplete = Cut(lambda tables: tables['spill']['eventincomplete'] == 0)

# Want mostly complete detectors. ND specific cut for when lights are left on
kFracDCMHitsCutND = kFracDCMHits <= 0.45
kMissingDCM = (kSpillIsND & kFracDCMHitsCutND & kMissDCMCut) | (kSpillIsFD & kMissDCMLGCut)

# FD specific cut to remove events where the detector is likely out of sync
kEdgeMatchCut = kEdgeMatch > 0.2
kOutofSync = ~kSpillIsFD | kSpillIsMC | kEdgeMatchCut

kStandardDQCuts = kGoodSpill & kComplete & kMissingDCM & kOutofSync

# Cuts on the pos/width/etc of the beam
kBeamWidthCut = (kWidthY >= 0.57) & (kWidthY <= 1.58) & (kWidthX >= 0.57) & \
    (((kWidthX <= 1.58) & (kSpillPeriod < 10)) | ((kWidthX <= 1.88) & (kSpillPeriod >= 10)))

kTightBeamQualityCuts = kSpillIsMC | (kTrigger == 2) | \
    ((kDeltaSpillTime == 0) & (kSpillTime == 0) & (kWidthX == 0) & kIsGoodSpill) | \
    ((kDeltaSpillTime <= 0.5e9) & (kSpillPOT >= 2e12) & (kHornI >= -202) & (kHornI <= -198) & \
     (kPosX <= 2) & (kPosY <= 2) & kBeamWidthCut)

# We want events with at least 4 contiguous diblocks active
@nb.vectorize([nb.boolean(nb.int32)], nopython=True, cache=True)
def kContiguousDibs(mask):
    count = 0
    for i in range(14):
        temp = mask >> i
        if temp & 1 == 1:
            count += 1
            if count >= 4:
                return True
        else:
            count = 0
    return False

def kRemoveSmallMasks(tables):
    mask = tables['spill']['dibmask']
    return pd.Series(kContiguousDibs(mask.to_numpy(dtype=np.int32)), mask.index)
kRemoveSmallMasks = kSpillIsND | Cut(kRemoveSmallMasks)

# Use this unless you're doing something non-standard
kStandardSpillCuts = kStandardDQCuts & kTightBeamQualityCuts & kRemoveSmallMasks
