from pandana.core import Cut

from Utils.misc import *
from Vars.Vars import *

kIsND = kDetID == detector.kND
kIsFD = kDetID == detector.kFD

# Basic cosrej
kVeto = Cut(lambda tables: tables['rec.sel.veto']['keep'] == 1)

# Basic Reco Cuts
kHasVtx  = Cut(lambda tables: tables['rec.vtx']['nelastic'] > 0)
kHasPng  = Cut(lambda tables: (tables['rec.vtx.elastic.fuzzyk']['npng'] > 0).groupby(level=KL).agg(np.all))
