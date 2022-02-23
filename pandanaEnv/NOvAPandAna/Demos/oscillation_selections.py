import sys
from time import time as now

from pandana.core import *
from Utils.index import index
from Cuts.NueCuts import *
from Cuts.NumuCuts import *
from Vars.NueVars import *
from Vars.NumuVars import *

# Create loader from an h5 file
f = sys.argv[1]
tables = Loader(f, idcol='evt.seq', main_table_name='spill', indices=index)

# Create Spectra
# FD
SpecNue  = Spectrum(tables, kNueFD, kNueEnergy)
SpecNumu = Spectrum(tables, kNumuFD, kNumuE)

# Uncomment for ND
#SpecNue  = Spectrum(tables, kNueNDCVNSsb, kNueEnergy)
#SpecNumu = Spectrum(tables, kNumuCutND, kNumuE)

# Let's do it!
start = now()
tables.Go()
stop = now()

print('Finished in',stop-start,'s')

print('Selected Nue Events:',SpecNue.entries())
print('Selected Numu Events:',SpecNumu.entries())
