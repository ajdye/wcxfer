#######################################################
#
# Test for the neutrino oscillation selections
# Written to run with:
# prod_sumdecaf_R17-11-14-prod4reco.d_fd_genie_nonswap_fhc_nova_v08_period5_v1_numu2018.h5
# and two text files from the cafana output
#
#######################################################

import sys
from time import time as now

from pandana import *
from Core.NOVALoader import NOVALoader
from Core.NOVASpectrum import NOVASpectrum
from Utils.index import index
from Cuts.NueCuts import *
from Cuts.NumuCuts import *
from Cuts.SpillCuts import kStandardSpillCuts
from Weight.cv_weights_2017 import kCVWgt2018

# Create the tables
f = sys.argv[1]
tables = NOVALoader(f, SpillCut=kStandardSpillCuts)

# Create Spectra for Numu and Nue
SpecNue  = NOVASpectrum(tables, kNueFD, kNueEnergy, kCVWgt2018)
SpecNumu = NOVASpectrum(tables, kNumuFD, kNumuE, kCVWgt2018)

# Let's do it!
start = now()
tables.Go()
stop = now()

print('finished in',stop-start)

print(SpecNue.df())
print(SpecNue.weight())

assert SpecNue.POT() == SpecNumu.POT()
print('Analyzed',SpecNue.POT(),'POT') # Should be 3.17541e22


print('Selected Nue Events:',SpecNue.entries())
print('Selected Numu Events:',SpecNumu.entries())

# Read CAFAna results from txt files
NueCAF = pd.read_csv('nue2018_selected.txt', sep=' ', index_col=False,
                      names=['run','subrun','cycle','evt','subevt','E','wgt'])
NueCAF.set_index(['run','subrun','cycle','evt','subevt'],inplace=True)
NueCAFDF = NueCAF['E']
NueCAFWgt = NueCAF['wgt']
NNueCAF = NueCAFDF.shape[0]
print('Selected Nue in CAF:',NNueCAF)

NumuCAF = pd.read_csv('numu2018_selected.txt', sep=' ', index_col=False,
                      names=['run','subrun','cycle','evt','subevt','E','wgt'])
NumuCAF.set_index(['run','subrun','cycle','evt','subevt'],inplace=True)
NumuCAFDF = NumuCAF['E']
NumuCAFWgt = NumuCAF['wgt']
NNumuCAF = NumuCAFDF.shape[0]
print('Selected Numu in CAF:',NNumuCAF)

names = ['numudf', 'numuwgt', 'nuedf', 'nuewgt']
PdRets = [SpecNumu.df(), SpecNumu.weight(), SpecNue.df(), SpecNue.weight()]
CafRets = [NumuCAFDF, NumuCAFWgt, NueCAFDF, NueCAFWgt]

for name, pdret, cafret in zip(names, PdRets, CafRets):
    assert pdret.shape[0] == cafret.shape[0]
    diff = (pdret - cafret).abs()
    idmax = diff.idxmax()

    print(name)
    print('Max separation for row:',idmax)
    print('Difference:',diff[idmax])
    print('PandAna Value:',pdret[idmax])
    print('CAFAna Value:',cafret[idmax])


