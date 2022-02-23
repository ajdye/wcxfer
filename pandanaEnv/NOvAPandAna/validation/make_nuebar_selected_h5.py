import sys
import time
sys.path.append('/nova/app/users/ddoyle/git')

from NOvAPandAna.Core.NOVALoader import NOVALoader
from NOvAPandAna.Core.NOVASpectrum import NOVASpectrum
from NOvAPandAna.Vars.nuebarcc_vars import *
from NOvAPandAna.Cuts.nuebarcc_cuts import *
from NOvAPandAna.Vars.nueid import NueID

from mpi4py import MPI

import pandas as pd

from pandana.utils.pandasutils import VarsToVarND

import ROOT

def read_ttree_from(input_file):
    f = ROOT.TFile.Open(input_file)
    t = f.Get('selection')
    data, labels = t.AsMatrix(dtype='float', return_labels=True)
    df = pd.DataFrame(data, columns=labels)
    df = df.set_index(KL)
    return df

input_file_list = sys.argv[1]
compare_selected_input_file = sys.argv[2]

files = []
with open(input_file_list, 'r') as f:
    for l in f.readlines():
        files.append(l.strip())


def kNueIDInputs(tables):
    import pandas as pd
    from NOvAPandAna.Utils.index import KL
    epi0llt = tables['rec.vtx.elastic.fuzzyk.png.shwlid.lid']['epi0llt']
    shw = tables['rec.vtx.elastic.fuzzyk.png.shwlid'][['width', 'gap']]
    electronid = tables['rec.vtx.elastic.fuzzyk.png.spprongcvnpart5label']['electronid']

    df = pd.concat([epi0llt, shw, electronid], axis=1, join='inner')
    # We sort so the row with highest electronid is first in each group
    # The result will be automaticaaly sorted back to the original order
    df = df.sort_values('electronid', ascending=False).groupby(KL).first()
    return df

weights_file = '/nova/app/users/ddoyle/development/NDAna/Classifiers/NueID/NueID.weights.xml'
NueIDBDT = NueID(weights_file)
kVar = VarsToVarND([NueIDBDT.Var(), kNueIDInputs])
kCut = kDecafPreselection


compare_selected = read_ttree_from(compare_selected_input_file)
# get appropriate binning before any timing measurements
bins = {col: np.histogram(compare_selected[col].values, bins=30)[1] for col in compare_selected.columns}


l = NOVALoader(files)
snueid = NOVASpectrum(l, kCut, kVar)

start_go = time.perf_counter_ns()
l.Go()
end_go = time.perf_counter_ns()

n = {}
for col in snueid._df.columns:
    n[col], _ = np.histogram(snueid._df[col], bins=bins[col])
    n[col] = MPI.COMM_WORLD.reduce(n[col], op=MPI.SUM, root=0)

end_reduce = time.perf_counter_ns()

print('[%d]' % MPI.COMM_WORLD.Get_rank(), end_go - start_go, 'ns Go()')
print('[%d]' % MPI.COMM_WORLD.Get_rank(), end_reduce - start_go, 'ns Go() + reduce')


acc_diff = {}
for col in snueid._df.columns:
    acc_diff[col] = (snueid._df[col].astype(float) - compare_selected[col].astype(float)).sum()
length = snueid._df.shape[0]

comm = MPI.COMM_WORLD
acc_diffs = comm.gather(acc_diff, root=0)
lengths = comm.gather(length, root=0)

if MPI.COMM_WORLD.Get_rank() == 0:
    import matplotlib.pyplot as plt

    for col in snueid._df.columns:
        col_diff = np.concatenate(np.atleast_2d([d[col] for d in acc_diffs]))
        diff = sum(col_diff) / sum(lengths)
        print('[%d] Diff %s = ' % (MPI.COMM_WORLD.Get_rank(), col), diff)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            compare_selected[col].values,
            bins=bins[col],
            color='k', 
            label='CAFAna %d' % compare_selected.shape[0],
            histtype='step',
            lw=2
        )

        ax.hist(
            bins[col][:-1],
            bins=bins[col],
            weights=n[col],
            color='r',
            ls='--',
            label='PandAna %d' % n[col].sum(),
            histtype='step',
            lw=2
        )
    
        plt.legend()
        ax.set_title('Average difference per event = %1.3g' % diff)
        ax.set_xlim
        ax.set_xlabel(col)

        if MPI.COMM_WORLD.Get_size() > 1:
            plt.savefig('compare_selected_%s_mpi.pdf' % col)
        else:
            plt.savefig('compare_selected_%s.pdf' % col)

