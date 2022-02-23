import array
import numpy as np
from NOvAPandAna.Vars.tmva import TMVABDT

class NueID(TMVABDT):
    def __init__(self, bdt_file_name=None):
        import sys
        import os
        super().__init__()
        self.shwwidth   = array.array('f', [0])
        self.epi0llt    = array.array('f', [0])
        self.electronid = array.array('f', [0])
        self.shwgap     = array.array('f', [0])

        self.reader.AddVariable('shwwidth'  , self.shwwidth)
        self.reader.AddVariable('epi0llt'   , self.epi0llt)
        self.reader.AddVariable('electronid', self.electronid)
        self.reader.AddVariable('shwgap'    , self.shwgap)


        if bdt_file_name is None:
            srt_public = os.getenv('SRT_PUBLIC_CONTEXT')
            srt_private = os.getenv('SRT_PRIVATE_CONTEXT')
            print(srt_public)
            print(srt_private)
            weightfile_loc = 'NDAna/Classifiers/NueID/NueID.weights.xml'
            if os.path.isfile(os.path.join(srt_private, weightfile_loc)):
                bdt_file_name = os.path.join(srt_private, weightfile_loc)
            elif os.path.isfile(os.path.join(srt_public, weightfile_loc)):
                bdt_file_name = os.path.join(srt_public, weightfile_loc)
            else:
                bdt_file_name = weightfile_loc
        if not os.path.isfile(bdt_file_name):
            print('Cannot find %s' % bdt_file_name)
            sys.exit(1)

        self.reader.BookMVA('NueID', bdt_file_name)

#        self.Eval = np.vectorize(
#            self._eval,
#            otypes=[float],
#        )

    def Eval(self, 
             vshwwidth, 
             vepi0llt,
             velectronid,
             vshwgap):
        return np.array(
            [
                self._eval(a, b, c, d)
                for a, b, c, d in zip(
                        vshwwidth, 
                        vepi0llt,
                        velectronid,
                        vshwgap,
                )
            ]
        )


    def __call__(self, tables):    
        import pandas as pd
        from NOvAPandAna.Utils.index import KL
        epi0llt = tables['rec.vtx.elastic.fuzzyk.png.shwlid.lid']['epi0llt']
        shw = tables['rec.vtx.elastic.fuzzyk.png.shwlid'][['width', 'gap']]
        electronid = tables['rec.vtx.elastic.fuzzyk.png.spprongcvnpart5label']['electronid']

        df = pd.concat([epi0llt, shw, electronid], axis=1, join='inner')
        # We sort so the row with highest electronid is first in each group
        # The result will be automaticaaly sorted back to the original order
        df = df.sort_values('electronid', ascending=False).groupby(KL).first()

        # This will be a TMVA evaluation using the above columns
        # Just take product for now
        values = df[['width', 'epi0llt', 'electronid', 'gap']].values
        return pd.Series(self.Eval(values[:,0], values[:,1], values[:,2], values[:,3]),
                         index=df.index,
                         name='NueID')

    def _eval(self, shwwidth, epi0llt, electronid, shwgap):
        self.shwwidth[0] = shwwidth
        self.epi0llt[0] = epi0llt
        self.electronid[0] = electronid
        self.shwgap[0] = shwgap

        return self.reader.EvaluateMVA('NueID')

    def Var(self):
        from pandana import Var
        return Var(self.__call__)
