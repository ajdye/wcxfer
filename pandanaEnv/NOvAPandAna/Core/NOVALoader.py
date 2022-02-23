from pandana.core.loader import Loader
from pandana.core.spectrum import Spectrum

from NOvAPandAna.Core.source import GetFileList
from NOvAPandAna.Utils.index import index
from NOvAPandAna.Vars.SpillVars import kSpillPOT
from NOvAPandAna.Cuts.SpillCuts import kNoSpillCut

class NOVALoader(Loader):
    def __init__(self, query, SpillCut = None, offset = 0, stride = 1, limit = None):
        files = GetFileList(query, offset, stride, limit)

        Loader.__init__(self, files, idcol='evt.seq', main_table_name='spill', indices=index)

        self.SpillCut = SpillCut
        self._POTSpectrum = Spectrum(self,
                                     kNoSpillCut if self.SpillCut is None else self.SpillCut,
                                     kSpillPOT)
        self.POT = 0

    def Finish(self):
        self._POTSpectrum.finish()
        self.POT = self._POTSpectrum.df().sum()
        self._specdefs.remove(self._POTSpectrum)

        Loader.Finish(self)
