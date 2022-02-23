import pandas as pd

from pandana.core.spectrum import Spectrum
from mpi4py import MPI


class NOVASpectrum(Spectrum):
    def __init__(self, loader, cut, var, weight=None):
        self._loader = loader

        if self._loader.SpillCut is not None:
            cut = loader.SpillCut & cut

        self._POT = 0

        Spectrum.__init__(self, loader, cut, var, weight)

    def finish(self):
        Spectrum.finish(self)
        self._POT = self._loader.POT

    def POT(self):
        return self._POT

    def histogram(self, bins, range=None, POT=None, mpireduce=False, root=0):
        n, bins = Spectrum.histogram(self, bins, range, mpireduce, root)
        # No scaling needed
        if POT is None:
            return n, bins

        if mpireduce:
            # Reduce the total POT across all ranks
            tot_POT = MPI.COMM_WORLD.reduce(self._POT, op=MPI.SUM, root=root)
        else:
            # Otherwise use the spectrum POT
            tot_POT = self._POT

        # Only the root rank has data at this point
        if MPI.COMM_WORLD.rank != root:
            return None, bins
        return n * (POT / tot_POT), bins

    def __add__(self, other):
        df = pd.concat([self._df, other._df])

        # Weight other POT to match self POT
        wgt = pd.concat([self._weight, other._weight * (self._POT / other._POT)])

        return FilledNOVASpectrum(df, wgt, self._POT)

    # Add first ask questions later
    def hadd(self, other):
        df = pd.concat([self._df, other._df])
        wgt = pd.concat([self._weight, other._weight])
        pot = self._POT + other._POT

        return FilledNOVASpectrum(df, wgt, pot)


class FilledNOVASpectrum(NOVASpectrum):
    def __init__(self, df, wgt, POT):
        self._df = df
        self._weight = wgt
        self._POT = POT
