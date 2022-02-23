from pandana import VarsToVarND
from Core.NOVASpectrum import NOVASpectrum,FilledNOVASpectrum
from Vars.TruthVars import kTrueE

class OscillatableSpectrum(NOVASpectrum):
    def __init__(self, loader, cut, var, weight=None):
        var = VarsToVarND([var, kTrueE])
        
        NOVASpectrum.__init__(self, loader, cut, var, weight=None)

    def Oscillate(self, calc, FlavBefore, FlavAfter):
        OscWeight = calc.P(FlavBefore, FlavAfter, self._df['E'])
        return FilledNOVASpectrum(self._df.drop(columns='E').squeeze(axis=1),
                                  self._weight*OscWeight, self._POT)
