from Core.NOVASpectrum import NOVASpectrum as NS
from Core.OscillatableSpectrum import OscillatableSpectrum as OS
from Cuts.TruthCuts import *

class TrivialExtrap:
    def __init__(self, loaderNonSwap, loaderFluxSwap, loaderTauSwap,
                 cut, var, weight=None):
        # Nue CC Component
        self._NueApp    = OS(loaderFluxSwap, cut & kIsSig       & kIsNu,     var, weight)
        self._ANueApp   = OS(loaderFluxSwap, cut & kIsSig       & kIsAntiNu, var, weight)
        self._NueSurv   = OS(loaderNonSwap,  cut & kIsBeamNue   & kIsNu,     var, weight)
        self._ANueSurv  = OS(loaderNonSwap,  cut & kIsBeamNue   & kIsAntiNu, var, weight)

        # Numu CC Component
        self._NumuSurv  = OS(loaderNonSwap,  cut & kIsNumuCC    & kIsNu,     var, weight)
        self._ANumuSurv = OS(loaderNonSwap,  cut & kIsNumuCC    & kIsAntiNu, var, weight)
        self._NumuApp   = OS(loaderFluxSwap, cut & kIsNumuApp   & kIsNu,     var, weight)
        self._ANumuApp  = OS(loaderFluxSwap, cut & kIsNumuApp   & kIsAntiNu, var, weight)

        # Nutau CC Component
        self._TauFromE  = OS(loaderTauSwap,  cut & kIsTauFromE  & kIsNu,     var, weight)
        self._ATauFromE = OS(loaderTauSwap,  cut & kIsTauFromE  & kIsAntiNu, var, weight)
        self._TauFromMu = OS(loaderTauSwap,  cut & kIsTauFromMu & kIsNu,     var, weight)
        self._ATauFromMu= OS(loaderTauSwap,  cut & kIsTauFromMu & kIsAntiNu, var, weight)

        # NC Component, can really load up on stats here since all files are valid
        self._NCN       = NS(loaderNonSwap,  cut & kIsNC        & kIsNu,     var, weight)
        self._ANCN      = NS(loaderNonSwap,  cut & kIsNC        & kIsAntiNu, var, weight)
        self._NCF       = NS(loaderFluxSwap, cut & kIsNC        & kIsNu,     var, weight)
        self._ANCF      = NS(loaderFluxSwap, cut & kIsNC        & kIsAntiNu, var, weight)
        self._NCT       = NS(loaderTauSwap,  cut & kIsNC        & kIsNu,     var, weight)
        self._ANCT      = NS(loaderTauSwap,  cut & kIsNC        & kIsAntiNu, var, weight)

        # CAFAna also includes the full NC component for reasons I wasn't able to follow
        # leave commented for now, til I understand why. It's just the sum of the above anyway.
        #self_NCTot      = OS(loaderNonSwap,  cut & kIsNC,                    var, weight),

    def NueAppComp(self, sign):
        return self._NueApp if sign>0 else self._ANueApp

    def NueSurvComp(self, sign):
        return self._NueSurv if sign>0 else self._ANueSurv

    def NumuSurvComp(self, sign):
        return self._NumuSurv if sign>0 else self._ANumuSurv

    def NumuAppComp(self, sign):
        return self._NumuApp if sign>0 else self._ANumuApp

    def TauFromEComp(self, sign):
        return self._TauFromE if sign>0 else self._ATauFromE

    def TauFromMuComp(self, sign):
        return self._TauFromMu if sign>0 else self._ATauFromMu

    def NCComp(self, sign):
        # Sum up NCs from each loader
        if sign>0:
            return self._NCN.hadd(self._NCF).hadd(self._NCT)
        return self._ANCN.hadd(self._ANCF).hadd(self._ANCT)
