from enum import Enum

# I want to be able to and these things together
# without writing it out each time
class MathEnum(Enum):
    def __and__(self, other):
        return self.value & other.value

class Flavor(MathEnum):
    kNuEToNuE    = 1<<0
    kNuEToNuMu   = 1<<1
    kNuEToNuTau  = 1<<2
    kNuMuToNuE   = 1<<3
    kNuMuToNuMu  = 1<<4
    kNuMuToNuTau = 1<<5

    kAllNuE   = kNuEToNuE | kNuMuToNuE
    kAllNuMu  = kNuEToNuMu | kNuMuToNuMu
    kAllNuTau = kNuEToNuTau | kNuMuToNuTau

    kAll = kAllNuE | kAllNuMu | kAllNuTau

class Current(MathEnum):
    kCC   = 1<<0
    kNC   = 1<<1
    kBoth = kCC | kNC

class Sign(MathEnum):
    kNu     = 1<<0
    kAntiNu = 1<<1
    kBoth   = kNu | kAntiNu

class PredictionExtrap:
    def __init__(self, extrap):
        self._extrap = extrap

    def Predict(self, calc):
        return self.PredictComponent(calc,
                                     Flavor.kAll,
                                     Current.kBoth,
                                     Sign.kBoth)

    def PredictComponent(self, calc, flav, curr, sign):
        SpecList = []

        if curr & Current.kCC:
            if sign & Sign.kNu:
                if flav & Flavor.kNuEToNuE:
                    SpecList.append(self._extrap.NueSurvComp(+1).Oscillate(calc, +12, +12))
                if flav & Flavor.kNuEToNuMu:
                    SpecList.append(self._extrap.NumuAppComp(+1).Oscillate(calc, +12, +14))
                if flav & Flavor.kNuEToNuTau:
                    SpecList.append(self._extrap.TauFromEComp(+1).Oscillate(calc, +12, +16))
                if flav & Flavor.kNuMuToNuE:
                    SpecList.append(self._extrap.NueAppComp(+1).Oscillate(calc, +14, +12))
                if flav & Flavor.kNuMuToNuMu:
                    SpecList.append(self._extrap.NumuSurvComp(+1).Oscillate(calc, +14, +14))
                if flav & Flavor.kNuMuToNuTau:
                    SpecList.append(self._extrap.TauFromMuComp(+1).Oscillate(calc, +14, +16))

            if sign & Sign.kAntiNu:
                if flav & Flavor.kNuEToNuE:
                    SpecList.append(self._extrap.NueSurvComp(-1).Oscillate(calc, -12, -12))
                if flav & Flavor.kNuEToNuMu:
                    SpecList.append(self._extrap.NumuAppComp(-1).Oscillate(calc, -12, -14))
                if flav & Flavor.kNuEToNuTau:
                    SpecList.append(self._extrap.TauFromEComp(-1).Oscillate(calc, -12, -16))
                if flav & Flavor.kNuMuToNuE:
                    SpecList.append(self._extrap.NueAppComp(-1).Oscillate(calc, -14, -12))
                if flav & Flavor.kNuMuToNuMu:
                    SpecList.append(self._extrap.NumuSurvComp(-1).Oscillate(calc, -14, -14))
                if flav & Flavor.kNuMuToNuTau:
                    SpecList.append(self._extrap.TauFromMuComp(-1).Oscillate(calc, -14, -16))
            
        if curr & Current.kNC:
            assert flav == Flavor.kAll # Can't calc anything else
            if sign & Sign.kNu:
                SpecList.append(self._extrap.NCComp(+1))
            if sign & Sign.kAntiNu:
                SpecList.append(self._extrap.NCComp(-1))

        # Sum everything up
        # TODO: Will SpecList ever be empty?
        ret = SpecList[0]
        for spec in SpecList[1:]:
            ret = ret + spec

        return ret
