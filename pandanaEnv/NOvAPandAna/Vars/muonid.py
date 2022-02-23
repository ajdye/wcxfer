import array
import numpy as np

from NOvAPandAna.Vars.tmva import TMVABDT

class MuonID(TMVABDT):
    def __init__(self, bdt_file_name = None):
    
        super().__init__()
        self.dedxll   = array.array('f', [0])
        self.scatll   = array.array('f', [0])
        self.dedx10cm = array.array('f', [0])
        self.dedx40cm = array.array('f', [0])

        self.reader.AddVariable('DedxLL', self.dedxll)
        self.reader.AddVariable('ScatLL', self.scatll)
        self.reader.AddVariable('Avededxlast10cm', self.dedx10cm)
        self.reader.AddVariable('Avededxlast40cm', self.dedx40cm)

        self.reader.BookMVA('BDTG', bdt_file_name)
    
    def Eval(self, 
             vdedxll, 
             vscatll,
             vdedx10cm,
             vdedx40cm):
        return np.array(
            [
                self._eval(a, b, c, d)
                for a, b, c, d in zip(
                        vdedxll,
                        vscatll,
                        vdedx10cm,
                        vdedx40cm
                )
            ]
        )
    def _eval(self, 
              dedxll, 
              scatll,
              dedx10cm,
              dedx40cm):
        self.dedxll[0] = dedxll
        self.scatll[0] = scatll
        self.dedx10cm[0] = dedx10cm
        self.dedx40cm[0] = dedx40cm

        return self.reader.EvaluateMVA('BDTG')
