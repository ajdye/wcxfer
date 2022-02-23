from pandana.core import Cut
from Utils.index import KL

def CCFlavSel(StartFlav, EndFlav):
    def kFlavCut(tables):
        nu = tables['rec.mc.nu']
        return ((nu['iscc'] == 1) & \
            (nu['pdgorig'].abs() == StartFlav) & \
            (nu['pdg'].abs() == EndFlav)).groupby(level=KL).first()
    return Cut(kFlavCut)

kIsSig       = CCFlavSel(14, 12)
kIsNumuCC    = CCFlavSel(14, 14)
kIsBeamNue   = CCFlavSel(12, 12)
kIsNumuApp   = CCFlavSel(12, 14)
kIsTauFromMu = CCFlavSel(14, 16)
kIsTauFromE  = CCFlavSel(12, 16)

kIsNC = Cut(lambda tables: (tables['rec.mc.nu']['iscc'] == 0).groupby(level=KL).first())

kIsNu     = Cut(lambda tables: (tables['rec.mc.nu']['pdg'] > 0).groupby(level=KL).first())
kIsAntiNu = Cut(lambda tables: (tables['rec.mc.nu']['pdg'] < 0).groupby(level=KL).first())
