from pandana import Var
from Utils.index import KL

kTrueE = Var(lambda tables: tables['rec.mc.nu']['E'].groupby(level=KL).first())
