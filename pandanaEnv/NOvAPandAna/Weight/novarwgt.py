import pandas as pd
import numpy as np
import h5py
import os

from pandana.utils import *
from Utils.enums import *
from Utils.index import KL

class LorentzVectorDF:
    def __init__(self, x, y, z, t):
        self.df = pd.concat([x, y, z, t], axis=1, join='inner')
        self.df.columns = ['x', 'y', 'z', 't']

        # caches
        self.gamma = None
        self.beta  = None
        self.rho   = None

    @property
    def xvec(self):
        return self.df[['x','y','z']]

    @xvec.setter
    def xvec(self, val):
        self.df[['x','y','z']] = val
        
    def Gamma(self):
        if self.gamma is None: 
            self.gamma =  1.0 / (1 - self.Beta().pow(2)).pow(1./2)
        return self.gamma

    def Beta(self):
        if self.beta is None:
            self.beta = self.Rho() / self.df['t']
        return self.beta

    def Rho(self):
        if self.rho is None:
            self.rho = self.xvec.pow(2).sum(axis=1).pow(1./2)
        return self.rho
    
    def SetRho(self, rho):
        factor = rho / self.Rho()
        self.xvec = self.xvec.multiply(factor, axis=0)
        
        # magnitude is expensive so cache it
        self.rho = rho
        
        # clear other caches
        self.gamma = None
        self.beta  = None
        
    def E(self):
        return self.df['t']
    
class EventRecordDF:
    def __init__(self, tables):
        self.generatorVersion = tables['rec.mc.nu.genVersion']['value'].groupby(KL).apply(np.stack)
        # TODO genConfigString?
        
        nu    = tables['rec.mc.nu'     ]
        prims = tables['rec.mc.nu.prim']
        
        self.nupdg      = nu['pdg'   ]
        self.isCC       = nu['iscc'  ]

        # TODO rgwt::reaction
        self.reaction   = nu['mode'  ]

        self.struckNucl = nu['hitnuc']
        self.Enu        = nu['E'     ]

        
        self.q = LorentzVectorDF(pd.Series(1, index=self.Enu.index, name='x'),
                                 pd.Series(0, index=self.Enu.index, name='y'),
                                 pd.Series(0, index=self.Enu.index, name='z'),
                                 nu['y'] * nu['E'])
        self.q.SetRho((nu['q2'] + self.q.E().pow(2)).pow(1./2))
        self.y = nu['y']
        self.W = nu['W2'].pow(1./2)
        self.A = nu['tgtA']
        
        self.npiplus = nu['npiplus']
        self.npizero = nu['npizero']
        self.npiminus = nu['npiminus']
    
        prims_4p = LorentzVectorDF(prims['p.px'],
                                   prims['p.py'],
                                   prims['p.pz'],
                                   prims['p.E' ])

        ke = pd.Series((1 - 1./prims_4p.Gamma()), name='KE')
        pdgKE = pd.concat([prims['pdg'], ke], axis=1)
        
        # this handy function does exactly what we want it to. Serendipty
        self.fsPartKE = pd.pivot_table(pdgKE,
                                       columns='pdg',
                                       values='KE',
                                       index=KL,
                                       aggfunc=np.sum)
        
        self.fsPartMult = pd.pivot_table(pdgKE,
                                         columns='pdg',
                                         values='KE',
                                         index=KL,
                                         aggfunc=len,
                                         fill_value=0)

        self.expectNoWeights = ~nu['isvtxcont']
        
        

        
        
        
