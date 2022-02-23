import numpy as np

class OscCalc:
    def __init__(self,
                 dmsq21 = 7.53e-5, # eV^2
                 dmsq32 = 2.45e-3, # eV^2
                 th12   = 0.58725, # sin^2th12 = 0.307
                 th13   = 0.14543, # sin^2th13 = 0.021
                 th23   = 0.7854,  # pi/4
                 dcp    = 0):
        self._Dmsq21 = dmsq21 # eV^2
        self._Dmsq32 = dmsq32 # eV^2
        
        self._Th12 = th12 # sin^2th12 = 0.307
        self._Th13 = th13 # sin^2th13 = 0.021
        self._Th23 = th23 # pi/4
        
        self._dCP = dcp

        # Put some default parameters here
        self._Rho = 2.84 # g/cm^3
        self._L = 810 # km

        # These don't change
        self._hbar_c_eV_km = 1.97326938E-10 # eV-km
        self._eVPerGeV = 1E9

        self._Na = 6.0221409e23 # mol^-1
        self._ZperA = 0.5 # e- per nucleon
        self._G_F = 1.16637E-23 # eV^-2
        self._hbar_c_eV_cm = 1.97326938E-5 # eV-cm

        self._updated = False

    def SetL(self, L):
        self._L = L
        self._updated = False

    def SetRho(self, rho):
        self._Rho = rho
        self._updated = False

    def SetDmsq32(self, dmsq32):
        self._Dmsq32 = dmsq32
        self._updated = False

    def SetDmsq21(self, dmsq21):
        self._Dmsq21 = dmsq21
        self._updated = False

    def SetTh12(self, th12):
        self._Th12 = th12
        self._updated = False

    def SetTh13(self, th13):
        self._Th13 = th13
        self._updated = False

    def SetTh23(self, th23):
        self._Th23 = th23
        self._updated = False

    def SetdCP(self, dcp):
        self._dCP = dcp
        self._updated = False

    def Update(self):
        if self._updated: return
        
        self._Dmsq31 = self._Dmsq21 + self._Dmsq32
        self._Alpha = self._Dmsq21 / self._Dmsq31
        self._sin_th12 = np.sin(self._Th12)
        self._sin_th13 = np.sin(self._Th13)
        self._sin_th23 = np.sin(self._Th23)
        self._cos_th12 = np.cos(self._Th12)
        self._cos_th13 = np.cos(self._Th13)
        self._cos_th23 = np.cos(self._Th23)
        self._sin_2th12 = np.sin(2*self._Th12)
        self._sin_2th13 = np.sin(2*self._Th13)
        self._sin_2th23 = np.sin(2*self._Th23)
        self._cos_2th12 = np.cos(2*self._Th12)
        self._cos_2th13 = np.cos(2*self._Th13)
        self._cos_2th23 = np.cos(2*self._Th23)
        self._sin_sq_th12 = self._sin_th12**2
        self._sin_sq_th13 = self._sin_th13**2
        self._sin_sq_th23 = self._sin_th23**2
        self._cos_sq_th12 = self._cos_th12**2
        self._cos_sq_th13 = self._cos_th13**2
        self._cos_sq_th23 = self._cos_th23**2
        self._sin_sq_2th12 = self._sin_2th12**2
        self._sin_sq_2th13 = self._sin_2th13**2
        self._sin_sq_2th23 = self._sin_2th23**2
        self._cos_sq_2th12 = self._cos_2th12**2
        self._cos_sq_2th13 = self._cos_2th13**2
        self._cos_sq_2th23 = self._cos_2th23**2

        self._V = np.sqrt(2)*self._G_F*self._Rho*self._ZperA*self._Na*self._hbar_c_eV_cm**3

        self._updated = True

    def UpdateEDep(self, E, antinu, fliptime):
        s = -1 if antinu else 1
        t = -1 if fliptime else 1

        self._A = s*2*self._V*E*self._eVPerGeV/self._Dmsq31
        self._D = self._Dmsq31*self._L/(4*E*self._eVPerGeV*self._hbar_c_eV_km)

        self._dCPproxy = s*t*self._dCP
        self._sin_dCPproxy = np.sin(self._dCPproxy)
        self._cos_dCPproxy = np.cos(self._dCPproxy)

        self._C12 = np.sqrt(self._sin_sq_2th12+(self._cos_2th12 - self._A/self._Alpha)**2)
        self._C13 = np.sqrt(self._sin_sq_2th13+(self._A-self._cos_2th13)**2)
        
    def P(self, FlavBefore, FlavAfter, E):
        antinu = FlavBefore<0 and FlavAfter<0
        if antinu:
            FlavBefore *= -1
            FlavAfter  *= -1
        
        if   FlavBefore==12 and FlavAfter==12: return self.P_ee(E,antinu)
        elif FlavBefore==12 and FlavAfter==14: return self.P_em(E,antinu)
        elif FlavBefore==12 and FlavAfter==16: return self.P_et(E,antinu)
        elif FlavBefore==14 and FlavAfter==12: return self.P_me(E,antinu)
        elif FlavBefore==14 and FlavAfter==14: return self.P_mm(E,antinu)
        elif FlavBefore==14 and FlavAfter==16: return self.P_mt(E,antinu)
        elif FlavBefore==16 and FlavAfter==12: return self.P_te(E,antinu)
        elif FlavBefore==16 and FlavAfter==14: return self.P_tm(E,antinu)
        elif FlavBefore==16 and FlavAfter==16: return self.P_tt(E,antinu)
        return 0

    def P_ee(self, E, antinu): return self.P_internal_ee(E,antinu,False)
    def P_em(self, E, antinu): return self.P_internal_me(E,antinu,True)
    def P_et(self, E, antinu): return self.P_internal_te(E,antinu,True)

    def P_me(self, E, antinu): return self.P_internal_me(E,antinu,False)
    def P_mm(self, E, antinu): return 1-self.P_me(E,antinu)-self.P_mt(E,antinu)
    def P_mt(self, E, antinu): return self.P_internal_mt(E,antinu,False)
    
    def P_te(self, E, antinu): return self.P_internal_te(E,antinu,False)
    def P_tm(self, E, antinu): return self.P_internal_mt(E,antinu,True)
    def P_tt(self, E, antinu): return 1 - self.P_te(E,antinu) - self.P_tm(E,antinu)

    def P_internal_ee(self, E, antinu, fliptime):
        self.Update()
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D = np.cos(self._C13*self._D)
        sinC13D = np.sin(self._C13*self._D)
        sinaC12D = np.sin(self._Alpha*self._C12*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        p1 = 1 - self._sin_sq_2th13*sinC13D*sinC13D/self._C13**2

        # Terms that appear at order alpha
        p2Inner = self._D*cosC13D*(1-self._A*self._cos_2th13)/self._C13 - \
                  self._A*sinC13D*(self._cos_2th13-self._A)/self._C13**2

        p2 = 2*self._sin_th12*self._sin_th12*self._sin_sq_2th13*sinC13D/self._C13**2*p2Inner*self._Alpha

        # p1 + p2 is the complete contribution for this expansion

        # Now for the expansion in orders of sin(th13) (good to all order alpha)

        pa1,pa2 = 1.0,0.0
        if np.abs(self._Alpha)> 1e-10:
            # leading order term
            pa1 = 1 - self._sin_sq_2th12*sinaC12D*sinaC12D/self._C12**2

        # pa1 is the complete contribution from this expansion, there is no order s13^1 term

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        repeated = 1

        # Calculate the total probability
        totalP = p1+p2 + (pa1+pa2) - repeated
        return totalP

    def P_internal_me(self, E, antinu, fliptime):
        self.Update()
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D =   np.cos(self._C13*self._D)
        sinC13D =   np.sin(self._C13*self._D)
        sin1pAD =   np.sin((self._A+1)*self._D)
        cos1pAD =   np.cos((self._A+1)*self._D)
        sinAD =     np.sin(self._A*self._D)
        sinAm1D =   np.sin((self._A-1)*self._D)
        cosdpD =    np.cos(self._dCPproxy+self._D)
        sinApam2D = np.sin((self._A+self._Alpha-2)*self._D)
        cosApam2D = np.cos((self._A+self._Alpha-2)*self._D)
        cosaC12D =  np.cos(self._Alpha*self._C12*self._D)
        sinaC12D =  np.sin(self._Alpha*self._C12*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        p1 = self._sin_sq_th23*self._sin_sq_2th13*sinC13D*sinC13D/self._C13**2

        # Terms that appear at order alpha
        p2Inner = self._D*cosC13D*(1-self._A*self._cos_2th13)/self._C13 - \
                  self._A*sinC13D*(self._cos_2th13-self._A)/self._C13**2

        p2 = -2*self._sin_sq_th12*self._sin_sq_th23*self._sin_sq_2th13*sinC13D/self._C13**2*p2Inner*self._Alpha

        p3Inner = -self._sin_dCPproxy*(cosC13D - cos1pAD)*self._C13 \
                  +self._cos_dCPproxy*(self._C13*sin1pAD - (1-self._A*self._cos_2th13)*sinC13D)

        p3 = self._sin_2th12*self._sin_2th23*self._sin_th13*sinC13D/(self._A*self._C13**2)*p3Inner*self._Alpha

        # p1 + p2 + p3 is the complete contribution for this expansion
  
        # Now for the expansion in orders of sin(th13) (good to all order alpha) 

        pa1,pa2 = 0.0,0.0
        if np.abs(self._Alpha)>1e-10:
            # leading order term
            pa1 = self._cos_th23*self._cos_th23*self._sin_sq_2th12*sinaC12D*sinaC12D/self._C12**2

            # the first order in s13 term
            t1 = (self._cos_2th12 - self._A/self._Alpha)/self._C12 \
	         - self._Alpha*self._A*self._C12*self._sin_sq_2th12/(2*(1-self._Alpha)*self._C12**2)
            t2 = -self._cos_dCPproxy*(sinApam2D-sinaC12D*t1)
            t3 = -(cosaC12D-cosApam2D)*self._sin_dCPproxy
            denom = (1-self._A-self._Alpha+self._A*self._Alpha*self._cos_th12**2)*self._C12
            t4 = self._sin_2th12*self._sin_2th23*(1-self._Alpha)*sinaC12D/denom

            pa2 = t4*(t3+t2)*self._sin_th13

        # pa1+pa2 is the complete contribution from this expansion

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        t1 = sinAD*cosdpD*sinAm1D/(self._A*(self._A-1))
        repeated = 2*self._Alpha*self._sin_2th12*self._sin_2th23*self._sin_th13*t1

        # Calculate the total probability
        totalP = p1+p2+p3 + (pa1+pa2) - repeated
        return totalP

    def P_internal_te(self, E, antinu, fliptime):
        self.Update()
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D = np.cos(self._C13*self._D)
        sinC13D = np.sin(self._C13*self._D)
        sin1pAD = np.sin((self._A+1)*self._D)
        cos1pAD = np.cos((self._A+1)*self._D)
        sinAD = np.sin(self._A*self._D)
        sinAm1D = np.sin((self._A-1)*self._D)
        cosdpD = np.cos(self._dCPproxy+self._D)
        sinApam2D = np.sin((self._A+self._Alpha-2)*self._D)
        cosApam2D = np.cos((self._A+self._Alpha-2)*self._D)
        cosaC12D = np.cos(self._Alpha*self._C12*self._D)
        sinaC12D = np.sin(self._Alpha*self._C12*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        p1 = self._cos_sq_th23*self._sin_sq_2th13*sinC13D*sinC13D/self._C13**2

        # Terms that appear at order alpha
        p2Inner = self._D*cosC13D*(1-self._A*self._cos_2th13)/self._C13 - \
                  self._A*sinC13D*(self._cos_2th13-self._A)/self._C13**2

        p2 = -2*self._sin_sq_th12*self._cos_sq_th23*self._sin_sq_2th13*sinC13D/self._C13**2*p2Inner*self._Alpha

        p3Inner = -self._sin_dCPproxy*(cosC13D - cos1pAD)*self._C13 \
                  +self._cos_dCPproxy*(self._C13*sin1pAD - (1-self._A*self._cos_2th13)*sinC13D)

        p3 = self._sin_2th12*(-self._sin_2th23)*self._sin_th13*sinC13D/(self._A*self._C13**2)*p3Inner*self._Alpha

        # p1 + p2 + p3 is the complete contribution for this expansion
        
        # Now for the expansion in orders of sin(th13) (good to all order falpha) 
        
        pa1,pa2 = 0.0,0.0
        if np.abs(self._Alpha)>1E-10:
            # leading order term
            pa1 = self._sin_th23*self._sin_th23*self._sin_sq_2th12*sinaC12D*sinaC12D/self._C12**2

            # the first order in s13 term
            t1 = (self._cos_2th12 - self._A/self._Alpha)/self._C12 \
	        - self._Alpha*self._A*self._C12*self._sin_sq_2th12/(2*(1-self._Alpha)*self._C12**2)
            t2 = -self._cos_dCPproxy*(sinApam2D-sinaC12D*t1)
            t3 = -(cosaC12D-cosApam2D)*self._sin_dCPproxy
            denom = (1-self._A-self._Alpha+self._A*self._Alpha*self._cos_th12**2)*self._C12
            t4 = self._sin_2th12*(-self._sin_2th23)*(1-self._Alpha)*sinaC12D/denom

            pa2 = t4*(t3+t2)*self._sin_th13

        # pa1+pa2 is the complete contribution from this expansion

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        t1 = sinAD*cosdpD*sinAm1D/(self._A*(self._A-1))
        repeated = 2*self._Alpha*self._sin_2th12*(-self._sin_2th23)*self._sin_th13*t1

        # Calculate the total probability
        totalP = p1+p2+p3 + (pa1+pa2) - repeated
        return totalP

    def P_internal_mt(self, E, antinu, fliptime):
        self.Update()
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D = np.cos(self._C13*self._D)
        sinC13D = np.sin(self._C13*self._D)
        sin1pAD = np.sin((self._A+1)*self._D)
        cos1pAD = np.cos((self._A+1)*self._D)
        sinAD = np.sin(self._A*self._D)
        sinAm1D = np.sin((self._A-1)*self._D)
        cosAm1D = np.cos((self._A-1)*self._D)
        sinApam2D = np.sin((self._A+self._Alpha-2)*self._D)
        cosApam2D = np.cos((self._A+self._Alpha-2)*self._D)
        cosaC12D = np.cos(self._Alpha*self._C12*self._D)
        sinaC12D = np.sin(self._Alpha*self._C12*self._D)
        sin1pAmCD = np.sin(0.5*(self._A+1-self._C13)*self._D)
        sin1pApCD = np.sin(0.5*(self._A+1+self._C13)*self._D)
        sinD = np.sin(self._D)
        sin2D = np.sin(2*self._D)
        cosaC12pApam2D = np.cos((self._Alpha*self._C12+self._A+self._Alpha-2)*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        pmt_0 = 0.5*self._sin_sq_2th23
        pmt_0 *= (1 - (self._cos_2th13-self._A)/self._C13)*sin1pAmCD**2 \
            +  (1 + (self._cos_2th13-self._A)/self._C13)*sin1pApCD**2 \
            - 0.5*self._sin_sq_2th13*sinC13D**2/self._C13**2

        # Terms that appear at order alpha
        t0 = (self._cos_th12**2-self._sin_th12**2*self._sin_th13**2 \
              *(1+2*self._sin_th13**2*self._A+self._A**2)/self._C13**2)*cosC13D*sin1pAD*2
        t1 = 2*(self._cos_th12**2*self._cos_th13**2-self._cos_th12**2*self._sin_th13**2 \
	        +self._sin_th12**2*self._sin_th13**2 \
	        +(self._sin_th12**2*self._sin_th13**2-self._cos_th12**2)*self._A)
        t1 *= sinC13D*cos1pAD/self._C13

        t2 = self._sin_th12**2*self._sin_sq_2th13*sinC13D/self._C13**3
        t2 *= self._A/self._D*sin1pAD+self._A/self._D*(self._cos_2th13-self._A)/self._C13*sinC13D \
            - (1-self._A*self._cos_2th13)*cosC13D

        pmt_1 = -0.5*self._sin_sq_2th23*self._D*(t0+t1+t2)   

        t0 = cosC13D-cos1pAD
        t1 = 2*self._cos_th13**2*self._sin_dCPproxy*sinC13D/self._C13*t0
        t2 = -self._cos_2th23*self._cos_dCPproxy*(1+self._A)*t0*t0

        t3 = self._cos_2th23*self._cos_dCPproxy*(sin1pAD+(self._cos_2th13-self._A)/self._C13*sinC13D)
        t3 *= (1+2*self._sin_th13**2*self._A + self._A**2)*sinC13D/self._C13 - (1+self._A)*sin1pAD

        pmt_1 += (t1+t2+t3)*self._sin_th13*self._sin_2th12*self._sin_2th23/(2*self._A*self._cos_th13**2)
        pmt_1 *= self._Alpha

        #  pmt_0 + pmt_1 is the complete contribution for this expansion

        # Now for the expansion in orders of sin(th13) (good to all order alpha)

        # Leading order term
        pmt_a0 =  0.5*self._sin_sq_2th23

        pmt_a0 *= 1 - 0.5*self._sin_sq_2th12*sinaC12D**2/self._C12**2 \
            - cosaC12pApam2D \
            - (1 - (self._cos_2th12 - self._A/self._Alpha)/self._C12)*sinaC12D*sinApam2D
            
        denom = (1-self._A-self._Alpha+self._A*self._Alpha*self._cos_th12**2)*self._C12

        t0 = (cosaC12D-cosApam2D)**2
        t1 = (self._cos_2th12 - self._A/self._Alpha)/self._C12*sinaC12D+sinApam2D
        t2 = ((self._cos_2th12 - self._A/self._Alpha)/self._C12 + \
              2*(1-self._Alpha)/(self._Alpha*self._A*self._C12))*sinaC12D + sinApam2D

        t3 = (self._Alpha*self._A*self._C12)/2*self._cos_2th23*self._cos_dCPproxy*(t0 + t1*t2)
        t3 += self._sin_dCPproxy*(1-self._Alpha)*(cosaC12D-cosApam2D)*sinaC12D

        pmt_a1 = self._sin_th13*self._sin_2th12*self._sin_2th23/denom*t3

        # pmt_a1+pmt_a2 is the complete contribution from this expansion

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        t1 = self._sin_dCPproxy*sinD*sinAD*sinAm1D/(self._A*(self._A-1))
        t2 = -1/(self._A-1)*self._cos_dCPproxy*sinD*(self._A*sinD-sinAD*cosAm1D/self._A)*self._cos_2th23/denom

        t0 =  2*self._Alpha*self._sin_2th12*self._sin_2th23*self._sin_th13*(t1+t2)

        t1 = self._sin_sq_2th23*sinD*sinD \
            - self._Alpha*self._sin_sq_2th23*self._cos_th12*self._cos_th12*self._D*sin2D

        repeated = t0+t1

        #  Calculate the total probability
        totalP = pmt_0 + pmt_1 + pmt_a0 + pmt_a1 - repeated

        return totalP

'''
import numpy as np
import numba as nb

spec = [
    ('_Rho',nb.float64),
    ('_L',nb.int64),
    ('_Dmsq21',nb.float64),
    ('_Dmsq32',nb.float64),
    ('_Dmsq31',nb.float64),
    ('_Th12',nb.float64),
    ('_Th13',nb.float64),
    ('_Th23',nb.float64),
    ('_dCP',nb.float64),
    ('_hbar_c_eV_cm',nb.float64),
    ('_hbar_c_eV_km',nb.float64),
    ('_eVPerGeV',nb.float64),
    ('_Na',nb.float64),
    ('_ZperA',nb.float64),
    ('_G_F',nb.float64),
    ('_Alpha',nb.float64),
    ('_sin_th12',nb.float64),
    ('_sin_th13',nb.float64),
    ('_sin_th23',nb.float64),
    ('_cos_th12',nb.float64),
    ('_cos_th13',nb.float64),
    ('_cos_th23',nb.float64),
    ('_sin_2th12',nb.float64),
    ('_sin_2th13',nb.float64),
    ('_sin_2th23',nb.float64),
    ('_cos_2th12',nb.float64),
    ('_cos_2th13',nb.float64),
    ('_cos_2th23',nb.float64),
    ('_sin_sq_th12',nb.float64),
    ('_sin_sq_th13',nb.float64),
    ('_sin_sq_th23',nb.float64),
    ('_cos_sq_th12',nb.float64),
    ('_cos_sq_th13',nb.float64),
    ('_cos_sq_th23',nb.float64),
    ('_sin_sq_2th12',nb.float64),
    ('_sin_sq_2th13',nb.float64),
    ('_sin_sq_2th23',nb.float64),
    ('_cos_sq_2th12',nb.float64),
    ('_cos_sq_2th13',nb.float64),
    ('_cos_sq_2th23',nb.float64),
    ('_V',nb.float64),
    ('_A',nb.float64[:]),
    ('_D',nb.float64[:]),
    ('_dCPproxy',nb.float64),
    ('_sin_dCPproxy',nb.float64),
    ('_cos_dCPproxy',nb.float64),
    ('_C12',nb.float64[:]),
    ('_C13',nb.float64[:]),
]

@nb.experimental.jitclass(spec)
class OscCalc:
    def __init__(self):
        # put some sensible defaults here...
        self._Rho = 2.75 # g/cm^3
        self._L = 810 # km
        
        self._Dmsq21 = 7.59e-5 # eV^2
        self._Dmsq32 = 2.43e-3 # eV^2
        
        self._Th12 = 0.601
        self._Th13 = 0.0
        self._Th23 = 7.85398163397448279e-01 # pi/4
        
        self._dCP = 0

        self._hbar_c_eV_cm = 1.97326938e-5 # eV-cm
        self._hbar_c_eV_km = 1.97326938e-10 # eV-km
        self._eVPerGeV = 1e9

        self._Na = 6.0221409e23 # mol^-1
        self._ZperA = 0.5 # e- per nucleon
        self._G_F = 1.16637e-23 # eV^-2

        self._Dmsq31 = self._Dmsq21 + self._Dmsq32
        self._Alpha = self._Dmsq21 / self._Dmsq31
        self._sin_th12 = np.sin(self._Th12)
        self._sin_th13 = np.sin(self._Th13)
        self._sin_th23 = np.sin(self._Th23)
        self._cos_th12 = np.cos(self._Th12)
        self._cos_th13 = np.cos(self._Th13)
        self._cos_th23 = np.cos(self._Th23)
        self._sin_2th12 = np.sin(2*self._Th12)
        self._sin_2th13 = np.sin(2*self._Th13)
        self._sin_2th23 = np.sin(2*self._Th23)
        self._cos_2th12 = np.cos(2*self._Th12)
        self._cos_2th13 = np.cos(2*self._Th13)
        self._cos_2th23 = np.cos(2*self._Th23)
        self._sin_sq_th12 = self._sin_th12**2
        self._sin_sq_th13 = self._sin_th13**2
        self._sin_sq_th23 = self._sin_th23**2
        self._cos_sq_th12 = self._cos_th12**2
        self._cos_sq_th13 = self._cos_th13**2
        self._cos_sq_th23 = self._cos_th23**2
        self._sin_sq_2th12 = self._sin_2th12**2
        self._sin_sq_2th13 = self._sin_2th13**2
        self._sin_sq_2th23 = self._sin_2th23**2
        self._cos_sq_2th12 = self._cos_2th12**2
        self._cos_sq_2th13 = self._cos_2th13**2
        self._cos_sq_2th23 = self._cos_2th23**2

        self._V = np.sqrt(2)*self._G_F*self._Rho*self._ZperA*self._Na*self._hbar_c_eV_cm**3

    def UpdateEDep(self, E, antinu, fliptime):
        s = -1 if antinu else 1
        t = -1 if fliptime else 1

        self._A = s*2*self._V*E*self._eVPerGeV/self._Dmsq31
        self._D = self._Dmsq31*self._L/(4*E*self._eVPerGeV*self._hbar_c_eV_km)

        self._dCPproxy = s*t*self._dCP
        self._sin_dCPproxy = np.sin(self._dCPproxy)
        self._cos_dCPproxy = np.cos(self._dCPproxy)

        self._C12 = np.sqrt(self._sin_sq_2th12+(self._cos_2th12 - self._A/self._Alpha)**2)
        self._C13 = np.sqrt(self._sin_sq_2th13+(self._A-self._cos_2th13)**2)
        
    def P(self, FlavBefore, FlavAfter, E):
        antinu = FlavBefore<0 and FlavAfter<0
        if antinu:
            FlavBefore *= -1
            FlavAfter  *= -1
        
        if   FlavBefore==12 and FlavAfter==12: return self.P_ee(E,antinu)
        elif FlavBefore==12 and FlavAfter==14: return self.P_em(E,antinu)
        elif FlavBefore==12 and FlavAfter==16: return self.P_et(E,antinu)
        elif FlavBefore==14 and FlavAfter==12: return self.P_me(E,antinu)
        elif FlavBefore==14 and FlavAfter==14: return self.P_mm(E,antinu)
        elif FlavBefore==14 and FlavAfter==16: return self.P_mt(E,antinu)
        elif FlavBefore==16 and FlavAfter==12: return self.P_te(E,antinu)
        elif FlavBefore==16 and FlavAfter==14: return self.P_tm(E,antinu)
        elif FlavBefore==16 and FlavAfter==16: return self.P_tt(E,antinu)
        return np.zeros_like(E)

    def P_ee(self, E, antinu): return self.P_internal_ee(E,antinu,False)
    def P_em(self, E, antinu): return self.P_internal_me(E,antinu,True)
    def P_et(self, E, antinu): return self.P_internal_te(E,antinu,True)

    def P_me(self, E, antinu): return self.P_internal_me(E,antinu,False)
    def P_mm(self, E, antinu): return 1-self.P_me(E,antinu)-self.P_mt(E,antinu)
    def P_mt(self, E, antinu): return self.P_internal_mt(E,antinu,False)
    
    def P_te(self, E, antinu): return self.P_internal_te(E,antinu,False)
    def P_tm(self, E, antinu): return self.P_internal_mt(E,antinu,True)
    def P_tt(self, E, antinu): return 1 - self.P_te(E,antinu) - self.P_tm(E,antinu)

    def P_internal_ee(self, E, antinu, fliptime):
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D = np.cos(self._C13*self._D)
        sinC13D = np.sin(self._C13*self._D)
        sinaC12D = np.sin(self._Alpha*self._C12*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        p1 = 1 - self._sin_sq_2th13*sinC13D*sinC13D/self._C13**2

        # Terms that appear at order alpha
        p2Inner = self._D*cosC13D*(1-self._A*self._cos_2th13)/self._C13 - \
                  self._A*sinC13D*(self._cos_2th13-self._A)/self._C13**2

        p2 = 2*self._sin_th12*self._sin_th12*self._sin_sq_2th13*sinC13D/self._C13**2*p2Inner*self._Alpha

        # p1 + p2 is the complete contribution for this expansion

        # Now for the expansion in orders of sin(th13) (good to all order alpha)

        #pa1,pa2 = 1.0,0.0
        if np.abs(self._Alpha) > 1e-10:
            # leading order term
            pa1 = 1 - self._sin_sq_2th12*sinaC12D*sinaC12D/self._C12**2
        else:
            pa1 = np.ones_like(self._A)
        pa2 = 0.0

        # pa1 is the complete contribution from this expansion, there is no order s13^1 term

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        repeated = 1.0

        # Calculate the total probability
        totalP = p1+p2 + (pa1+pa2) - repeated
        return totalP

    def P_internal_me(self, E, antinu, fliptime):
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D =   np.cos(self._C13*self._D)
        sinC13D =   np.sin(self._C13*self._D)
        sin1pAD =   np.sin((self._A+1)*self._D)
        cos1pAD =   np.cos((self._A+1)*self._D)
        sinAD =     np.sin(self._A*self._D)
        sinAm1D =   np.sin((self._A-1)*self._D)
        cosdpD =    np.cos(self._dCPproxy+self._D)
        sinApam2D = np.sin((self._A+self._Alpha-2)*self._D)
        cosApam2D = np.cos((self._A+self._Alpha-2)*self._D)
        cosaC12D =  np.cos(self._Alpha*self._C12*self._D)
        sinaC12D =  np.sin(self._Alpha*self._C12*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        p1 = self._sin_sq_th23*self._sin_sq_2th13*sinC13D*sinC13D/self._C13**2

        # Terms that appear at order alpha
        p2Inner = self._D*cosC13D*(1-self._A*self._cos_2th13)/self._C13 - \
                  self._A*sinC13D*(self._cos_2th13-self._A)/self._C13**2

        p2 = -2*self._sin_sq_th12*self._sin_sq_th23*self._sin_sq_2th13*sinC13D/self._C13**2*p2Inner*self._Alpha

        p3Inner = -self._sin_dCPproxy*(cosC13D - cos1pAD)*self._C13 \
                  +self._cos_dCPproxy*(self._C13*sin1pAD - (1-self._A*self._cos_2th13)*sinC13D)

        p3 = self._sin_2th12*self._sin_2th23*self._sin_th13*sinC13D/(self._A*self._C13**2)*p3Inner*self._Alpha

        # p1 + p2 + p3 is the complete contribution for this expansion
  
        # Now for the expansion in orders of sin(th13) (good to all order alpha) 

        #pa1,pa2 = 0.0,0.0
        if np.abs(self._Alpha) > 1e-10:
            # leading order term
            pa1 = self._cos_th23*self._cos_th23*self._sin_sq_2th12*sinaC12D*sinaC12D/self._C12**2

            # the first order in s13 term
            t1 = (self._cos_2th12 - self._A/self._Alpha)/self._C12 \
	         - self._Alpha*self._A*self._C12*self._sin_sq_2th12/(2*(1-self._Alpha)*self._C12**2)
            t2 = -self._cos_dCPproxy*(sinApam2D-sinaC12D*t1)
            t3 = -(cosaC12D-cosApam2D)*self._sin_dCPproxy
            denom = (1-self._A-self._Alpha+self._A*self._Alpha*self._cos_th12**2)*self._C12
            t4 = self._sin_2th12*self._sin_2th23*(1-self._Alpha)*sinaC12D/denom

            pa2 = t4*(t3+t2)*self._sin_th13
        else:
            pa1 = np.zeros_like(self._A)
            pa2 = np.zeros_like(self._A)

        # pa1+pa2 is the complete contribution from this expansion

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        t1 = sinAD*cosdpD*sinAm1D/(self._A*(self._A-1))
        repeated = 2*self._Alpha*self._sin_2th12*self._sin_2th23*self._sin_th13*t1

        # Calculate the total probability
        totalP = p1+p2+p3 + (pa1+pa2) - repeated
        return totalP

    def P_internal_te(self, E, antinu, fliptime):
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D = np.cos(self._C13*self._D)
        sinC13D = np.sin(self._C13*self._D)
        sin1pAD = np.sin((self._A+1)*self._D)
        cos1pAD = np.cos((self._A+1)*self._D)
        sinAD = np.sin(self._A*self._D)
        sinAm1D = np.sin((self._A-1)*self._D)
        cosdpD = np.cos(self._dCPproxy+self._D)
        sinApam2D = np.sin((self._A+self._Alpha-2)*self._D)
        cosApam2D = np.cos((self._A+self._Alpha-2)*self._D)
        cosaC12D = np.cos(self._Alpha*self._C12*self._D)
        sinaC12D = np.sin(self._Alpha*self._C12*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        p1 = self._cos_sq_th23*self._sin_sq_2th13*sinC13D*sinC13D/self._C13**2

        # Terms that appear at order alpha
        p2Inner = self._D*cosC13D*(1-self._A*self._cos_2th13)/self._C13 - \
                  self._A*sinC13D*(self._cos_2th13-self._A)/self._C13**2

        p2 = -2*self._sin_sq_th12*self._cos_sq_th23*self._sin_sq_2th13*sinC13D/self._C13**2*p2Inner*self._Alpha

        p3Inner = -self._sin_dCPproxy*(cosC13D - cos1pAD)*self._C13 \
                  +self._cos_dCPproxy*(self._C13*sin1pAD - (1-self._A*self._cos_2th13)*sinC13D)

        p3 = self._sin_2th12*(-self._sin_2th23)*self._sin_th13*sinC13D/(self._A*self._C13**2)*p3Inner*self._Alpha

        # p1 + p2 + p3 is the complete contribution for this expansion
        
        # Now for the expansion in orders of sin(th13) (good to all order falpha) 
        
        #pa1,pa2 = 0.0,0.0
        if np.abs(self._Alpha) > 1E-10:
            # leading order term
            pa1 = self._sin_th23*self._sin_th23*self._sin_sq_2th12*sinaC12D*sinaC12D/self._C12**2

            # the first order in s13 term
            t1 = (self._cos_2th12 - self._A/self._Alpha)/self._C12 \
	        - self._Alpha*self._A*self._C12*self._sin_sq_2th12/(2*(1-self._Alpha)*self._C12**2)
            t2 = -self._cos_dCPproxy*(sinApam2D-sinaC12D*t1)
            t3 = -(cosaC12D-cosApam2D)*self._sin_dCPproxy
            denom = (1-self._A-self._Alpha+self._A*self._Alpha*self._cos_th12**2)*self._C12
            t4 = self._sin_2th12*(-self._sin_2th23)*(1-self._Alpha)*sinaC12D/denom

            pa2 = t4*(t3+t2)*self._sin_th13
        else:
            pa1 = np.zeros_like(self._A)
            pa2 = np.zeros_like(self._A)

        # pa1+pa2 is the complete contribution from this expansion

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        t1 = sinAD*cosdpD*sinAm1D/(self._A*(self._A-1))
        repeated = 2*self._Alpha*self._sin_2th12*(-self._sin_2th23)*self._sin_th13*t1

        # Calculate the total probability
        totalP = p1+p2+p3 + (pa1+pa2) - repeated
        return totalP

    def P_internal_mt(self, E, antinu, fliptime):
        self.UpdateEDep(E,antinu,fliptime)

        cosC13D = np.cos(self._C13*self._D)
        sinC13D = np.sin(self._C13*self._D)
        sin1pAD = np.sin((self._A+1)*self._D)
        cos1pAD = np.cos((self._A+1)*self._D)
        sinAD = np.sin(self._A*self._D)
        sinAm1D = np.sin((self._A-1)*self._D)
        cosAm1D = np.cos((self._A-1)*self._D)
        sinApam2D = np.sin((self._A+self._Alpha-2)*self._D)
        cosApam2D = np.cos((self._A+self._Alpha-2)*self._D)
        cosaC12D = np.cos(self._Alpha*self._C12*self._D)
        sinaC12D = np.sin(self._Alpha*self._C12*self._D)
        sin1pAmCD = np.sin(0.5*(self._A+1-self._C13)*self._D)
        sin1pApCD = np.sin(0.5*(self._A+1+self._C13)*self._D)
        sinD = np.sin(self._D)
        sin2D = np.sin(2*self._D)
        cosaC12pApam2D = np.cos((self._Alpha*self._C12+self._A+self._Alpha-2)*self._D)

        # First we calculate the terms for the alpha expansion (good to all orders in th13)

        # Leading order term 
        pmt_0 = 0.5*self._sin_sq_2th23
        pmt_0 = pmt_0 * ((1 - (self._cos_2th13-self._A)/self._C13)*sin1pAmCD**2 \
            +  (1 + (self._cos_2th13-self._A)/self._C13)*sin1pApCD**2 \
            - 0.5*self._sin_sq_2th13*sinC13D**2/self._C13**2)

        # Terms that appear at order alpha
        t0 = (self._cos_th12**2-self._sin_th12**2*self._sin_th13**2 \
              *(1+2*self._sin_th13**2*self._A+self._A**2)/self._C13**2)*cosC13D*sin1pAD*2
        t1 = 2*(self._cos_th12**2*self._cos_th13**2-self._cos_th12**2*self._sin_th13**2 \
	        +self._sin_th12**2*self._sin_th13**2 \
	        +(self._sin_th12**2*self._sin_th13**2-self._cos_th12**2)*self._A)
        t1 = t1 * (sinC13D*cos1pAD/self._C13)

        t2 = self._sin_th12**2*self._sin_sq_2th13*sinC13D/self._C13**3
        t2 = t2 * (self._A/self._D*sin1pAD+self._A/self._D*(self._cos_2th13-self._A)/self._C13*sinC13D \
            - (1-self._A*self._cos_2th13)*cosC13D)

        pmt_1 = -0.5*self._sin_sq_2th23*self._D*(t0+t1+t2)   

        t0 = cosC13D-cos1pAD
        t1 = 2*self._cos_th13**2*self._sin_dCPproxy*sinC13D/self._C13*t0
        t2 = -self._cos_2th23*self._cos_dCPproxy*(1+self._A)*t0*t0

        t3 = self._cos_2th23*self._cos_dCPproxy*(sin1pAD+(self._cos_2th13-self._A)/self._C13*sinC13D)
        t3 = t3 * ((1+2*self._sin_th13**2*self._A + self._A**2)*sinC13D/self._C13 - (1+self._A)*sin1pAD)

        pmt_1 += (t1+t2+t3)*self._sin_th13*self._sin_2th12*self._sin_2th23/(2*self._A*self._cos_th13**2)
        pmt_1 *= self._Alpha

        #  pmt_0 + pmt_1 is the complete contribution for this expansion

        # Now for the expansion in orders of sin(th13) (good to all order alpha)

        # Leading order term
        pmt_a0 =  0.5*self._sin_sq_2th23

        pmt_a0 = pmt_a0 * (1 - 0.5*self._sin_sq_2th12*sinaC12D**2/self._C12**2 \
            - cosaC12pApam2D \
            - (1 - (self._cos_2th12 - self._A/self._Alpha)/self._C12)*sinaC12D*sinApam2D)
            
        denom = (1-self._A-self._Alpha+self._A*self._Alpha*self._cos_th12**2)*self._C12

        t0 = (cosaC12D-cosApam2D)**2
        t1 = (self._cos_2th12 - self._A/self._Alpha)/self._C12*sinaC12D+sinApam2D
        t2 = ((self._cos_2th12 - self._A/self._Alpha)/self._C12 + \
              2*(1-self._Alpha)/(self._Alpha*self._A*self._C12))*sinaC12D + sinApam2D

        t3 = (self._Alpha*self._A*self._C12)/2*self._cos_2th23*self._cos_dCPproxy*(t0 + t1*t2)
        t3 += self._sin_dCPproxy*(1-self._Alpha)*(cosaC12D-cosApam2D)*sinaC12D

        pmt_a1 = self._sin_th13*self._sin_2th12*self._sin_2th23/denom*t3

        # pmt_a1+pmt_a2 is the complete contribution from this expansion

        # Now we need to add the two expansions and subtract off the terms that are
        # in both (falpha^1, s13^1)

        t1 = self._sin_dCPproxy*sinD*sinAD*sinAm1D/(self._A*(self._A-1))
        t2 = -1/(self._A-1)*self._cos_dCPproxy*sinD*(self._A*sinD-sinAD*cosAm1D/self._A)*self._cos_2th23/denom

        t0 =  2*self._Alpha*self._sin_2th12*self._sin_2th23*self._sin_th13*(t1+t2)

        t1 = self._sin_sq_2th23*sinD*sinD \
            - self._Alpha*self._sin_sq_2th23*self._cos_th12*self._cos_th12*self._D*sin2D

        repeated = t0+t1

        #  Calculate the total probability
        totalP = pmt_0 + pmt_1 + pmt_a0 + pmt_a1 - repeated

        return totalP
'''
