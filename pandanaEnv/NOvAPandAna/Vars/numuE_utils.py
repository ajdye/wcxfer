import numpy as np

from Utils.enums import *
from Utils.misc import GetPeriod

# For the Numu Energy Estimator
class SplineFit():
  def __init__(self, spline_spec):
    self.nstitch = len(spline_spec)//2
    self.x0 = spline_spec[0::2]
    self.slopes = spline_spec[1::2]
    self.intercepts = np.zeros(self.nstitch)

    self.intercepts[0] = self.x0[0]
    for i in range(1, self.nstitch):
      self.intercepts[i] = self.intercepts[i-1] + (self.slopes[i-1] - self.slopes[i]) * self.x0[i]

    self.x0[0] = 0.

  def __call__(self, var):
    stitchpos = np.argwhere(self.x0 <= var).max()
    return self.slopes[stitchpos]*var + self.intercepts[stitchpos]

kSplineProd4MuonFDp1 = SplineFit(np.array([
  1.264631942673353215e-01, 2.027211173807817457e-01,
  8.774231219753678701e+00, 2.163481314097564778e-01
]))

kSplineProd4HadFDp1 = SplineFit(np.array([
  7.281506207958776677e-02, 5.441509444740477708e-01,
  5.995673938377335532e-02, 2.016769249502490702e+00
]))

kSplineProd4MuonFDp2 = SplineFit(np.array([
  1.257629394848571724e-01, 2.036422681279472513e-01,
  9.855445529033460161e+00, 2.195160029549789171e-01
]))

kSplineProd4HadFDp2 = SplineFit(np.array([
  5.877366209096868133e-02, 1.519704739772891111e+00,
  7.898381593661769895e-02, 2.169042213902584670e+00,
  4.771203681401176011e-01, 1.694250867119319715e+00,
  7.539010791835750736e-01, 2.059894991370703199e+00
]))

kSplineProd4MuonFDp3 = SplineFit(np.array([
  1.258258375981389232e-01, 2.033924315007327455e-01,
  9.466540687161570489e+00, 2.184781804887378220e-01
]))

kSplineProd4HadFDp3 = SplineFit(np.array([
  5.673223776636682203e-02, 1.465342397469045377e+00,
  8.044448532774936544e-02, 2.106105448794447277e+00,
  4.350000000342279516e-01, 1.829072531499642107e+00
]))

kSplineProd4MuonFDp4 = SplineFit(np.array([
  1.463359246892861343e-01, 1.972096058268091312e-01,
  4.248366183914889405e+00, 2.072087518392393413e-01,
  1.048696659085294414e+01, 2.187994552957214234e-01
]))

kSplineProd4HadFDp4 = SplineFit(np.array([
  4.249458417893325901e-02, 2.193901299460967902e+00,
  1.049999999339812778e-01, 1.981193414403220387e+00,
  4.492042726487709969e-01, 1.540708309857498071e+00,
  6.193806271369780569e-01, 2.032400170863909228e+00
]))

kSplineProd4MuonFDp5 = SplineFit(np.array([
  1.303357512360951986e-01, 2.024876303943876354e-01,
  9.173839619568841641e+00, 2.184865210120549850e-01
]))

kSplineProd4HadFDp5 = SplineFit(np.array([
  5.513045392968107805e-02, 1.534780212240209885e+00,
  8.119825924952998875e-02, 2.102191592086820382e+00,
  4.829923244287704365e-01, 1.655956208512852301e+00,
  7.849999988138458562e-01, 2.030853659350569718e+00
]))

kSplineProd4ActNDp3 = SplineFit(np.array([
  1.514900417035001112e-01, 1.941290659171270860e-01,
  3.285152850349305265e+00, 2.027000969328388302e-01,
  5.768910882104949955e+00, 2.089391482903513730e-01
]))

kSplineProd4CatNDp3 = SplineFit(np.array([
  8.720084706388187001e-03, 5.529278858789209439e-01,
  2.270042802448197783e+00, 1.711916184621219417e+00,
  2.307938644096652947e+00, 3.521795029684806622e-01
]))

kSplineProd4HadNDp3 = SplineFit(np.array([
  6.390888289071572359e-02, 1.416435376103045707e+00,
  4.714208143879788232e-02, 2.144449735801436052e+00,
  2.402894748598130015e-01, 2.453526979604870206e+00,
  4.581240238035835244e-01, 1.730599464528853160e+00
]))

kSplineProd4ActNDp4 = SplineFit(np.array([
  1.534847032701298630e-01, 1.939572586752992267e-01,
  3.076153165048225446e+00, 2.016367496419133321e-01,
  5.139959732881990817e+00, 2.085728450137292189e-01
]))

kSplineProd4CatNDp4 = SplineFit(np.array([
  6.440384628800144284e-02, 1.663238457993588826e-01,
  1.466666666666678887e-01, 5.572409982576314036e-01,
  2.270441314113081255e+00, 2.043488369726459641e+00,
  2.299237514383191794e+00, 3.673229067047741880e-01
]))

kSplineProd4HadNDp4 = SplineFit(np.array([
  4.743190085416526536e-02, 2.510494758751641520e+00,
  9.795449140279462175e-02, 1.981710272551915564e+00
]))

########## PROD 5 SPLINE VALS #########
### Far Detector ###

kSplineProd5MuonFDp1 = SplineFit(np.array([
        1.333815966663624564e-01, 2.006655289624899308e-01,
        8.500671131829244942e+00, 2.154209380453118716e-01,
        2.035090204717860729e+01, 6.660211866896596611e+00,
        2.037655171956283340e+01, 0.000000000000000000e+00
]))
    
kSplineProd5HadFDp1 = SplineFit(np.array([
        1.085140541354344575e-02, 2.003132845468682977e+00,
        5.249999999978114396e-01, 1.587909262160043244e+00,
        8.547074710510785822e-01, 2.070213642469894921e+00
]))
    
kSplineProd5MuonFDp2 = SplineFit(np.array([
        1.333041428039729304e-01, 2.010374129825994727e-01,
        9.015314992888956880e+00, 2.172102602545122607e-01,
        1.881250004932426734e+01, 1.939648844865335120e-01
]))
    
kSplineProd5HadFDp2 = SplineFit(np.array([
        3.384568670664851731e-02, 1.916508023623156864e+00,
        2.749999891254556461e-01, 2.284913434279694400e+00,
        4.495957896158719880e-01, 1.631687421408977157e+00,
        7.893618284090087034e-01, 2.015303688339076693e+00
]))
    
kSplineProd5MuonFDfhc = SplineFit(np.array([
        1.412869875558434574e-01, 1.985202329476516148e-01,
        7.247665189483523562e+00, 2.144069735971011192e-01,
        2.218750000031716141e+01, 6.699485408121087782e-02
]))
    
kSplineProd5HadFDfhc = SplineFit(np.array([
        5.767990231564357195e-02, 1.091963220147889491e+00,
        4.894474691585748438e-02, 2.031445922414648386e+00,
        5.142642860092461188e-01, 1.567915254401344383e+00,
        8.200421075858435049e-01, 2.016845013606002102e+00
]))
    
kSplineProd5MuonFDrhc = SplineFit(np.array([
        1.245271319206025379e-01, 2.033997627592860902e-01,
        9.766311956246607195e+00, 2.180838285862531922e-01,
        2.003715340979164949e+01, 1.863267567727432683e-01,
        2.256004612234155360e+01, 4.754398422961426951e-02
]))
    
kSplineProd5HadFDrhc = SplineFit(np.array([
        4.022415096001341617e-02, 2.011711823080491790e+00,
        4.199763458287808504e-01, 1.595097006634894843e+00,
        7.030242302962290690e-01, 2.148979944911536766e+00,
        1.293968553045185210e+00, 1.500071121804977814e+00
]))
    
### Near Detector ###
kSplineProd5ActNDfhc = SplineFit(np.array([
        1.522067501417963542e-01, 1.935351432875078992e-01,
        3.534675721653096403e+00, 2.025064113727464976e-01,
        6.048717848712632517e+00, 2.086419146240798550e-01
]))
    
kSplineProd5CatNDfhc = SplineFit(np.array([
        6.860056229074447398e-02, 1.021995188252620562e-01,
        1.466667613491428046e-01, 5.498842494606275277e-01,
        2.260114901099927298e+00, 1.411396843018650538e+00,
        2.313275230972585472e+00, 3.115156857428397208e-01
]))
    
kSplineProd5HadNDfhc = SplineFit(np.array([
        5.049552462442885581e-02, 1.422732975320812443e+00,
        6.048754927389610181e-02, 2.709662443207628613e+00,
        1.015235485148796579e-01, 2.173545876693023349e+00,
        5.064530757547176520e-01, 1.725707450251668051e+00
]))
    
kSplineProd5ActNDrhc = SplineFit(np.array([
        1.717171287078189390e-01, 1.853305227171077318e-01,
        2.502586270065958907e+00, 1.990563298599958286e-01,
        5.036450674404544081e+00, 2.083816760775504540e-01
]))
    
kSplineProd5CatNDrhc = SplineFit(np.array([
        1.689154853867225192e-03, 5.492279050571418075e-01
]))
    
kSplineProd5HadNDrhc = SplineFit(np.array([
        4.676766851054844215e-02, 2.206317277398726073e+00,
        3.848300672745982309e-01, 1.593035140670105099e+00,
        6.819800276504310865e-01, 2.100597007299316310e+00,
        1.362679543056420250e+00, 1.417283364717454974e+00
]))



def kApplySpline(run, det, isRHC, comp, len):
  if len <= 0: return 0.
  
  period = GetPeriod(run, det)
  if (period==1) and (det == detector.kFD):
    f = {"muon" : kSplineProd4MuonFDp1, "had" : kSplineProd4HadFDp1}[comp]
  if (period==2) and (det == detector.kFD):
    f = {"muon" : kSplineProd4MuonFDp2, "had" : kSplineProd4HadFDp2}[comp]
  if (period==3) and (det == detector.kFD):
    f = {"muon" : kSplineProd4MuonFDp3, "had" : kSplineProd4HadFDp3}[comp]
  if (period>=4) and (det == detector.kFD) and isRHC:
    f = {"muon" : kSplineProd4MuonFDp4, "had" : kSplineProd4HadFDp4}[comp]
  if (period>=5) and (det == detector.kFD) and not isRHC:
    f = {"muon" : kSplineProd4MuonFDp5, "had" : kSplineProd4HadFDp5}[comp]
  if (det == detector.kND) and isRHC:
    f = {"act" : kSplineProd4ActNDp4, "cat": kSplineProd4CatNDp4, "had": kSplineProd4HadNDp4}[comp]
  if (det == detector.kND) and not isRHC:
    f = {"act" : kSplineProd4ActNDp3, "cat": kSplineProd4CatNDp3, "had": kSplineProd4HadNDp3}[comp]
  return f(len)

kApplySpline = np.vectorize(kApplySpline, otypes=[np.float64])

def kApplySplineProd5(run, det, isRHC, comp, len):
  if len <= 0: return 0.

  period = GetPeriod(run, det)
  if (period==1) and (det == detector.kFD):
    f = {"muon" : kSplineProd5MuonFDp1, "had" : kSplineProd5HadFDp1}[comp]
  if (period==2) and (det == detector.kFD):
    f = {"muon" : kSplineProd5MuonFDp2, "had" : kSplineProd5HadFDp2}[comp]
  if ((period==3) or (period==5) or (period==9) or (period==10)) and (det == detector.kFD):
    f = {"muon" : kSplineProd5MuonFDfhc, "had" : kSplineProd5HadFDfhc}[comp]
  if ((period==4) or (period==6) or (period==7) or (period==8)) and (det == detector.kFD):
    f = {"muon" : kSplineProd5MuonFDrhc, "had" : kSplineProd5HadFDrhc}[comp]
  if (det == detector.kND) and ((period==1) or (period==2) or (period==3) or (period==5) or (period==9)):
    f = {"act" : kSplineProd5ActNDfhc, "cat": kSplineProd5CatNDfhc, "had": kSplineProd5HadNDfhc}[comp]
  if (det == detector.kND) and ((period==4) or (period==6) or (period==7) or (period==8)):
    f = {"act" : kSplineProd5ActNDrhc, "cat": kSplineProd5CatNDrhc, "had": kSplineProd5HadNDrhc}[comp]
  return f(len)

kApplySplineProd5 = np.vectorize(kApplySplineProd5, otypes=[np.float64])
