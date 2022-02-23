import sys
from time import time as now

from mpi4py import MPI

from Utils.index import index
from Utils.nd_flux import DeriveFlux
from Cuts.nuebarcc_cuts import *
from Vars.nuebarcc_vars import *

from pandana import *
from Weight.cv_weights_2020 import kCVWgt2020

if MPI.COMM_WORLD.rank == 0:
    nranks = MPI.COMM_WORLD.size
    print(f"Analyzing nuebarccinc with {nranks} ranks.")

input_file = "/global/cfs/cdirs/m3253/pandana_data/grohmc_h5concat_nd_genie_N1810j0211a_nonswap_rhc_nova_v08_full_ndphysics_contain_v1.h5"


l = Loader(input_file, "evt.seq", "spill", indices=index)

# efficiency
phase_space_efficiency_numerator = Spectrum(
    l, kDecafPreselection & kNuebarCC, kTrueElectronEVsCosTheta, weight=kCVWgt2020
)
phase_space_efficiency_denominator = Spectrum(
    l, kMC & kNuebarCC, kTrueElectronEVsCosTheta, weight=kCVWgt2020
)

enu_efficiency_numerator = Spectrum(
    l, kDecafPreselection & kNuebarCC, kTrueNeutrinoE, weight=kCVWgt2020
)
enu_efficiency_denomenator = Spectrum(
    l, kMC & kNuebarCC, kTrueNeutrinoE, weight=kCVWgt2020
)

q2_efficiency_numerator = Spectrum(
    l, kDecafPreselection & kNuebarCC, kTrueQ2, weight=kCVWgt2020
)
q2_efficiency_denomenator = Spectrum(l, kMC & kNuebarCC, kTrueQ2, weight=kCVWgt2020)

# unfolding
phase_space_reco = Spectrum(
    l, kDecafPreselection & kNuebarCC, kRecoElectronEVsCosTheta, weight=kCVWgt2020
)
phase_space_truth = Spectrum(
    l, kDecafPreselection & kNuebarCC, kTrueElectronEVsCosTheta, weight=kCVWgt2020
)

enu_reco = Spectrum(
    l, kDecafPreselection & kNuebarCC, kRecoNeutrinoE, weight=kCVWgt2020
)
enu_truth = Spectrum(
    l, kDecafPreselection & kNuebarCC, kTrueNeutrinoE, weight=kCVWgt2020
)

q2_reco = Spectrum(l, kDecafPreselection & kNuebarCC, kRecoQ2, weight=kCVWgt2020)
q2_truth = Spectrum(l, kDecafPreselection & kNuebarCC, kTrueQ2, weight=kCVWgt2020)

# templates
template_fake_data = Spectrum(l, kDecafPreselection, kNueID, weight=kCVWgt2020)
template_nuelike = Spectrum(
    l, kDecafPreselection & kNueTemplateCut, kNueID, weight=kCVWgt2020
)
template_numulike = Spectrum(
    l, kDecafPreselection & kNumuTemplateCut, kNueID, weight=kCVWgt2020
)
template_nc = Spectrum(
    l, kDecafPreselection & kNCTemplateCut, kNueID, weight=kCVWgt2020
)

# Skip the flux measurement for now since the neutrino tree isn't updated in the current file.
# flux
#flux_spectrum = DeriveFlux(
#    l, 14, min_extent=vNuebarCCIncFiducialMin, max_extent=vNuebarCCIncFiducialMax
#)

start = now()
l.Go()
stop = now()
total = stop - start

total = MPI.COMM_WORLD.reduce(total, MPI.SUM, root=0)

# Turn the templates into histograms
# Just use two bins to get "low pid" and "high pid"
data, bins = template_fake_data.histogram(2, (0, 1), mpireduce=True)
nue, _ = template_nuelike.histogram(bins, mpireduce=True)
numu, _ = template_numulike.histogram(bins, mpireduce=True)
nc, _ = template_nc.histogram(bins, mpireduce=True)

if MPI.COMM_WORLD.rank != 0:
    sys.exit(0)

print(f"Filling took on average {total / nranks} s")

print(f"\t| Low PID\t | High PID")
print(f"Data\t| {data[0]:.4f}\t | {data[1]:.4f}")
print(f"Nue\t| {nue[0]:.4f}\t | {nue[1]:.4f}")
print(f"Numu\t| {numu[0]:.4f}\t | {numu[1]:.4f}")
print(f"NC\t| {nc[0]:.4f}\t | {nc[1]:.4f}")
