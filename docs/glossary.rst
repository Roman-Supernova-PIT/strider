Glossary
========

visit
    One prism spectrum of the transient, taken at one MJD.
MJD
    Modified Julian Date, in days.
f_λ
    Flux per unit wavelength.
native pixel
    A pixel of a spectrum as you give it, before STRIDER resamples it onto the model's wavelength grid.
S/N
    Signal-to-noise ratio. Model cards give it per native pixel.
phase
    Days since the transient's peak brightness, in its rest frame. STRIDER does not estimate it.
prior
    How common each class is at each redshift, before looking at the data. STRIDER uses the mix of transients Roman
    is expected to find, given on the model card.
posterior
    The probability of each class or redshift, from the data and the prior together.
redshift cell
    One of the redshift bins STRIDER scores. ``redshift_grid`` gives their centres.
source probability
    The probability that the visits hold a transient at all.
subtype
    A simulated type within a class. In ``strider-15class``, each class is a single subtype.
model package
    The one file that holds a model, with its priors and model card.
model card
    A short description of a model: its classes, training, valid range, performance and known failures.
fingerprint
    A code that identifies a model: its network, brightness term, priors and settings. Every result records it,
    with a checksum of every file of the model package.
PIT
    The Roman Supernova Project Infrastructure Team, which maintains STRIDER.
SMDC
    The Roman computing centre where the PIT runs its software (:doc:`smdc`).
