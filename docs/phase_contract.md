# Phase, dates and external redshift information

## Frames and names

Modified Julian Date (MJD) is always an observer-frame calendar date. A date is
not converted to the rest frame. Only a time interval is converted:

\[
\mathrm{rest\ phase}(z,t_0)=\frac{\mathrm{MJD}-t_0}{1+z}.
\]

Here, both the spectrum MJD and peak date \(t_0\) are observer-frame dates.
STRIDER evaluates the interval for every candidate redshift rather than
constructing it once with simulation truth.

SNANA and SALT3 use similar names for different products. In the Hourglass2
files used here:

- raw HEAD `PEAKMJD` is an initial light-curve estimate;
- raw HEAD `SIM_PEAKMJD` is simulation truth; and
- SALT3 FITRES `PKMJD` is the fitted observer-frame peak date, with `PKMJDERR`.

The code stores the first two as `estimated_peak_mjd` and
`simulation_peak_mjd`. It does not call either one a fitted date.

## Initial ONIR profile interval

Use (-20) to (+80) rest-frame days for the first clean ONIR profile bank.
This covers the useful rise, maximum and post-maximum evolution while avoiding
very early sparse profiles and very late spectra that would dominate a
phase-neutral average.

This interval is an offline profile-construction choice, not an input-data cut.

## Stored and model input visits

- Store every valid simulated or observed visit.
- Do not reject visits using truth-derived rest phase.
- Keep observer dates in days and wavelength in the observer frame.
- Use every available visit for validation, testing and final inference.
- During training, mix complete histories with shorter prefixes independently
  of redshift, so the same model can classify both early and mature events.

The prepared simulation record may retain `simulation_peak_mjd` for supervised
bank construction and phase training. `collate_objects` keeps the resulting
`simulation_rest_phase_days` as a target, while `measurement_inputs` passes only
flux, masks, visit coverage and observer-time offsets to the model. Changing the
phase target therefore cannot change a forward prediction.

The all-visit NERSC configs have no scientific visit cap. Half of the training
draws use the complete recorded history and half use shorter prefixes. Long
histories use smaller GPU microbatches with gradient accumulation; this changes
only the computation and does not discard spectra. Missing visits are never
invented. Earlier 32-visit configs remain available as historical controls, not
as the final input design.

## What the current model learns

The spectral ONIR branch is phase-neutral. The temporal branch learns whether
measured cross-visit spectral change is compatible with
`observer_interval_days / (1 + candidate_redshift)`. It can help class and reject
incompatible redshift aliases, but it is not assumed to set redshift precision.

One visit has no cross-visit temporal evidence. Two visits provide one measured
change and may be useful, but absolute phase and evolution-rate degeneracies can
make that evidence weak. The value must be measured by visit count rather than
declared from a degrees-of-freedom argument.

## Peak-date-assisted phase mode

The auxiliary phase head predicts class-conditioned phase bins for every visit
and trial redshift. Simulation phase enters its training loss only at the
labelled class and redshift. The first consistency experiment marginalized an
unknown first-visit phase. It did not materially improve the primary result,
which is expected when absolute phase remains weakly constrained.

The peak-date-assisted experiment instead uses `estimated_peak_mjd`, the
observer-frame light-curve estimate. The model receives only its offset from
the first retained spectrum and a validity flag. For each candidate redshift,
the deterministic consistency calculation converts that interval to candidate
rest phases and integrates over the configured peak-date uncertainty. An
outlier mixture limits the influence of a poor estimate, and the route returns
zero evidence when the estimate is missing or too little of the trajectory is
inside the trained phase interval. `simulation_peak_mjd` remains supervision
and evaluation metadata and is never substituted for a missing estimate.

The fixed uncertainty used in the first comparison is a controlled
approximation. The inference interface should ultimately accept each object's
measured peak-date probability distribution rather than one uncertainty shared
by the sample. Do not precompute one rest phase using a fixed redshift, and do
not force the phase estimate to be more precise than the light curve supports.

A host-galaxy photometric-redshift result can enter as an external probability
distribution over redshift, not as one fixed value. Keep three reported modes
distinct:

1. spectra only: no peak-date or host-redshift information;
2. spectra plus host redshift: multiply by an external host \(p(z)\); and
3. spectra plus light curve: integrate a measured joint \(p(z,t_0)\) when the
   same light-curve fit constrained both quantities.

Using separate \(p(z)\) and \(p(t_0)\) factors from the same fit can count the
same photometric information twice. The joint distribution is the correct
long-term interface. A very precise host spectroscopic redshift changes the
task into classification at effectively known redshift; it is not a fair test
of spectroscopic redshift recovery.
