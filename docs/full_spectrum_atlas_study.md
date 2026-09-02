# Full-spectrum atlas feasibility study

## Question

Can STRIDER use a measurement-faithful accumulated spectrum as a clear,
physically interpretable class--redshift anchor, while retaining individual
visits only for measured spectral evolution?

This is an architecture-selection study. It does not replace the frozen v3
models, reopen their checkpoint selection, or use the calibration/final-test
roles.

## Information boundary

Training simulations may use their class, redshift, clean flux, and simulated
phase to construct and audit a rest-frame reference atlas. A selection or real
observation supplies only measured flux, reported uncertainty, wavelength
coverage, and observation dates.

The same boundary must hold in every study arm:

```text
clean training simulations + training labels
                     |
                     v
       frozen rest-frame reference atlas

new observer-frame measurements
                     |
                     v
       scan every candidate redshift
                     |
                     v
        relative class--redshift reference match
```

Truth redshift or truth phase must never select the location or reference
profile used to score a selection object.

## First bounded comparison

The first study uses the local normal-Ia binary train/selection data and
compares six matched arms:

1. best measured spectrum, full spectral shape;
2. best measured spectrum, continuum-removed spectrum;
3. best measured spectrum, fixed equal combination of the two spectral views;
4. inverse-variance coadd, full spectral shape;
5. inverse-variance coadd, continuum-removed spectrum;
6. inverse-variance coadd, fixed equal combination of the two spectral views.

Each measurement choice gets its own clean training atlas. The best spectrum is
selected by the signed median measured-bin signal-to-noise ratio. The coadd
uses the exact production inverse-variance definition and its propagated
reported error. Neither choice uses clean flux to select a selection spectrum.

The full-spectrum and continuum-removed scores are kept separate before their
fixed combination. This determines whether continuum removal adds useful feature
contrast or merely repeats the same solution. The current 12,000 km/s
mask-aware smoothing width is the declared baseline; it is not optimized on the
selection result.

## Reference profiles

For each sampled training object:

1. construct the clean version of the same best-spectrum or coadd measurement;
2. use training redshift truth to place that clean spectrum on the rest grid;
3. mean-centre and normalize only measured bins;
4. retain several deterministic profiles per class to represent real spectral
   diversity.

The first implementation uses deterministic masked spherical clusters. It is a
reference-profile feasibility test, not the final learned atlas. The resulting
match scores compare hypotheses and must not be described as probabilities.

## Phase decision

Phase indexing is not enabled immediately. The first run records how many
training objects and visits support each broad phase interval. That audit comes
before a phase-indexed atlas so that unsupported class/phase cells cannot be
silently invented.

If support is adequate, the next matched comparison will be:

- **phase-independent reference:** phase variation is represented only by
  multiple class profiles;
- **phase averaged during inference:** STRIDER evaluates possible starting
  phases using observed visit separations and averages over them;
- **truth-informed phase upper bound:** simulated phase is supplied only to
  quantify the greatest possible phase benefit. It is never a deployable arm.

Phase enters a candidate trajectory through

```text
phase_i = starting_phase + (observer_day_i - observer_day_0) / (1 + candidate_z).
```

The phase-aware design advances only if averaging over possible phase improves
selection performance and recovers a meaningful part of the truth-informed
upper bound.

## Required outputs

The study reports, for every arm:

- balanced classification accuracy and normal-Ia F1;
- normal-Ia median absolute redshift error and outlier fraction;
- the separation between the truth-consistent solution and the strongest
  incompatible class/redshift solution;
- performance versus redshift;
- performance versus coadded and best-spectrum measured S/N;
- unsupported wavelength cases;
- class/phase training support.

Later full-scale controls must also include noise-only measurements, amplitude
rescaling, broad continuum tilts, missing visits, altered visit order, and
reported-error noise draws.

## Decision

The study is promising only if the proper coadd improves or preserves both
classification and redshift recovery, particularly at moderate measured S/N,
without acquiring confident structure on noise-only controls. A small gain
that substantially increases runtime does not justify replacing v3.

If the local binary result is promising, repeat the unchanged study on NERSC
with the full binary train/selection roles, then the seven-class and 15-class
selection roles. Calibration and final test remain closed until the complete
architecture is frozen.

## Local command

```bash
PYTHONPATH=src python scripts/study_full_spectrum_atlas.py \
  --config configs/experiments/local_ia_binary_20k.yaml \
  --output-dir runs/atlas_study/local_binary \
  --train-objects 800 \
  --selection-objects 240 \
  --max-visits 32
```

Use small object counts only for code checks. Scientific decisions require the
full NERSC selection study.

## NERSC command

After the study files have been committed and pulled on NERSC:

```bash
cd /pscratch/sd/m/mdixon7/strider
mkdir -p logs

atlas_job=$(sbatch --parsable \
  --time=03:00:00 \
  --export=ALL,STRIDER_CONFIG="$PWD/configs/nersc/ia_binary_20k.yaml",STRIDER_ATLAS_STUDY_OUTPUT="$PWD/runs_detector/atlas_study/binary_selection" \
  nersc/full_spectrum_atlas_study.sh)

echo "Full-spectrum atlas study: $atlas_job"
```

The job uses all prepared binary training and selection objects, all retained
visits for the coadd, and the six best measured visits for the phase-sequence
comparison. It creates the browser-downloadable file
`runs_detector/atlas_study_results.tar.gz`.
