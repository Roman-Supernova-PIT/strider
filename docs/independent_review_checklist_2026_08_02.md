# Independent review checklist for the STRIDER local pilot

Conduct an independent, adversarial review of the new STRIDER local Sundial
pilot. Spend time reading the implementation and the saved outputs, rerun
focused tests where useful, and distinguish verified facts from interpretations.
Do not merely agree with the design notes.

The project folder is:

the STRIDER repository root

Start with:

- `README.md`
- `docs/architecture.md`
- `docs/local_pilot_results_2026_08_02.md`
- `docs/next_steps.md`
- `configs/local_pilot.yaml`
- `src/strider/data/prepare.py`
- `src/strider/data/dataset.py`
- `src/strider/model/strider.py`
- `src/strider/model/redshift_scan.py`
- `src/strider/training/losses.py`
- `src/strider/evaluation/evaluate.py`

Saved numerical outputs are under `runs/sundial_local_pilot`.  The matched
phase-neutral experiment is under `runs/sundial_local_pilot_no_phase` and its
configuration is `configs/experiments/no_phase.yaml`.

## What was built and tested

- Raw Sundial HEAD/SPEC FITS to validated Parquet tables and ragged native-bin
  HDF5 arrays.
- Fixed split by complete FITS block: 1–7 training, 8 validation, 9–10 test.
- 1,200/300/500 objects, balanced binary class and broad redshift groups.
- Original `FLAM`, clean `SIM_FLAM`, source-free new noise, residual-only,
  fresh reported-error noise, and no-source views.
- Noise creation on native bins followed by one resampling step.
- Candidate-dependent phase from observer time for every trial redshift; no
  truth rest-frame phase is an inference input.
- A phase-neutral spectral CNN and a full rest-frame redshift scan.
- A separate adequacy result trained from cross-visit summaries.
- Metal training and evaluation on an M5 Pro Mac.

Key final results on 500 held-out objects:

- Original input: class accuracy 0.870, median absolute delta z 0.230.
- New source-free noise: class accuracy 0.926, median absolute delta z 0.200.
- Clean input: class accuracy 0.906, median absolute delta z 0.183.
- Residual-only and fresh reported-error no-source views do not recover
  redshift better than source-free no-source noise.
- No-source mean adequacy: 0.165 source-free, 0.249 residual, 0.254 fresh
  reported-error; fractions above 0.5 are 0%, 7%, and 8%.
- With normal times, no-source inputs fall within delta z=0.1 of simulation z
  33.4% of the time.  With all times set to zero this falls to 6.2%.  Reassigning
  the same times to different spectra within each object leaves it at 34.2%.
- On source inputs, that within-object reassignment lowers class accuracy from
  0.926 to 0.886 and worsens median absolute delta z from 0.200 to 0.229.
- The matched phase-neutral model gives 0.720 class accuracy and median absolute
  delta z 0.414; its no-source near-z rate is 5.6%.
- More visits strongly improve binary classification, but also strengthen the
  no-source timing association.

## Questions to attack

1. Is the fixed FITS-block split genuinely safe, including SNID uniqueness,
   simulator template overlap, and any block-level shared seeds?
2. Is lower-quartile positive `FLAMERR` a defensible temporary source-free scale,
   or can its scalar value still encode source or redshift information?
3. Is masking bins above twenty times that scale scientifically defensible, and
   can the resulting mask pattern itself carry redshift information?
4. Is the continuous adequacy target
   `sigmoid(4 log(coadded clean S/N))` well defined?  Review the local S/N
   calculation and suggest a better target if needed.
5. Does separating `has_source`, adequacy, and the conditional joint posterior
   correctly prevent faint sources from becoming negative training examples?
6. Does the timing control really demonstrate a cadence/time-dilation route, or
   is there another implementation explanation?  Inspect the phase embedding,
   visit selection, padding, and aggregation.
7. Is reversing time values within an object a sufficient temporal-evolution
   control?  Design stronger controls that preserve or alter specific timing
   statistics one at a time.
8. Why does the phase-neutral full-spectrum scanner perform so poorly?  Decide
   whether this is data volume, model capacity, normalization, loss design,
   redshift-grid resolution, or absence of named spectral anchors.
9. Does the residual comparison have enough statistical power and the right
   chance baseline?  Examine results by redshift and visit count, not only the
   aggregate.
10. Is the proposed next step—clean ONIR named-feature evidence plus deliberate
    cadence variation—the right order?  Specify the smallest decisive local
    experiment before a full NERSC run.
11. Review the folder and code for correctness, readability, duplicated logic,
    hidden coupling, and paths that would fail on NERSC.
12. Identify any result that is being stated too strongly.

## Review areas

Review these areas separately:

1. data format, split integrity, native-bin noise generation, and run repeatability;
2. model architecture, phase handling, redshift scan, and visit aggregation;
3. statistics, adequacy target, calibration, and control-test power;
4. ONIR integration, cadence variation, and the smallest next local experiment.

Cross-check the claims between areas. Return:

- confirmed findings;
- disputed findings;
- bugs or invalid tests;
- immediate fixes before NERSC;
- the next three experiments in priority order;
- a clear go/no-go decision for using this project as the STRIDER base.
