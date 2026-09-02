# Research history and repository scope

STRIDER developed through several simulation studies. Earlier configurations,
scripts and reports are kept because they explain scientific decisions and may
be needed to repeat paper and NERSC work. Their presence does not make every
route part of the public tool.

## How to read the repository

| Category | Current contents | Treatment |
|---|---|---|
| Public core | measured-data preparation, accumulation, Roman reference matching, posterior, calibration, deployment and their tests | Keep small, documented and stable |
| Run records and experiment history | detailed runbooks, result notes, controls, plots and historical configs | Preserve with clear status and provenance |
| Candidate awaiting results | reference architecture plus the corrected two-epoch selection gate | Keep explicit; make no superiority claim |
| Possible later archive | superseded model routes and one-off launch/config files after dependencies are mapped | Review after the paper and active NERSC work; do not delete by age |

The public name, command and Python import are all **STRIDER** / `strider`. The
distribution keeps the existing `roman-snpit-strider` identity so a later
install can supersede the old public package cleanly. Numeric suffixes on
artifact formats identify incompatible file schemas; they are not versions of
the research tool.

The earlier public phase-input implementation is historical provenance. Before
replacing either public repository, preserve its current commit (`32a9719`) on
a named tag, branch or archival mirror. It must not be used to explain the
current runtime boundary: the current tool receives neither truth redshift nor
truth-derived rest-frame phase.

The reference candidate documented here was audited from commit `6c05521`. Its
frozen calibrated comparison model is still the verified baseline. The
candidate is awaiting matched selection evaluation under the corrected
two-epoch gate.

Before moving anything to an archive, trace imports, configuration inheritance,
batch-script references, paper references and artifact provenance. A useful
next cleanup pass can then replace duplicated scientific behavior with one
tested implementation while retaining small compatibility adapters and exact
run records.
