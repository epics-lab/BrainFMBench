# Labels

`NKI.csv`, `HBN.csv`, `PNC.csv`, `BHRC.csv` and `CCNP.csv` are derived from the participant tables of
[Reproducible Brain Charts (RBC)](https://reprobrainchart.github.io/), which shares these
studies without a data use agreement under **CC BY-NC-SA 4.0**. These label files are
therefore distributed under CC BY-NC-SA 4.0, not under the repository's MIT license
(which covers the code).

| File | Study | Columns |
| --- | --- | --- |
| `NKI.csv` | NKI Rockland Sample | sex, age, bmi, p_factor, internalizing, externalizing, handedness |
| `HBN.csv` | Healthy Brain Network | sex, age, bmi, p_factor, internalizing, externalizing, handedness |
| `PNC.csv` | Philadelphia Neurodevelopmental Cohort | sex, age, bmi, p_factor_nonzero, p_factor, internalizing, externalizing, handedness |
| `BHRC.csv` | Brazilian High Risk Cohort | sex, age, bmi, p_factor, internalizing, externalizing |
| `CCNP.csv` | Chinese Color Nest Project | sex, age, bmi, p_factor, internalizing, externalizing |

`sex` is 1 for female and 0 for male. `p_factor_nonzero` is 1 when RBC's harmonized p factor
(`p_factor_mcelroy_harmonized_all_samples`) is above its floor value (no endorsed symptoms) and
0 at the floor. `p_factor` is the harmonized p factor itself, and `internalizing` and `externalizing`
are the matching harmonized scores (`internalizing_mcelroy_harmonized_all_samples`,
`externalizing_mcelroy_harmonized_all_samples`). In NKI these scores exist for 139 of 958 subjects,
because the CBCL they are derived from covers children only. `handedness` is 1 for left and 0 for
right; other values are left empty. It is included only where at least 50 subjects are left handed
(PNC 201, HBN 112, NKI 88; BHRC has 35 and CCNP 7). PNC includes the 1439 subjects whose T1w passes RBC QC, BHRC 542 and CCNP 178. BHRC is longitudinal: the first session that passes QC is used, and its labels come from that same session. For HBN and NKI, the scores and handedness come from the session whose age matches the age label.

If you use these labels, please cite RBC and the original studies:

- Shafiei, G. et al. Reproducible Brain Charts: An open data resource for mapping brain
  development and its associations with mental health. *Neuron* 113(22), 3758–3779 (2025).
  https://doi.org/10.1016/j.neuron.2025.08.026
- Nooner, K. B. et al. The NKI-Rockland Sample. *Front. Neurosci.* 6, 152 (2012).
- Salum, G. A. et al. High risk cohort study for psychiatric disorders in childhood: rationale, design, methods and preliminary results. *Int. J. Methods Psychiatr. Res.* 24, 58–73 (2015).
- Liu, S. et al. Chinese Color Nest Project: an accelerated longitudinal brain-mind cohort. *Dev. Cogn. Neurosci.* 52, 101020 (2021).
  https://doi.org/10.3389/fnins.2012.00152
- Alexander, L. M. et al. An open resource for transdiagnostic research in pediatric mental
  health and learning disorders. *Sci. Data* 4, 170181 (2017). https://doi.org/10.1038/sdata.2017.181
- Satterthwaite, T. D. et al. Neuroimaging of the Philadelphia Neurodevelopmental Cohort.
  *NeuroImage* 86, 544–553 (2014). https://doi.org/10.1016/j.neuroimage.2013.07.064
