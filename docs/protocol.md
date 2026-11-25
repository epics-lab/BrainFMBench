# Protocol

The protocol is identical for every model. Only the frozen features change.

## Splits and seeds

For each model, dataset and task, at the full training size:

- a held out test set of 10 percent, drawn with five seeds: 0, 1, 2, 3 and 42
- 5 fold cross validation on the remaining data for each seed, 25 values in total

Boxplots show the 25 cross validation values as the box and the five held out results as points. Tables and the heatmap report the held out mean and standard deviation over the five seeds.

## Probes

| Probe | Classification | Regression |
|---|---|---|
| **Linear** | L2 logistic regression | ridge |
| **Random forest** | 200 trees, depth 6 | 200 trees, depth 6 |

The linear probe picks its regularization by an inner 5 fold cross validation on the training data. Dependencies are pinned, scikit-learn 1.7.1, numpy 2.4.2 and pandas 2.2.3, so the numbers are identical run to run.

## Metrics

Classification reports **balanced accuracy** and **AUC**. Regression reports **MAE** and **R²**. The site opens on balanced accuracy and MAE, and can switch to AUC and R².

## Ranking

There is no single overall score, but the overview orders its columns automatically:

1. For each row, a model's gain over chance is divided by the best model's gain in that row.
2. Only rows where the best model clears chance by at least 0.1 count. Rows where every model sits near chance, such as most mental health scores today, stay visible but do not change the order.
3. Columns are sorted by the average of those values.

The order is recomputed on every rebuild, separately for each probe and registration setting. A box marks the best model in each row; rows where every model ties get no box.

## Versioning

- **New model:** no new version, the leaderboard updates on merge.
- **Minor release:** new datasets, tasks or preprocessing options. Existing numbers do not change.
- **Major release:** a new probe head or cross validation scheme, or any change to existing numbers.

Releases are quarterly. See [Releases](releases.md).
