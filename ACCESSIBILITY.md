# Accessibility

BrainFMBench is a research benchmark, and its results should be usable by everyone, including people who use screen readers, keyboard navigation, magnification or high contrast settings, and people with colour vision differences. This document covers the leaderboard, the repository documentation and the submission process, and explains how to tell us about a barrier.

## Priorities

- Every result is available as text, not only as a figure. LEADERBOARD.md lists all scores in tables that screen readers can read.
- Figures do not rely on colour alone. In the overview heatmap, each cell shows its number, the best model per row is marked with a box, and each row shows whether higher or lower is better.
- Documentation is written in plain language, and submitting a model only needs a text editor, git and a pull request.

We aim to follow WCAG 2.2 AA where it applies, as a goal. The project has not been formally evaluated against it.

## Contributor expectations

- Any new figure needs the same results available as text in LEADERBOARD.md or the README.
- New figures should not rely on colour alone; use labels, numbers, markers or patterns as well.
- Images in documentation need alt text that says what they show.
- Pull requests that change the leaderboard site or figures should say how the change keeps the information readable without the figure.

## Reporting accessibility issues

Email arel.encin@mail.utoronto.ca or open a GitHub issue. It helps to include what you were trying to do, the page or file, what happened, and your browser, operating system and any assistive technology. Screenshots are optional, and you never need to share anything about a disability.

### Severity

- High: you cannot get a result or complete a submission at all.
- Medium: you can, but only with significant extra effort.
- Low: an inconvenience that does not block the task.

Maintainers assign severity; reporters do not need to.

### How we respond

We aim to acknowledge reports within one week, suggest a workaround where we can, and tell you when a fix is released so you can check it.

## Ownership and maintenance

The BrainFMBench maintainers at the EPICS Lab are responsible for accessibility. We review this document with each quarterly release.

## Supported environments

The repository and LEADERBOARD.md are rendered by GitHub and inherit its accessibility support. The interactive leaderboard at epics-lab.ca and the Hugging Face Space are built with Plotly and have not been tested with assistive technologies.

## Known limitations

- The interactive leaderboard has not been tested with screen readers or keyboard only navigation. The same results are available as text in LEADERBOARD.md.
- The overview and boxplot images have short alt text only. Their full content is in the LEADERBOARD.md tables.
- Model colours in the boxplots were chosen to match the paper and have not been checked for colour vision differences.

## Feedback and improvements

Suggestions for this document or our practices are welcome by email or as a GitHub issue. Please report active barriers through the process above.
