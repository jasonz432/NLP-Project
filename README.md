<h1 align="center" style="font-size: 3em;">
  Uncovering Bias in Simplified Text Summaries
</h1>

<p align="center">
  An analysis of semantic distortion introduced by automatic summarization systems
</p>

---

## Overview

This project examines **semantic distortion and bias introduced during automatic text summarization**, focusing on whether simplified summaries preserve the original meaning, sentiment, and intent of the source text.

Summarization systems are commonly evaluated using aggregate metrics that emphasize conciseness and readability. However, these metrics can mask **subtle but impactful meaning changes**, such as shifts in polarity, subjectivity, or emphasis.

We investigate whether simplified summaries remain faithful to the source text beyond surface-level quality measures.

---

## Methodology

Adapting the framework proposed by **Ribeiro et al.**, this project evaluates summaries along semantic dimensions rather than purely lexical similarity.

The analysis includes:

- Generating simplified summaries from original text
- Comparing summaries against source text for semantic consistency
- Measuring shifts in polarity, subjectivity, and salient content
- Qualitative analysis of recurring distortion patterns

---

## References

Ribeiro et al., Generating Summaries with Controllable Readability Levels, 2023.
