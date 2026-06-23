# Response to Comments

**Paper ID:** satwcon104s1
**Title:** Dynamic Urban Heat Exposure Modeling: A Trajectory-based Agent Simulation Approach for Vulnerable Population
**Authors:** Escudero Bell, Gil-Costa, and Marín

We thank the three reviewers and the track coordinators for their careful and constructive
assessment of our paper. The reviews were unanimous in recommending acceptance with minor
changes, and we have revised the manuscript accordingly. The revision preserves the original
model and scope; the main additions are (i) a new robustness-checks subsection (Section 4.5),
(ii) a statement of computational requirements (Section 3.5), and (iii) a set of clarifications
and corrections requested by the reviewers.

Below we address each comment in turn. References to sections and figures correspond to the
revised manuscript. All substantive edits are marked in the LaTeX source with `%% REV:`
comments for traceability.

---

## Summary of main changes

1. **New Section 4.5 (Robustness Checks).** Adds (a) a sensitivity analysis of the vegetation
   cooling coefficient and (b) a comparison against a complementary WBGT-based exposure
   formulation that incorporates meteorological forcing, time-of-day dynamics, and a
   physiological threshold. Both show that the paper's structural findings are not artifacts
   of the simplified metric.
2. **Computational requirements** are now reported in Section 3.5 (network size, runtime, and
   a scaling projection to the metropolitan scale).
3. **Reproducibility.** All results were regenerated with a fixed random seed; the reported
   figures and statistics now correspond exactly to a single reproducible run.
4. **Corrections for internal consistency** between the described method and the implementation
   (cooling coefficient value, destination-assignment rule) and minor language edits.

---

## Reviewer 1

> *Overall positive assessment; "Accept with Minor Changes"; minor editing needed.*

We thank Reviewer 1 for the supportive evaluation and for confirming the relevance and
originality of the trajectory-based approach. In response to the general request for minor
editing, we have corrected typographical errors (e.g., "morbidity" in Section 1), tightened
several passages, and ensured terminological consistency (heat load / heat exposure) throughout.

---

## Reviewer 2

**(R2-1) Agent behavior is highly simplified: single-trip, shortest-path, no temporal dynamics
or behavioral adaptation.**

We agree, and this is a deliberate design choice that we have made more explicit. We clarify in
Section 3 that the single-trip, shortest-path structure yields a *lower bound* on cumulative
exposure, and we expanded the Limitations (Section 5.3) to note that older adults may make
multiple daily trips, which would increase exposure beyond what the present model captures.
Importantly, the new WBGT formulation in Section 4.5 **does** incorporate time-of-day dynamics
(a diurnal WBGT profile, departure times, and dwell), and shows that the central findings are
preserved under a temporally explicit model.

**(R2-2) The heat exposure formulation is intentionally simple: no meteorological variables,
no temporal variation, no physiological model, no empirical calibration.**

We have addressed this directly with the new complementary WBGT-based model (Section 4.5),
which introduces meteorological forcing (a diurnal WBGT profile derived from temperature,
humidity, and radiation), time-of-day variation, and a physiological exposure threshold adjusted
by vulnerability. Re-simulating the same trajectories under this physically grounded model
reproduces the same qualitative structure as the NDVI-based metric (distance dominant,
vegetation a secondary moderator, vulnerable groups more exposed), with strong agreement at the
agent level (Spearman rho = 0.76). We also state explicitly in the Limitations that empirical
calibration against observed thermal or mobility data remains future work.

**(R2-3) Several findings are strongly influenced by the mathematical structure of the exposure
metric, particularly the dominant role of travel distance.**

This is an important point and we have responded to it in two ways. First, the sensitivity
analysis in Section 4.5 varies the only free coefficient of the metric, the vegetation cooling
strength beta, across {0.2, 0.5, 0.8, 1.0}. The dominance of distance (Pearson r between 0.93
and 1.00) and the overestimation by static indicators (80-93% of trajectories) hold across the
entire range, demonstrating that the conclusions are not artifacts of a particular parameter
choice. Second, we corrected the exposure formulation so that it contains **no
vulnerability-specific term**: the exposure rate now depends only on distance and route-level
vegetation. Consequently, the higher exposure of vulnerable groups (Section 4.4) is **emergent**,
arising from the spatial co-location of vulnerability and low vegetation, rather than being
imposed by the metric. This is now stated explicitly in Section 4.4.

**(R2-4) The paper would be strengthened by sensitivity analyses, scenario exploration, or
validation using observed data.**

We added a sensitivity analysis (Section 4.5, Figure on beta) and a complementary-model
comparison (Section 4.5, WBGT). Regarding empirical validation, we agree it is valuable but
beyond the scope of this conference paper; we now frame it explicitly as the principal direction
for future work in Section 5.3.

---

## Reviewer 3

**(R3-1) Insufficient information about the computational requirements of the simulation.**

We have added a dedicated statement of computational requirements in Section 3.5. We report the
size of the pedestrian network (12,535 nodes; 36,488 edges), the runtime of the full pipeline
(a few seconds on a standard desktop, with shortest-path routing accounting for ~1 s for the
simulated trips), and a linear projection to the older-adult population of Greater Santiago
(~38x larger), under which the routing stage remains well under one minute. This shows the
framework scales to metropolitan extents on commodity hardware.

**(R3-2) Insufficient comparison with other potential models available in the literature.**

We expanded the discussion of related approaches in the Introduction, situating our framework
relative to static residential exposure indices and to alternative dynamic/agent-based exposure
models. In addition, the new Section 4.5 provides a direct, quantitative comparison of our
NDVI-based metric against a WBGT-based formulation, a standard heat-stress model, on the same
case study.

**(R3-3) The work has high-impact potential if deeper models and analysis are carried out.**

The new Section 4.5 responds to this by adding a deeper, physically grounded WBGT model and a
parametric sensitivity analysis, while keeping the original, interpretable model as the paper's
core contribution. We believe this strengthens the analysis without sacrificing the
parsimony that makes the approach transferable to data-scarce contexts.

---

## Note on updated numerical values

To ensure full reproducibility, all results were regenerated from a single run with a fixed
random seed. As a result, some reported quantities differ slightly from the originally submitted
values (e.g., trip counts and mean travel distances), but all qualitative findings are unchanged
and, in several cases, strengthened. We also corrected two descriptive inconsistencies between
the method text and the implementation: the value of the cooling coefficient (beta = 0.8, the
value actually used) and the destination-assignment rule (nearest amenity of the drawn purpose),
now described as implemented.

We thank the reviewers again for their valuable feedback, which has materially improved the paper.
