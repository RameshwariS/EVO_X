# Limitations (synopsis §9.3)

State these in the report; each is a property of the evidence, not a defect to hide.

1. **Synthetic drift construction.** Re-sequencing class proportions is prior/label-distribution shift. Feature
   perturbation and relabelling are synthetic concept/covariate drift. CICIDS2017's natural drift is almost entirely
   novel-class drift because attacks are scheduled by weekday. Controlled drift over real flows is not a naturally
   recorded drift event.
2. **Dataset age and artefacts.** CICIDS2017, UNSW-NB15 and CSE-CIC-IDS2018 are lab captures with known labelling and
   duplication issues. The UNSW adapter drops categorical columns.
3. **Small set of base models.** Random Forest and XGBoost only; no neural IDS. H5 (portability) is therefore evidence
   about two tree ensembles, not about model families.
4. **Threshold and weight sensitivity.** Controller weights and gate thresholds are chosen on validation scenarios from
   a pre-registered grid; the sensitivity sweeps are part of the evidence and must be read with the headline numbers.
5. **Label policy.** Adaptation uses delayed (and optionally fractional) labels for every system. Results do not
   transfer to settings with different label availability.
6. **Approximate baselines.** The SSF-inspired and online + active learning baselines are re-implementations, labelled
   "-inspired"; they are not the published systems.
7. **Cross-algorithm design.** Gate thresholds and detector settings are the target algorithm's own; only the
   controller's decision logic (weights, thresholds, scales, cost profiles) is transferred.
8. **No live deployment.** The system is replayed from stored streams. Latency figures are for this software and
   hardware. The dashboard replays finished runs; it does not monitor a live deployment.
9. **Statistics.** Six or more paired seeds are required before a p-value is reported; fewer seeds validate the
   pipeline only. Multiple scenario kinds raise a multiple-comparison concern that per-kind tables expose but do not correct.
