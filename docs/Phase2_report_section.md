# Draft — Chapter 7 addendum: Dataset preprocessing and time-ordered stream construction
*(fill the `[ ]` placeholders from your run; delete this line)*

## 7.x.1 Objective
Convert CICIDS2017 into a leakage-free, time-ordered stream consisting of a reference window (used to train the
initial IDS) followed by a live stream replayed in batches, as required by §7.6 (no random split).

## 7.x.2 Procedure
1. **Loading.** Day files are loaded in capture order (Mon→Fri); alphabetical order would start on Friday.
2. **Cleaning.** Header whitespace stripped; duplicated `Fwd Header Length` removed; `Infinity`/NaN values
   treated as missing; label artefact `Web Attack \x96 …` normalised; feature+label duplicates removed
   ([ ] rows → [ ] rows).
3. **Time ordering.** Timestamps are d/m/Y on a 12-hour clock without AM/PM; hours < 8 are shifted by +12 h
   (afternoon flows would otherwise sort before morning flows). Sorting is stable, so flows within the same
   minute keep file order. Ordering source: [timestamp].
4. **Split.** Reference = first [ ]% of flows ([ ] ref_train / [ ] ref_val); live stream = remaining [ ] flows,
   replayed in batches of [1000] ([ ] batches).
5. **Leakage control.** Median imputation and standardisation are fitted on ref_train only. Features constant
   over the whole dataset ([ ] removed) are dropped; features constant only in the reference are kept, since a
   change in them is a genuine drift signal.
6. **Class space.** Class ids are fixed (BENIGN = 0). Classes with no reference flows are flagged novel.

## 7.x.3 Validation
Automated checks (shapes, finiteness, label range, monotone time, reference precedes stream, scaler fitted on
reference only) all pass: [paste `phase2_stream_summary.json` validation block]. Figure 7.x: class composition
and descriptive feature shift per batch (`phase2_stream_overview.png`).

## 7.x.4 Observation and design decision
Attacks in CICIDS2017 follow a weekday schedule, so the natural stream is dominated by novel-class events
([ ]% of stream flows belong to classes absent from the reference). Table 7.x (reference sweep) shows [ ].
Decision: reference fraction = [ ] because [ ]. Drift types involving *known* classes (recurring, gradual,
incremental, temporary) are generated synthetically in Phase 5.

## 7.x.5 Threats to validity
Minute-resolution timestamps; known labelling artefacts of CICIDS2017 (§9.3); single capture week;
results are relative comparisons between methods, not absolute real-world performance.
