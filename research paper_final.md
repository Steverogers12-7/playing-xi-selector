# Form-Aware Multi-Objective Playing XI Selection in T20 Cricket: Predict-then-Optimise with Gradient Boosting, Neural Networks and NSGA-II

**[Author names]**, Department of Mathematics and Computing, Netaji Subhas University of Technology (NSUT), Delhi

> DRAFT v2 (28 Oct 2026). Numbers in Sections 3, 4.5, 5.2, 5.3 and 5.6 come from the latest notebook run (49 India T20Is, data up to 6 Oct 2026). Items marked **[RE-RUN]** were produced on the earlier 45-match snapshot and must be regenerated on the current data before submission. Items marked **[TODO]** need your own check or literature work. References marked (as cited in [1]) must be verified against the original source. See "Revision notes" at the end (delete before submission).

---

## Abstract

Choosing a playing XI from a squad is a multi-objective decision: a selector wants strong batting and strong bowling from the same eleven players. Ahmed, Deb and Jindal [1] showed that this can be treated as a bi-objective problem solved with NSGA-II, and Verma, Pandey, Pant and Snasel [9] extended the idea to balanced IPL squads with new NSGA-II variants. Both rely on static career statistics (batting average, strike rate, bowling average) and validate the chosen teams only theoretically or through a qualitative review of player performances. We propose a *predict-then-optimise* pipeline that (i) builds leakage-free, recency-aware features for 5,134 players from 4,835 ball-by-ball T20 matches (3,592 international, 1,243 IPL), (ii) predicts each player's next-match runs and wickets with Ridge regression, XGBoost and a feed-forward neural network, (iii) converts the predictions into team batting and bowling strength, and (iv) solves the resulting constrained bi-objective problem with NSGA-II, choosing a final XI at the knee of the Pareto front. On a strict time split (test: 2025 onwards) XGBoost and the neural network reduce the mean absolute error of predicted runs from 14.42 (rolling-form baseline) to 13.78 and 13.81, with match-level bootstrap intervals excluding zero; the two models are statistically indistinguishable from each other and from Ridge. NSGA-II recovers the exact Pareto front (verified by enumerating every feasible XI) with a hypervolume ratio of 0.9999. In a backtest on 49 India T20Is played in 2025 or later, the recommended XI shares on average 8.80 of 11 players with the XI actually fielded, against 8.50 for random valid XIs and 9.24 for repeating the previous match's XI. We also report a prospective prediction for New Zealand v India (22 Oct 2026) from the announced 16-player squad, including a sensitivity analysis for players returning after a long absence. We position the method as a decision-support tool that exposes the batting-bowling trade-off, not as a predictor of selectors' choices, and we report the limitations that lead to this conclusion.

**Keywords:** cricket team selection, multi-objective optimisation, NSGA-II, XGBoost, neural network, T20, decision support

---

## 1. Introduction

Team selection in cricket combines two conflicting goals. A batting-heavy XI reduces the number of specialist bowlers, and a bowling-heavy XI shortens the batting order. Selectors resolve this trade-off with experience, and objective methods are comparatively recent. Earlier work used single-objective integer programming (Gerber and Sharp; Lourens; Warren; all as cited in [1]) and genetic algorithms for ODI selection [18]. Ahmed et al. [1] formulated IPL team selection as a constrained bi-objective problem (maximise team batting average, minimise team bowling average) under a budget, a foreign-player limit and captain and wicket-keeper constraints, solved it with a modified NSGA-II, and selected a final team using a knee region and multi-criteria non-dominated sorting. Chand et al. [10] showed that integer linear programming can handle two to five objectives (batting, bowling, fielding, cost, star power). Verma et al. [9] recast IPL *squad* selection as a 0/1 multi-objective knapsack problem, replaced batting and bowling averages by the Batting Performance Factor [11] and the Combined Bowling Rate [12], added bounds on pure batters, pure bowlers, all-rounders and star players, and proposed binary and integer NSGA-II variants (BNSGA-II, INSGA-II).

Three aspects of this literature motivate our work.

1. **Static player quality.** Players are described by career statistics. Ahmed et al. [1] guard against small samples with hand-made rules (300 runs to count as a batsman, 20 wickets to count as a bowler, a penalty of 100 for non-bowlers). Verma et al. [9] list the same issue as a limitation: a player's past record may not hold in the next matches, and in-form players should perhaps receive more weight than out-of-form players. They also list pitch conditions and home ground as factors left out.
2. **No out-of-sample evidence.** In [1] the optimised teams are compared with the 2011 Chennai Super Kings team on the same statistics, so the superiority is "theoretical". In [9] the check is a table showing that 30 of 44 frequently selected players appeared in IPL 2020 and did well on selected rankings. Neither shows that the method recovers teams that selectors really fielded.
3. **Optimiser validation.** In [9] algorithms are compared with each other (hypervolume, IGD, number of Pareto solutions, Friedman test over 40 runs), but not against the true Pareto front.

We keep the bi-objective NSGA-II structure, the constraint-domination principle and the knee-point decision rule of [1] and [9], and replace the weak parts: player scores become *learned, form-aware predictions*, the optimiser is validated against *exhaustive enumeration*, and evaluation becomes a *time-split backtest against real XIs* plus a prospective prediction.

**Contributions.**

- A leakage-free feature pipeline over 4,835 ball-by-ball T20 matches, using shrinkage towards a prior so that players with few matches are handled without threshold rules.
- A statistically tested comparison of a rolling-form baseline, Ridge, XGBoost and a neural network for predicting next-match runs and wickets (match-level cluster bootstrap).
- A predict-then-optimise selector: NSGA-II over 11-player subsets, validated against exhaustive enumeration, with knee-point decision making and a player-frequency analysis.
- An honest backtest on 49 India T20Is, including negative findings: the previous match's XI is the strongest baseline in point estimate, rolling-form scores do as well as learned scores, and opposition and venue features change the chosen XI very little.
- A prospective prediction for a future match from the announced squad, with a sensitivity analysis for players returning after a long gap.

**[TODO]** Add 2 to 4 references on ML *prediction* of player or match performance (not selection) published after 2013. Selection-related ML work is now covered by [14] and [15].

---

## 2. Related Work

Ahmed et al. [1] and Verma et al. [9] are the closest works; Table 1 compares them with ours.

*Optimisation approaches.* Gerber and Sharp, Lourens and Warren (as cited in [1]) use integer programming with one weighted objective, which hides the trade-off. Bhattacharjee and Saikia [13] use binary integer programming for a balanced playing XI with role constraints. Chand et al. [10] guarantee optimality with integer linear programming in two to five objectives. Sathya and Jamal [18] apply a genetic algorithm to choose eleven of fifty ODI players. Data envelopment analysis has also been used [21]. Saikia and Bhattacharjee (as cited in [1]) classify IPL all-rounders using strike rate and economy rate, which supports using several conflicting criteria.

*NSGA-II based selection.* Ahmed et al. [1] introduced a code-vector and variable-vector representation that keeps captain and wicket-keeper in fixed positions and handles budget and foreign-player limits through constraint domination; fielding is a secondary criterion used for tie-breaking in tournament selection. Verma et al. [9] use a plain binary inclusion vector with a *repair* step that restores the number of selected players, which is also what we use, and show that it gives better hypervolume and IGD than the representation of [1] when extended to a 23-player squad. Their balanced-squad model also bounds the number of pure batters, bowlers and all-rounders so that trade-off squads are not filled with all-rounders. Our role constraints are simpler (one keeper, at least five bowling options) because Cricsheet has no role field.

*Player-quality measures.* [9] adopt the Batting Performance Factor `SR^a * avg^(1-a)` of Barr and Kantor [11] with a = 0.6 and the Combined Bowling Rate of Lemmer [12]. Both are still functions of career totals. We instead learn the expected next-match output from recent form.

*Machine learning in selection.* Ghosh et al. [14] and Prakash et al. [15] use machine-learning models for IPL player selection, and [9] cite them as the ML side of the literature. **[TODO]** read both and add one sentence each on what they predict and how they are evaluated; we have not described their methods here.

*Algorithms and metrics.* NSGA-II [2] is the standard elitist multi-objective evolutionary algorithm, a review of its use on combinatorial problems is in [16], and knee points as preferred solutions are analysed in [3]. Gradient boosting [4] is a strong default for tabular data, which is why we compare it with a neural network.

**Table 1. Comparison with Ahmed et al. [1] and Verma et al. [9].**

| Aspect | Ahmed et al. [1] | Verma et al. [9] | This work |
|---|---|---|---|
| Task | Playing XI (11 of 129 auctioned players) | Balanced squad (23 of 185 auctioned players) | Playing XI from a squad pool of about 14 to 16 players |
| Data | International T20 career averages, IPL-4 players | Franchise leagues and T20I up to Aug 2020 (bigbashboard.com), IPL 2020 players | 5,134 players, 4,835 ball-by-ball matches (international and IPL), 2005 to 2026 (Cricsheet) |
| Player score | Batting average, bowling average, threshold rules | Batting Performance Factor and Combined Bowling Rate, career based | Learned prediction of next-match runs and wickets from recent form, shrinkage for small samples |
| Objectives | Team batting average (max), team bowling average (min) | Sum of BPF (max), sum of CBR (min) | Expected team batting strength (max), expected team bowling strength (max) |
| Constraints | Captain, keeper, at most 4 foreign players, budget | Budget, at most 8 overseas, captain, keeper, bounds on pure batters, pure bowlers, all-rounders, star players | 11 players, at least 1 keeper, at least 5 bowling options |
| Representation | Code-vector and variable-vector | Binary (BNSGA-II) and integer (INSGA-II), repair mechanism | Binary inclusion vector with repair to exactly 11 players |
| Algorithm settings | SBX, polynomial mutation, population 400, 400 generations | Two-point crossover and swap mutation (binary), population 200, 500 generations | pymoo NSGA-II, uniform crossover, bit-flip mutation, population 80, 60 generations |
| Decision making | Knee region, then multi-criteria non-dominated sorting | Frequency of players on the trade-off squads | Knee point, frequency of players on the front |
| Optimiser validation | Not against the true front | Comparison with 4 other algorithms (hypervolume, IGD, NPS, Friedman test, 40 runs) | Exhaustive enumeration of every feasible XI (hypervolume ratio to the exact front) |
| Validation of selected teams | Compared with the 2011 CSK team on the same statistics | Review of the players' IPL 2020 records | Time-split backtest on 49 India T20Is against real XIs and four baselines, bootstrap CIs; prospective prediction |

---

## 3. Data

**Source.** Ball-by-ball JSON files from Cricsheet [6]: `t20s_male_json` (3,592 men's international T20 matches) and `ipl_male_json` (1,243 IPL matches), 4,835 matches in total. The latest India T20I in the data is 6 Oct 2026 (West Indies tour of India). Cricsheet withholds men's Afghanistan matches and the Afghanistan Premier League, so these are absent.

**Parsing rules.** Each match is converted into one row per player who was named in the XI (106,942 rows; 79,033 international and 27,909 IPL; 5,134 distinct players, identified by the Cricsheet registry identifier because names are abbreviated and not unique). Only men's matches with a listed player set are kept.

- *Batting:* runs, balls faced (wides excluded), fours, sixes, dot balls, dismissal, batting position (order of arrival at the crease), balls faced in the powerplay (overs 1 to 6) and in the death overs (overs 16 to 20). A player counts as having batted if he reached the crease, which separates a duck from not batting.
- *Bowling:* legal balls, runs conceded (byes and leg-byes excluded), wickets (bowled, caught, caught and bowled, lbw, stumped, hit wicket; run-outs are not credited to the bowler), dot balls, wides, no-balls, powerplay and death balls.
- *Fielding:* catches, stumpings and run-outs from the fielder lists; substitute fielders are ignored.
- *Context:* date, venue, team, opposition, innings, toss and team totals. Super overs are excluded.

A spot check against an independently prepared dataset reproduced the same innings for the same player (100 runs off 47 balls, 7 fours, 8 sixes).

**Split.** All evaluation uses time: training before 2024-01-01, validation in 2024 (used only to tune XGBoost), test from 2025-01-01. For the prediction experiments (Section 5.1) we keep players with at least 15 T20 appearances; the batting task has 50,498 training-plus-validation and 14,051 test rows, the bowling task 37,352 and 10,188 **[RE-RUN on the current data]**. Runs are modelled only for innings in which the player batted, wickets only for innings in which he bowled.

---

## 4. Methodology

Figure 1 shows the full pipeline.

![Figure 1. End-to-end pipeline](paper_figs/fig1_pipeline.png)

### 4.1 Leakage-free features

For player *i* and match *t*, every feature is computed from matches strictly before *t* (a one-step shift inside each player's history), so a match never sees its own outcome.

- **Rolling statistics:** means of the batting and bowling quantities over the last 5 matches, the last 15 matches, and the whole career, using only matches in which the player batted (or bowled). Matches from both international cricket and the IPL are pooled; the feature `is_intl` marks the source of the match being predicted.
- **Shrinkage:** to stabilise small samples we use

  `mu_hat = (S + K * mu_0) / (N + K)`, with K = 5,

  where S is the sum of the statistic over the window, N the number of qualifying matches and mu_0 the mean over all training data (before 2024). A debutant therefore starts at the population mean and moves towards his own record as N grows; no hand-made thresholds are needed.
- **Usage:** `p_bat_15` and `p_bowl_15`, the share of the last 15 matches in which the player batted or bowled; fielding rates; matches played; days since the previous match (any T20 in the data).
- **Context:** the venue's mean team total (shrunk to the global mean with K = 5), the team's batting and runs-conceded form, and the opposition's runs-conceded and batting form over their last 20 matches (all past-only, filled with the global mean when missing).
- **As-of rows:** one artificial row per player dated in the far future (2100-01-01, with zero match statistics) gives the features of any player at any date, including players who did not play that match. The selector needs this for squad members who are not in the actual XI, and for future matches. The match-level context for a future match (venue, opposition) is built separately from the last 20 matches of each team and all innings at the venue.

### 4.2 Prediction models

Two regression targets: runs (given the player batted) and wickets (given he bowled). All models use the same features.

- **Baseline:** the shrunk rolling 15-match mean of the target.
- **Ridge regression** (alpha = 50) on standardised features.
- **XGBoost** [4] with Poisson objective (both targets are non-negative and skewed counts), grid search over depth {2, 3, 4}, learning rate {0.03, 0.08} and minimum child weight {10, 30} with early stopping on the 2024 validation year; the comparison models average 5 seeds.
- **Neural network (ANN):** a feed-forward network with hidden layers (64, 32), L2 penalty 0.1, early stopping on 15% of the training data, averaged over 5 seeds (scikit-learn [7]).

Evaluation metrics are MAE, RMSE and Spearman rank correlation on the test period. Uncertainty of the difference between two models is estimated with a **match-level cluster bootstrap** (400 resamples of matches, since the players of one match are not independent).

**Scoring models used for selection.** The selector uses XGBoost (Poisson, subsample 0.8, column sample 0.7, lambda 5, 3 seeds) trained on all rows before 2025-01-01 and not restricted to players with 15 appearances: batting depth 4, learning rate 0.03, min child weight 30, 386 trees; bowling depth 2, learning rate 0.03, min child weight 30, 717 trees. Toss and innings are not features because the XI is named before the toss.

### 4.3 Team objectives

For a squad pool P and a team T ⊂ P with |T| = 11, using predictions for the match being selected:

```
f1(T) = sum over i in T of  p_bat_i  *  E[runs_i | batted]        (maximise)
f2(T) = sum over i in T of  p_bowl_i *  E[wickets_i | bowled]     (maximise)
```

Weighting by usage probability avoids counting the runs of a specialist bowler who rarely bats. Interpreted literally, f1 is the expected runs scored by the XI off the bat and f2 the expected wickets taken by the XI. As in [1], fielding is a secondary criterion and is not optimised here. Unlike [1] and [9], both objectives are maximised on the same scale of "expected output", so no penalty value for non-bowlers is needed.

Constraints: exactly 11 distinct players; at least one wicket-keeper (career stumpings at least 2, because Cricsheet has no role field); at least five bowling options (`p_bowl_15` at least 0.35). Budget, foreign-player and captaincy constraints of [1] and [9] do not apply to national-team selection.

### 4.4 NSGA-II and decision making

A team is a binary vector over the pool. The initial population contains random 11-subsets; uniform crossover (probability 0.9) and bit-flip mutation are followed by a **repair** step that adds or removes random players until exactly 11 remain, so the cardinality constraint never fails (the same idea as the repair mechanism of [9]). The two role constraints are handled as inequality constraints with NSGA-II's constraint-domination rule [2]. Population 80, 60 generations, implemented with pymoo [5].

Because the pool is small (about 15 players, so at most a few thousand feasible teams), we also compute the **exact** Pareto front by enumerating every combination. This gives a ground truth for the optimiser: we report the *hypervolume ratio* (NSGA-II front divided by exact front, reference point at the origin).

From the front we choose the **knee point**: after normalising both objectives to [0, 1], the solution with the largest distance to the straight line joining the two extreme solutions [3].

### 4.5 Backtest protocol

For every India T20I from 2025-01-01 (49 matches evaluated, none skipped):

1. **Pool:** all India players who appeared in an India T20I of the same series or tournament within 45 days of the match (mean pool size 14.4; matches with a pool outside 11 to 24 players are skipped). This is a proxy for the announced squad.
2. Compute each pool player's as-of features and the match context (venue, opposition), predict, form f1 and f2, run NSGA-II, take the knee XI.
3. Compare with the XI actually fielded by the number of common players (0 to 11).

Baselines: (a) the same optimiser using the rolling-form scores instead of the learned predictions; (b) the previous India match's XI (persistence); (c) the eleven most frequently selected players over India's last 10 matches; (d) a random valid XI from the pool (300 draws per match). Confidence intervals are bootstrap percentile intervals over matches (2,000 resamples); they move by about 0.02 to 0.03 with the random seed.

---

## 5. Results

### 5.1 Prediction accuracy (test: 2025 onwards) **[RE-RUN]**

![Figure 2. Model comparison](paper_figs/fig2_model_comparison.png)

**Table 2. Batting: predicting runs when the player batted (14,051 test innings).**

| Model | MAE | RMSE | Spearman |
|---|---|---|---|
| Global mean | 15.428 | 20.646 | n/a |
| Rolling-15 baseline | 14.418 | 19.500 | 0.339 |
| Ridge | 13.820 | 19.055 | 0.384 |
| XGBoost | **13.776** | 19.049 | **0.386** |
| ANN (64, 32) | 13.809 | **19.030** | 0.384 |

**Table 3. Bowling: predicting wickets when the player bowled (10,188 test innings).**

| Model | MAE | RMSE | Spearman |
|---|---|---|---|
| Global mean | 0.803 | 1.073 | n/a |
| Rolling-15 baseline | 0.853 | 1.077 | 0.111 |
| Ridge | 0.822 | 1.058 | 0.171 |
| XGBoost | 0.823 | **1.057** | **0.173** |
| ANN (64, 32) | 0.830 | 1.059 | 0.163 |

**Table 4. Improvement over the rolling-form baseline (MAE difference, 95% match-level bootstrap CI).**

| Comparison | Batting | Bowling |
|---|---|---|
| XGBoost minus baseline | [-0.710, -0.574] | [-0.035, -0.027] |
| ANN minus baseline | [-0.683, -0.539] | [-0.028, -0.017] |

Findings. (i) Both learned models improve significantly on the baseline, since the intervals exclude zero. (ii) XGBoost, the neural network and Ridge are within about 0.05 runs of each other, so the signal is largely linear in the engineered features; we do not claim that either XGBoost or the ANN is better. (iii) For wickets, MAE is misleading: a constant prediction of the training mean has a lower MAE (0.803) than any model because counts are dominated by zeros and ones, while RMSE and rank correlation, which are more informative for counts, favour the models. (iv) The largest XGBoost importances for batting are the recent batting-position history and career runs; for bowling they are the career dot-ball count, career wickets and recent bowling usage.

*Sanity check of the selection models.* The scoring models of Section 4.2 are trained on all players (not only those with 15 appearances) and tested on all innings from 2025 onwards. On this broader population the mean absolute error of predicted runs falls from 13.772 (rolling-15 baseline) to 12.879 (-6.5%), and that of predicted wickets from 0.844 to 0.816 (-3.3%). The numbers differ from Tables 2 and 3 because the population is different, but the direction and size of the gain agree.

*Preliminary experiment.* On a first dataset of only 34 India players (1,498 innings) no model beat a rolling-form baseline (runs MAE 16.92 for the ANN against 16.85 for the baseline). Enlarging the data to thousands of players is what produced the significant gains above, which suggests that data volume, not model complexity, is the bottleneck.

### 5.2 Validity of the optimiser

Across the 49 selection problems, the NSGA-II front reaches a mean hypervolume ratio of 0.9999 (minimum 0.9958) relative to the exact front, and the knee XI obtained from NSGA-II and from the exact front share 8.80 vs 8.78 players with the real XI on average. NSGA-II is therefore a reliable solver on this problem size, which is smaller than the 23-of-185 squad problem of [9] where exhaustive search is impossible; its advantage over enumeration would appear for larger pools (for example, all Indian players active in the last 12 months, or squad selection as in [9]), which we leave to future work.

### 5.3 Backtest against real India XIs

![Figure 5. Overlap with the real XI](paper_figs/fig5_backtest_overlap.png)

**Table 5. Mean overlap with the XI actually fielded (out of 11), 49 India T20Is, 2025 onwards.**

| Method | Mean overlap | 95% CI |
|---|---|---|
| NSGA-II knee XI (learned scores) | 8.80 | [8.47, 9.12] |
| Exact-front knee XI (learned scores) | 8.78 | [8.45, 9.12] |
| Knee XI with rolling-form scores (no learning) | 8.92 | [8.61, 9.22] |
| Previous match's XI (persistence) | **9.24** | [8.90, 9.59] |
| Top 11 by appearances in last 10 matches | 8.20 | [7.82, 8.59] |
| Random valid XI from the pool | 8.50 | [8.26, 8.76] |

Findings. (i) The learned-score XI is only about 0.3 players above a random valid XI, because the pool is small (about 14 players, from which 11 are chosen), so any method overlaps heavily; the two intervals overlap. (ii) Replacing the learned scores by rolling-form scores does not improve the result (point difference -0.12 players): better *prediction accuracy* did not translate into better *reproduction of selectors' choices*. (iii) Repeating the previous XI has the highest mean overlap (9.24, a gap of 0.44 players to NSGA-II). Real selection is dominated by continuity, fitness, rotation and tactical roles that are not in our data. The unpaired intervals of persistence and NSGA-II overlap, so a paired match-level bootstrap is needed before calling this difference significant **[TODO: run paired CIs for NSGA-II vs persistence, NSGA-II vs form, NSGA-II vs random]**.

In 65% of matches the real XI is dominated by our predicted Pareto front, i.e. some feasible XI has at least as high predicted batting *and* bowling strength. This statement is about *predicted* strength, not about real match outcomes, and should not be read as evidence that the XI we propose would have performed better. For comparison, [1] reports the same kind of statement for CSK. In 92% of matches the real XI satisfies our two role constraints, so the constraints are slightly tighter than real practice.

### 5.4 Case studies

**(a) India tour of Zimbabwe, 26 July 2026.** The pool contained 13 players and 51 feasible XIs, of which 9 are Pareto-optimal (the exact front and the NSGA-II front coincide). **[RE-RUN: same case on the current model if the scores changed]**

![Figure 3. Trade-off front](paper_figs/fig3_pareto_front.png)

**Table 6. Selected XIs on the front (names as in Cricsheet).**

| | Bowling-extreme | **Knee** | Batting-extreme | Real XI |
|---|---|---|---|---|
| Batting strength f1 | 137.55 | 151.24 | 164.48 | 161.60 |
| Bowling strength f2 | 6.310 | 6.184 | 4.929 | 4.768 |
| Players | V Suryavanshi, Y Thakur, Ashok Sharma, Ishan Kishan, Prince Yadav, Suryansh Shedge, S Dube, Tilak Varma, MP Yadav, R Bishnoi, Abhishek Sharma | V Suryavanshi, Y Thakur, Ashok Sharma, Ishan Kishan, Prince Yadav, SS Iyer, S Dube, Tilak Varma, MP Yadav, R Bishnoi, Abhishek Sharma | RK Singh, V Suryavanshi, Ashok Sharma, Ishan Kishan, Prince Yadav, SS Iyer, S Dube, Tilak Varma, MP Yadav, R Bishnoi, Abhishek Sharma | RK Singh, V Suryavanshi, Y Thakur, Ashok Sharma, Ishan Kishan, SS Iyer, Suryansh Shedge, Tilak Varma, MP Yadav, R Bishnoi, Abhishek Sharma |

The knee XI shares 9 of 11 players with the real XI (it selects S Dube and Prince Yadav where the selectors chose RK Singh and Suryansh Shedge). The real XI is dominated by the batting-extreme team of the front (higher f1 and higher f2). Moving from the knee to the batting extreme gains about 13 units of batting strength and loses about 1.25 units of bowling strength, which quantifies the trade-off a selector faces.

**Player frequency on the front.** Following the "frequency of players on the trade-off front" idea of [1] (also used in [9]), six of the 13 players (V Suryavanshi, Ishan Kishan, Ashok Sharma, S Dube, Prince Yadav, Abhishek Sharma) appear in every Pareto-optimal team and can be regarded as the core of the squad; SS Iyer and Tilak Varma appear in 89% of the teams and RK Singh in 33% (Figure 4). In [1] only 35 of 129 players appeared on the front; with a pool of 13 all players appear at least once.

![Figure 4. Player frequency on the front](paper_figs/fig4_player_frequency.png)

**(b) Most recent match in the data: West Indies tour of India, 6 Oct 2026.** The knee XI (Arshdeep Singh, AR Patel, Ishan Kishan, SS Iyer, Kuldeep Yadav, SV Samson, S Dube, Tilak Varma, MP Yadav, Abhishek Sharma, Naman Dhir) is identical to the XI fielded (11 of 11). We report it because it is the latest match, not because it is typical: the mean overlap over 49 matches is 8.80, and with a pool of about 14 players a perfect overlap is not unusual.

### 5.5 Sensitivity to opposition and venue **[RE-RUN on 49 matches]**

To test whether the chosen XI adapts to conditions, we re-ran the selection for the 45 matches of the earlier snapshot with the opposition and venue features set to their 10th percentile (low-scoring conditions) and to their 90th percentile (high-scoring conditions). On average only **0.36 of 11 players changed**, the XI was identical in 64% of matches, and the rank correlation of player scores across the two settings was 0.999. Opposition and venue currently enter only as team-level quantities common to all players, so they shift every score in the same direction. Player-specific features (a bowler's record against a given opposition, a batsman's record at a ground) are required to make the selector genuinely conditions-aware. This agrees with the limitation noted in [9], where pitch conditions and home ground were left out of the model.

### 5.6 Prospective prediction: New Zealand v India, 1st T20I, 22 Oct 2026, Hagley Oval

To avoid the pool-proxy problem of the backtest, we ran the selector on the announced 16-player India squad. **[TODO: confirm squad against an official source and cite it.]**

*Setup.* Pool: SS Iyer, Tilak Varma, Abhishek Sharma, V Suryavanshi, Ishan Kishan, SV Samson, S Dube, Kuldeep Yadav, AR Patel, Nithish Kumar Reddy, Arshdeep Singh, Prince Yadav, MP Yadav, Naman Dhir, B Kumar, CV Varun (Cricsheet names). Features are as-of 22 Oct 2026. Match context: venue average total 152.1 (29 innings at the ground, shrunk with K = 5), India batting form 175.2, India runs conceded 165.3, New Zealand runs conceded 168.3, New Zealand batting form 165.2. There are 3,878 feasible XIs, 70 of them Pareto-optimal (exact front; NSGA-II front size agrees).

**Table 7. Predicted contributions of the pool (as-of 22 Oct 2026).**

| Player | Days since last India T20I | p_bat | E[runs given batted] | p_bowl | E[wickets given bowled] |
|---|---|---|---|---|---|
| SS Iyer | 16 | 1.00 | 29.3 | 0.00 | 0.54 |
| Tilak Varma | 16 | 0.93 | 21.7 | 0.07 | 0.45 |
| Abhishek Sharma | 16 | 1.00 | 27.6 | 0.27 | 0.45 |
| V Suryavanshi | 19 | 1.00 | 32.1 | 0.00 | 0.66 |
| Ishan Kishan | 16 | 1.00 | 27.3 | 0.00 | 0.55 |
| SV Samson | 16 | 1.00 | 27.2 | 0.00 | 0.66 |
| S Dube | 16 | 0.87 | 16.3 | 0.67 | 0.61 |
| Kuldeep Yadav | 16 | 0.33 | 3.7 | 0.93 | 1.16 |
| AR Patel | 16 | 0.87 | 13.1 | 0.87 | 0.93 |
| Nithish Kumar Reddy | **638** | 0.87 | 20.1 | 1.00 | 0.70 |
| Arshdeep Singh | 16 | 0.33 | 3.8 | 0.87 | 1.13 |
| Prince Yadav | 89 | 0.20 | 3.1 | 1.00 | 1.15 |
| MP Yadav | 16 | 0.13 | 4.2 | 1.00 | 1.18 |
| Naman Dhir | 16 | 0.93 | 24.8 | 0.07 | 0.59 |
| B Kumar | **1,430** | 0.27 | 3.5 | 1.00 | 1.20 |
| CV Varun | 107 | 0.20 | 3.2 | 0.93 | 1.13 |

**Table 8. Selected XIs on the front (full pool).**

| | Bowling-extreme | **Knee** | Batting-extreme |
|---|---|---|---|
| Batting strength f1 | 103.05 | 175.61 | 211.99 |
| Bowling strength f2 | 8.67 | 6.23 | 4.13 |
| Players | Abhishek Sharma, Ishan Kishan, S Dube, Kuldeep Yadav, AR Patel, Nithish Kumar Reddy, Arshdeep Singh, Prince Yadav, MP Yadav, B Kumar, CV Varun | SS Iyer, Abhishek Sharma, V Suryavanshi, Ishan Kishan, SV Samson, Kuldeep Yadav, AR Patel, Nithish Kumar Reddy, Prince Yadav, MP Yadav, B Kumar | SS Iyer, Abhishek Sharma, V Suryavanshi, Ishan Kishan, SV Samson, S Dube, Kuldeep Yadav, AR Patel, Nithish Kumar Reddy, Arshdeep Singh, Naman Dhir |

*Frequency on the front.* Ishan Kishan and Nithish Kumar Reddy appear in 100% of the Pareto-optimal XIs, B Kumar in 99%, V Suryavanshi 97%, Abhishek Sharma 93%, SS Iyer 87%, Kuldeep Yadav 79%, AR Patel and Prince Yadav 67%, MP Yadav 64%, S Dube 60%, SV Samson 57%, Arshdeep Singh 50%, CV Varun 43%, Naman Dhir 34%, Tilak Varma 3%.

*Returning players: a honest sensitivity analysis.* Two squad members have not played an India T20I for a long time (Nithish Kumar Reddy 638 days, B Kumar 1,430 days), but their as-of features are strong because they pool recent IPL matches (both played a T20 within 150 days) and their bowling usage is 1.0. Because the model has no information on whether selectors will pick them, we repeated the selection without them. The pool then has 14 players, 317 feasible XIs and 28 Pareto-optimal XIs. The knee XI becomes SS Iyer, Abhishek Sharma, V Suryavanshi, Ishan Kishan, SV Samson, Kuldeep Yadav, AR Patel, Arshdeep Singh, Prince Yadav, MP Yadav, CV Varun (f1 = 159.17, f2 = 6.37). It shares 9 of 11 players with the full-pool knee XI: the two players added are Arshdeep Singh and CV Varun, replacing B Kumar and Nithish Kumar Reddy. The conclusion is that the recommendation is robust except for these two places, and that those two places depend on a selection assumption outside the model.

*Comparison with the previous XI.* The full-pool knee XI shares 7 of 11 players with the XI of the previous India match (6 Oct 2026). Tilak Varma, Naman Dhir, S Dube and Arshdeep Singh, who played on 6 Oct, are replaced by V Suryavanshi, Nithish Kumar Reddy, Prince Yadav and B Kumar. Given Section 5.3, this is a large departure from continuity and should be treated as a hypothesis to be tested, not as a recommendation.

*Test after the match.* The prediction (full, no-returning and previous-XI variants) was saved to `pred_nz_t20i_1_honest.csv`. **[TODO: insert the real XI of 22 Oct 2026 and report overlap of (a) the full-pool knee XI, (b) the no-returning knee XI and (c) the previous XI. Confirm that the file was generated before the match.]** A single match cannot confirm or refute the method, so this result should be reported together with Section 5.3 and not instead of it.

---

## 6. Discussion and Limitations

- **Pool construction.** The backtest proxy uses players who appeared later in the same series, which mildly favours all methods (visible in the 8.50 random baseline). The prospective test in Section 5.6 uses an announced squad, but it is a single match. A larger test would use official squads for all matches, or all players active in the previous 12 months, and there NSGA-II would be needed rather than enumeration.
- **Returning players.** Players with a long gap in India T20Is receive features computed mainly from IPL matches. Their predictions may not transfer to international cricket, and selectors may not pick them. Section 5.6 quantifies the effect for one match; a general treatment (an international-specific form feature or a selection-probability model) is left to future work.
- **Circularity of usage.** `p_bat_15` and `p_bowl_15` are derived from past selections and batting orders, so the model partly inherits selectors' habits; this makes the objectives realistic but blurs the line between performance and selection.
- **Proxy constraints.** Wicket-keeper status is inferred from stumpings, and "bowling option" from recent bowling share. Cricsheet contains no role, fitness, injury or captaincy data, and no budget or price data, so the squad-level constraints of [1] and [9] (budget, overseas players, star players, balanced roles) are not modelled.
- **Coverage.** Only international and IPL matches are included; Afghanistan matches are withheld by the source; the backtest covers one team and 49 matches.
- **Metric.** Overlap with the real XI measures agreement with selectors, not team quality. Predicted strength is not validated against match results.
- **Objective simplicity.** Two additive objectives ignore batting order interactions, overs distribution among bowlers, fielding and captaincy; they also do not reward strike rate explicitly, whereas [9] argue that strike rate matters in T20 (their BPF).
- **Tuning.** Hyperparameters were tuned on 2024 only and fixed for 2025+, which is deliberately conservative but may leave some accuracy unused.

---

## 7. Conclusion and Future Work

We replaced the static career averages of established NSGA-II team-selection models [1, 9] with learned, form-aware predictions and evaluated the result out of sample. The learned predictions are significantly more accurate than a rolling-form baseline, the optimiser matches the exact Pareto front, and the framework exposes the batting-bowling trade-off in a form usable by a selector (front, knee XI, player frequencies, sensitivity to uncertain squad members). The backtest, however, shows that real XIs are reproduced at least as well by continuity as by optimisation, that better prediction accuracy did not improve agreement with selectors, and that opposition and venue currently have almost no influence. We therefore present the system as decision support.

Future work: (1) a third objective or constraint given by a learned *probability of selection* trained on past India XIs, combining performance with selector behaviour; (2) player-versus-opposition and player-at-venue features with shrinkage; (3) larger pools and squad-level selection with budget and role bounds as in [9], where evolutionary search is necessary, including their BPF and CBR as alternative scores and baselines; (4) ODI and Test formats with format-specific models, since performance does not transfer across formats [1]; (5) validation against match outcomes; (6) paired significance tests and a multi-team, multi-season prospective evaluation.

---

## References

1. F. Ahmed, K. Deb, A. Jindal. Multi-objective optimization and decision making approaches to cricket team selection. *Applied Soft Computing* 13(1), 402-414, 2013.
2. K. Deb, A. Pratap, S. Agarwal, T. Meyarivan. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation* 6(2), 182-197, 2002.
3. K. Deb, S. Gupta. Understanding knee points in bicriteria problems and their implications as preferred solution principles. *Engineering Optimization* 43(11), 1175-1204, 2011.
4. T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. *Proc. ACM SIGKDD*, 2016.
5. J. Blank, K. Deb. pymoo: multi-objective optimization in Python. *IEEE Access* 8, 89497-89509, 2020.
6. Cricsheet: ball-by-ball cricket data. https://cricsheet.org (data accessed for this study).
7. F. Pedregosa et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research* 12, 2825-2830, 2011.
8. K. Deb. *Multi-objective Optimization using Evolutionary Algorithms*. Wiley, 2001.
9. S. Verma, V. Pandey, M. Pant, V. Snasel. A balanced squad for Indian Premier League using modified NSGA-II. *IEEE Access* 10, 100463-100477, 2022. doi: 10.1109/ACCESS.2022.3204649.
10. S. Chand, H. K. Singh, T. Ray. Team selection using multi-/many-objective optimization with integer linear programming. *Proc. IEEE Congress on Evolutionary Computation (CEC)*, 2018, pp. 1-8. doi: 10.1109/CEC.2018.8477945.
11. G. D. I. Barr, B. S. Kantor. A criterion for comparing and selecting batsmen in limited overs cricket. *Journal of the Operational Research Society* 55(12), 1266-1274, 2004.
12. H. H. Lemmer. The combined bowling rate as a measure of bowling performance in cricket. *South African Journal for Research in Sport, Physical Education and Recreation* 24(2), 37-44, 2002.
13. D. Bhattacharjee, H. Saikia. An objective approach of balanced cricket team selection using binary integer programming method. *OPSEARCH* 53(2), 225-247, 2016.
14. A. Ghosh, A. Sinha, P. Mondal, A. Roy, P. Saha. Indian premier league player selection model based on Indian domestic league performance. *Proc. IEEE 11th Annual Computing and Communication Workshop and Conference (CCWC)*, 2021, pp. 277-282.
15. C. D. Prakash, C. Patvardhan, C. V. Lakshmi. Team selection strategy in IPL 9 using random forests algorithm. *International Journal of Computer Applications* 139(12), 1-7, 2016. (Author list as printed in [9] is duplicated; verify.)
16. S. Verma, M. Pant, V. Snasel. A comprehensive review on NSGA-II for multi-objective combinatorial optimization problems. *IEEE Access* 9, 57757-57791, 2021.
17. H. Gerber, G. D. Sharp. Selecting a limited overs cricket squad using an integer programming model. *South African Journal for Research in Sport, Physical Education and Recreation*, 2006 (as cited in [1]); M. Lourens. *Integer optimisation for the selection of a Twenty20 cricket team*. Master's thesis, Nelson Mandela Metropolitan University, 2009; B. Warren. *Integer optimisation for the selection of a fantasy league cricket team*. Master's thesis, Nelson Mandela Metropolitan University, 2010; H. Saikia, D. Bhattacharjee. On classification of all-rounders of the Indian Premier League (IPL): a Bayesian approach. *Vikalpa* 36(4), 51-66, 2011. (Details as printed in [1]; verify volume and pages.)
18. S. S. Sathya, M. S. Jamal. Applying genetic algorithm to select an optimal cricket team. *Proc. Int. Conf. on Advances in Computing, Communication and Control*, 2009, pp. 43-47.
19. A. W. Iorio, X. Li; D. E. Goldberg; J. Derrac et al. (algorithm and test references used in [9]; cite only if the Friedman test or related methods are added to this paper).
20. F. Ahmed, A. Jindal, K. Deb. Cricket team selection using evolutionary multi-objective optimization. *Proc. 2nd Int. Conf. Swarm, Evolutionary and Memetic Computing*, LNCS 7077, 2011, pp. 71-78. (Conference version of [1].)
21. G. R. Amin, S. K. Sharma. Cricket team selection using data envelopment analysis. *European Journal of Sport Science* 14(1), S369-S376, 2014.

---

## Appendix A. Reproducibility

**Software.** Python 3, pandas, NumPy, SciPy, scikit-learn, XGBoost, pymoo, tqdm. Notebook cells: parsing (JSON to player-match table), feature construction (including as-of rows), selection and backtest (final models, NSGA-II, exact front, bootstrap), results tables, prospective prediction (squad pool, returning-player sensitivity). Scripts for the model comparison (`model2.py`) are separate.

**Sizes.** Player-match table 106,942 x 39 (79,033 international rows, 27,909 IPL rows; 4,835 matches; 5,134 players); with features and 5,134 as-of rows 112,076 x 118. Modelling subset (players with at least 15 appearances): **[RE-RUN: re-count on the current data]**.

**Hyperparameters.**

| Component | Setting |
|---|---|
| Shrinkage | K = 5, prior = training mean before 2024 |
| Rolling windows | last 5, last 15, career |
| Team context | last 20 matches; venue mean shrunk with K = 5; global mean for missing values |
| XGBoost (batting, model comparison) | Poisson, depth 4, learning rate 0.03, min child weight 30, 351 trees, subsample 0.8, column sample 0.7, lambda 5 **[RE-RUN]** |
| XGBoost (bowling, model comparison) | Poisson, depth 2, learning rate 0.03, min child weight 30, 652 trees, same sampling **[RE-RUN]** |
| XGBoost for selection | trained on all players before 2025, no toss/innings features, 3 seeds (386 trees batting, 717 trees bowling) |
| ANN | MLP (64, 32), alpha 0.1, early stopping (15% validation, patience 15), up to 400 iterations, 5 seeds |
| Ridge | alpha 50 |
| NSGA-II | population 80, 60 generations, uniform crossover 0.9, bit-flip mutation (per-variable probability 2/n), repair to 11 players, duplicates eliminated |
| Constraints | 11 players, at least 1 keeper (career stumpings >= 2), at least 5 bowlers (`p_bowl_15` >= 0.35) |
| Decision rule | knee point: maximum distance from the line joining the normalised extreme solutions |
| Backtest pool | India players in the same series or tournament within 45 days; 11 to 24 players |
| Prospective pool | announced 16-player squad; "returning" player = no India T20I for more than 365 days |
| Bootstrap | 400 match-level resamples (prediction CIs), 2,000 match resamples (backtest CIs), 300 random XIs per match for the random baseline |

---

## Revision notes (delete before submission)

**What changed from draft v1.**
- Data: 5,108 to 5,134 players, 4,801 to 4,835 matches, 3,558 to 3,592 international files, rows 106,194 to 106,942.
- Backtest: 45 to 49 matches. NSGA-II overlap 8.60 to 8.80, exact 8.58 to 8.78, form 8.73 to 8.92, persistence 9.33 to 9.24, last-10 frequency 8.18 to 8.20, random 8.27 to 8.50. Hypervolume mean 0.9999 (min 0.9958). Real XI dominated 71% to 65%; real XI feasible 91% to 92%. Mean pool 14.7 to 14.4.
- Because the persistence gap shrank from 0.73 to 0.44 and the intervals now overlap, the statement "persistence is significantly better than every optimisation method" was removed; a paired bootstrap is needed.
- Added: Verma et al. 2022 [9] (introduction, related work, Table 1, discussion, future work), Chand 2018, Barr and Kantor 2004, Lemmer 2002, Bhattacharjee and Saikia 2016, Ghosh 2021, Prakash 2016, Verma 2021 review, plus fuller reference details for items previously "as cited in [1]". Corrected the description of [1]: fielding is a tie-breaking criterion in tournament selection, not a full third objective.
- Added: sanity check on the full test population (runs MAE 13.772 to 12.879, wickets 0.844 to 0.816), the most recent match (6 Oct 2026, 11/11), Section 5.6 (NZ v India prospective prediction with returning-player sensitivity), the selection-model hyperparameters, the as-of row and context construction details, and new limitations.

**To re-run on the current data before submission.**
1. Tables 2 to 4 and the data split sizes (Sections 3 and 5.1), plus the XGBoost hyperparameter rows of Appendix A (the dataset grew by 34 matches).
2. Section 5.5 sensitivity (opposition and venue percentiles) on all 49 matches.
3. Zimbabwe case study (Table 6, Figures 3 and 4) if the scoring models or pool changed.
4. Paired bootstrap CIs for the Table 5 comparisons.
5. Figure 5 (overlap) with 49 matches; a new figure for the NZ front would help (Table 8).
6. Real XI of NZ v India (22 Oct 2026) and the post-match overlap; verify the date stamp of the saved prediction file.
7. Verify the 16-player squad against an official source and the author list of [15].
