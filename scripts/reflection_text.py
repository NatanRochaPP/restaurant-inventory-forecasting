"""Chapter 9 and the Section 3.2 calibration text, kept apart from the generator like
literature_text.py. Every factual claim is checked against the development record."""

CALIBRATION = [
    'The effect sizes are set per product in baseline_forecasting.py (lines 71 to '
    "92). Each day's demand mean includes a base level, a 12% trend over two years, "
    'annual seasonality, weekend, temperature and bank-holiday factors, and a 35% '
    'promotional uplift on days drawn with probability 0.08 (lines 111 to 119); units'
    ' sold are Poisson around that mean (line 124). Weekend multipliers run from 1.1 '
    '(Coffee Beans) to 1.6 (Soft Drinks and Truffle Oil), and the realised weekend '
    'uplift across all SKUs is 48.0%.',
    'Temperature sensitivity per 10 C runs from -0.2 (Coffee Beans) to 0.55 (Soft '
    'Drinks) (set at lines 71 to 92, applied at line 116). Relative to the generated '
    'mean of 10.9 C, at the 95th percentile of 20.2 C the demand mean rises by 51.2% '
    'for Soft Drinks, 37.2% for Seasonal Berries, 27.9% for Lettuce, 23.3% for '
    'Tomatoes and 4.7% at a sensitivity of 0.05. Badorf and Hoberg (2020), across 673'
    ' brick-and-mortar stores, report in their abstract that the effect on daily '
    'sales "can be as high as 23.1% based on the store location and as high as 40.7% '
    'based on the sales theme". Soft Drinks alone exceeds 40.7%; Seasonal Berries, '
    'Lettuce and Tomatoes lie between 23.1% and 40.7%; the other 16 lines stay below '
    '23.1% at the 95th percentile. The measures differ: theirs are upper values of '
    "the weather effect on a store's daily sales from a non-linear random coefficient"
    " model, mine the change in one product's demand mean at a temperature "
    'percentile, with temperature standing in for all weather. This is a comparison '
    'of unlike measures, not a calibration.',
    'The bank-holiday uplift was not calibrated to any source. Line 117 multiplies '
    'the demand mean by one plus a per-product holiday value, set at lines 71 to 92 '
    'from 0.15 (Coffee Beans) to 0.55 (Truffle Oil). Saputra and Kumar (2024), cited '
    'for it in an earlier draft, report no effect size: in their SARIMAX model the '
    'state holiday coefficient is 0 (p = 1.000000) and the school holiday coefficient'
    ' is 165.2158 (p = 0.919689). Against same-weekday non-holiday, non-promotion '
    'days, the realised uplift on open bank holidays has a median of 38.8% across the'
    ' 20 SKUs, from 2.8% (Coffee Beans) to 183.4% (Truffle Oil).',
    'The widest values combine the holiday factor with a generator rule. Zeros are '
    'imposed on intermittent lines only when the day is not a bank holiday (line '
    '127), so those lines gain the holiday factor and lose their imposed zero days '
    'together. Truffle Oil, with a zero-day probability of 0.45 and a factor of 0.55,'
    ' shows 183.4%, Seasonal Berries 144.7% and Gluten-free Buns 141.7%. Excluding '
    'the eight intermittent lines, the median uplift is 35.2%. That pattern belongs '
    'to the generator, not to restaurants.',
]

REFLECTION = [
    ('9.1 Project management', [
        'The AE1 report, due on 10 July 2026 before 4:00 PM, planned eight milestones'
        ' against a 400-hour budget from June to September 2026, with scoping and the'
        ' data pipeline complete, modelling in progress, and "about 170 of the 400 '
        'hours" used. It contradicts itself: the milestone table puts the simulator '
        'in July 2026 and the dashboard in August 2026, while the implementation '
        'paragraph puts "the simulator in August, the dashboard in September", with '
        'write-up "through October-November", past the "September hand-in" it names '
        'earlier. It was marked on 21 July 2026 at 74, and the marker warned: "If the'
        ' inventory policy simulator encounters logic or cost-modelling delays, the '
        'entire evaluation phase will be severely compromised."',
        'The dated record follows neither version. The session on 10 August 2026 '
        'began when I pasted a brief of 10 phases at 10:32 BST. The AI assistant '
        '(Claude) reported the roll28 defects from its Phase 1 code audit at 10:36, '
        'ran the first baseline-versus-AI comparison at 11:22, began the dashboard at'
        ' 12:17 and reported all 10 phases complete at 13:25. The evaluation and '
        'pilot testing milestone has no matching event: the assistant prepared ethics'
        ' materials on 25 August 2026, but no pilot appears in the dated record. I '
        'asked for the report to be started on 28 August 2026. Work then returned '
        'twice to finished phases during the write-up: the backward-date cache '
        'defect, reported on 4 September 2026 and fixed on 11 September 2026, and the'
        ' order-dependent refit defect found on 14 September 2026.',
        'Hours cannot be compared. The AE1 report said "I log my time against the '
        'eight work packages", but no such log is in the record. The only recorded '
        'time is 8.0 hours of sessions with the assistant between 10 August and 14 '
        'September 2026; time outside those sessions is not recorded, so neither the '
        'budget nor the "about 170" hours can be checked. Until 14 September 2026 the'
        ' repository held two commits, both from 9 July 2026, and 112 untracked '
        'files.',
    ]),
    ('9.2 Technical decisions and what they cost', [
        'Separating the inventory policy from the forecast was a requirement in my '
        'brief, not a discovery: "Implement inventory policies independently of '
        'forecasting." The alternative was order quantities computed inside the '
        'forecasting code. The cost was an interface and compute: the ForecastResult '
        'the assistant wrote at 10:45 has to carry sigma for safety stock, and the '
        'sweep needed a forecast cache "so sweeps are affordable".',
        'Sizing safety stock from forecast error went beyond the brief, which asked '
        'only for "the configured service-level target and demand uncertainty". The '
        'module the assistant wrote at 11:07 on 10 August 2026 uses forecast error '
        'because "a SKU whose demand swings predictably with the weekend does not '
        'need cover for that swing, because the forecast already anticipates it", an '
        'argument set out in Section 2.6 (Prak, Teunter and Syntetos 2017). The '
        'alternative, the standard deviation of demand, would have given buffers 133%'
        ' larger summed over the 20 lines, at the same z and protection period and a '
        'decision date of 5 December 2025. The cost, by my reasoning, is dependence '
        'on the error estimate: a model whose errors are understated is given too '
        'little stock. The gain is also uneven: for Gluten-free Buns, Vegan Cheese, '
        'Seasonal Berries and Truffle Oil the two measures give the same buffer.',
        'Counterfactual ablation was not a requirement either. The brief asked for '
        'feature contributions "where practical" and named no attribution library. '
        'The assistant wrote src/explainability/forecast_drivers.py at 11:37 on 10 '
        'August 2026; it predicts the day twice, with the real feature value and with'
        ' a neutral reference, and reports the difference. The alternative was a '
        'library such as SHAP or LIME. The record states no reason for rejecting one;'
        " it shows only that ablation added no dependency and uses the model's own "
        'prediction call. The cost is written into the module: contributions "do not '
        'sum exactly to the prediction, because the model contains interactions", so '
        'they are shown as directional, with a residual.',
    ]),
    ('9.3 Finding errors in my own earlier work', [
        'The roll28 defects were found by the assistant on 10 August 2026 at 10:36, '
        'in the Phase 1 code audit my brief required ("First inspect the repository '
        '... before modifying anything"). In the AE1 baseline script (line 233) the '
        'rolling mean crossed SKU boundaries and landed on the wrong rows: measured '
        'against a correct implementation it correlated at 0.19, with 87% of rows '
        'differing by more than 1 unit. The same audit reported the second defect: '
        'computed over the whole frame, test day origin+6 saw actual demand from days'
        ' origin to origin+5. Neither was visible in the results. A feature acting as'
        ' noise still yields a plausible figure, as the AE1 sMAPE of 33.67 below ETS '
        'at 34.22 shows, and the leak was never measured on its own. The audit '
        'predicted the fix "should improve the headline number"; instead sMAPE rose '
        'to 34.06, and is 34.11 in the final version, with the gain over seasonal '
        'naive down from 15.5% to 14.4%. The record does not separate the noise from '
        'the leak, and Section 5.9 reports the corrected figure.',
        'The backward-date cache defect was reported by the assistant on 4 September '
        '2026 at 16:34, after I asked it to run the code, while smoke-testing the '
        "dashboard's page code paths. The staleness check in "
        'src/forecasting/service.py (lines 43 to 46) treated a negative date '
        'difference as "not stale", so going back in time reused a model fitted on '
        'later data: for Chicken Breast at 15 August 2025 the 7-day forecast differed'
        ' by -4.20 to 5.28 units a day depending on viewing order. The assistant '
        'wrote service.py at 10:58 on 10 August 2026; the record does not show '
        'whether the check was one-directional in that version. The batch scripts '
        'advance monotonically and no reported run moved backwards, so results could '
        'not reveal it and no reported number changed, and the tests missed it '
        'because "every leakage test tampers with future demand". When the assistant '
        'reverted each fix on 11 September 2026 at 14:01, the explain_forecast bypass'
        ' could be removed with all 6 tests passing. The fixes were reported complete'
        ' at 14:05, and the record does not show whether the 231-test suite at 14:05 '
        'catches the bypass.',
        'A further defect, found on 14 September 2026, did change reported numbers. '
        'Capturing the dashboard, the assistant found the Simulation page showing '
        'forecast-driven waste of 2,309.79 units against 2,363.79 in the results '
        'file. Stored forecasts had been made under a refit cadence that depended on '
        'which dates were requested first. No model saw later data, so it was not a '
        'leak. From the options the assistant set out, I chose to fix refit dates to '
        'a grid; a new test, eval and end-to-end test all failed on the old code, and'
        ' the matched waste reductions moved from 25.8% and 38.4% to 28.6% and 39.6%.',
    ]),
    ('9.4 The comparison that nearly went wrong', [
        'On 10 August 2026 at 11:22 the first baseline-versus-AI run showed waste '
        'down 65% and stockouts up 55%. Before treating it as a finding the assistant'
        ' checked whether it was a defect, and that check contained an error of its '
        "own: its diagnostic's model column was misaligned, while the system routed "
        'correctly. At 11:23 it diagnosed an operating-point mismatch: the manual '
        'buffer ran at about 99% fill rate (98.9% in its 13:25 summary) while the AI '
        'policy targeted 95%, so the comparison measured stock level rather than '
        'policy. It then began the frontier sweep.',
        'The assistant recommended the matched comparison at 13:25, and I report it '
        'as the result. Section 5.6 still reports the unmatched one: waste improved '
        'by 64.0% and stockout units by -53.4% before the refit correction, and by '
        '65.9% and -59.8% after it. With the refit correction from Section 9.3 '
        'applied, at a matched service level of 96.76% the AI policy wastes 28.6% '
        'less at 10.8% lower total cost, and at 97.9% it wastes 39.6% less; at '
        'matched waste of 2498.8 units it serves 1.66 percentage points more demand.',
        'What that event taught is that a correct calculation can answer the wrong '
        'question. It was caught because a defect check came before interpretation. '
        "Had the 65% been the headline, the dissertation's objective to compare the "
        'two policies fairly would have been nominally met and not delivered.',
    ]),
    ('9.5 Professional and ethical considerations', [
        'I carried out this project under a model that permits AI assistance without '
        'restriction or justification, and used it throughout: to write and test the '
        'code, to locate and verify sources, and to draft this document.',
        'The record shows the division of work. I pasted my brief on 10 August 2026 '
        'at 10:32 BST. From it the assistant audited the code, wrote the modules and '
        'tests, and reported all 10 phases complete at 13:25 with 227 tests passing. '
        'On 28 August 2026, at my request, it generated a first draft of this '
        "document of about 7,700 words with 58 markers. The assistant's checks "
        'produced each correction reported here: the roll28 defects, the operating-'
        'point diagnosis, the cache defect on 4 September 2026, the refit defect on '
        '14 September 2026 and, from a source check on 14 September 2026, the finding'
        ' that Saputra and Kumar (2024) give no holiday effect size. My later '
        'recorded actions include asking the assistant to run the code on 4 September'
        ' 2026, pasting eight drafted blocks on 12 September 2026 and choosing the '
        'refit correction on 14 September 2026. My role was to specify the system, '
        'decide which findings to adopt, and answer for each claim the document '
        'makes. The earlier calibration claim in Section 3.2 shows that this last '
        'part is where the risk sits.',
        'Decision support rather than automation was my requirement: "The system is '
        'decision-support only. Never automatically place an order." It limits the '
        'evidence. No manager has used the dashboard and the simulated manager '
        'accepts every recommendation, so the simulation measures the recommendations'
        ' as if they were automated, and human override is untested.',
        'Synthetic data was also my requirement: "Keep all operational data '
        'anonymised/synthetic." No real data is processed, and the leakage tests '
        'multiply future demand and assert that earlier decisions are unchanged.',
    ]),
    ('9.6 What I would do differently', [
        'First, I would keep the time log the AE1 report said I kept, and commit at '
        'the end of each phase. The only recorded time is 8.0 hours of sessions, and '
        'until 14 September 2026 the repository held two commits from 9 July 2026 and'
        ' 112 untracked files. Weekly hours against the eight milestones would let '
        'Section 9.1 compare the plan with effort, not only with dates.',
        'Second, I would test every cache for order-dependence and revert every fix '
        'once. The revert check on 11 September 2026 at 14:01 showed the '
        'explain_forecast bypass fix could be removed with all 6 tests passing. Each '
        'cache test would require identical results whether dates are visited '
        'forwards, backwards or in a clean process.',
        "Third, I would record each source's figure beside the generator parameter "
        'before writing that the parameter is calibrated. The earlier Section 3.2 '
        'attributed the bank-holiday uplift to Saputra and Kumar (2024); their '
        'journal page, fetched on 14 September 2026 at 15:30, shows no effect size.',
    ]),
]
