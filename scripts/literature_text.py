"""The written Literature Review, as authored, with the agreed corrections applied:

* Posch et al. cited as (2022), the published International Journal of Forecasting
  version, rather than the 2020 arXiv preprint.
* Lost-sales modelling cross-referenced to Section 1.5, not 3.2.
* Matched-operating-point protocol cross-referenced to Section 3.11, not 3.8.
* Section 2.6 separates the two distinct arguments drawn from Prak et al.
* Em dashes normalised to the hyphen convention used elsewhere in the document.
"""

INTRO = (
    "This chapter establishes the evidential basis for each design decision taken in the "
    "project. It moves from the operational problem to the instruments used to address it: "
    "the scale and composition of food-service waste and the cost of the availability "
    "failures that sit opposite it (2.1); the forecasting granularity the replenishment "
    "decision actually requires (2.2) and the model families capable of delivering it (2.3, "
    "2.4); the inventory theory that converts a forecast into an order under perishability "
    "(2.5, 2.6) and the segmentation that differentiates that conversion across a "
    "heterogeneous range (2.7); the obligations that attach to a system whose output a human "
    "must accept or reject (2.8); and the protocol by which any of it can be honestly "
    "evaluated (2.9). Section 2.10 draws these together into the gap the project occupies."
)

SECTIONS = [
    ("2.1 Food waste and availability in food service", [
        'The problem this project addresses is well documented at sector level. WRAP '
        '(2025) estimates that UK hospitality and food service generates 1.1 million '
        'tonnes of food waste a year, of which 800,000 tonnes could have been eaten, '
        "valued at £3.21 billion at 2021 prices, and that the sector's food waste "
        "amounts to 18% of the food it buys. WRAP's sector page puts the average cost"
        ' at £10,000 per outlet per year (WRAP n.d.). Both figures are modelled '
        'rather than newly measured, a limitation Section 1.1 sets out.',

        "The more useful question for a project concerned with ordering is not how much waste "
        "exists but how much of it an ordering decision can reach. WRAP (2013) attributes 45% of "
        "the sector's food waste to preparation, 34% to customer plates and 21% to spoilage. "
        "Only the last category is even potentially addressable by ordering differently. "
        "Preparation waste is determined by trimming, portioning and kitchen technique; plate "
        "waste by what a guest elects to leave. Neither responds to a change in delivery "
        "quantity. Spoilage is the category an ordering decision creates, and WRAP lists "
        "over-supply among its causes alongside equipment failure, improper storage and "
        "poor-quality purchasing.",

        "This is where the literature stops being able to help, and the limitation should be "
        "stated rather than elided. No published breakdown isolates over-ordering as a share of "
        "sector food waste. The published categories are defined by the stage at which waste is "
        "observed, not by the decision that caused it, so a spoiled case of tomatoes is recorded "
        "identically whether it reflects an over-large order, a failed chiller or a short-dated "
        "delivery. The 21% spoilage share is therefore an upper bound on the addressable "
        "fraction and not an estimate of it. This project consequently evaluates simulated waste "
        "and service level directly rather than claiming a share of the sector total - a "
        "decision that follows from the state of the evidence rather than from convenience.",

        "Waste is only half of the trade-off, and the literature on the other half is "
        "considerably firmer, though it comes from retail rather than hospitality. Corsten and "
        "Gruen (2003) establish both the extent of retail out-of-stocks and, more significantly "
        "for this project, the mechanism by which an availability failure becomes a lost sale: a "
        "shopper facing an absent item substitutes, defers, buys elsewhere, or abandons the "
        "purchase, and only some of those responses preserve the retailer's revenue. The "
        "restaurant case is structurally similar but less forgiving. A guest cannot defer a meal "
        "to a later visit in the way a shopper defers a grocery purchase, and an unavailable "
        "menu item is a lost sale at the moment it is discovered. This asymmetry is the "
        "justification for modelling unmet demand as lost sales rather than backorders, as "
        "stated in Section 1.5, and for costing a stockout above the unit cost of the item.",

        "That ordering is in practice a matter of human judgement rather than system output is "
        "also evidenced, again from retail. Van Donselaar et al. (2010) study how store managers "
        "behave when an automated replenishment system advises a quantity, and find systematic "
        "departure from that advice rather than random noise around it - the deviations follow "
        "regularities, particularly for perishable and slow-moving lines. This finding does two "
        "things for the present work. It supports the premise that manual judgement is the "
        "operative mechanism, which is what the baseline in this project formalises. And it "
        "establishes that such judgement is patterned, which means an automated system can be "
        "designed to work with it rather than against it - the reasoning behind retaining "
        "manager override and recording it, rather than presenting a recommendation as final.",

        "The absence worth naming is that none of the sources reviewed studies small hospitality "
        "operations. The evidence that ordering is manual, and that manual ordering departs "
        "predictably from system advice, is drawn from supermarkets. Restaurants differ in shelf "
        "life, order frequency, supply chain complexity and the sophistication of the systems "
        "available to them, and the direction in which those differences push the finding is not "
        "established. That absence is itself part of the gap this project sits in, and it is "
        "returned to in Section 2.10.",
    ]),

    ("2.2 Demand forecasting for perishable goods", [
        "An ordering decision is taken for a specific item, on a specific day, for a specific "
        "delivery. The forecast that informs it must therefore be at item level and at daily "
        "resolution; a weekly or category-level forecast cannot be converted into an order for "
        "Tuesday's chicken. This apparently obvious point determines a great deal of what "
        "follows, because daily item-level series are short, noisy, frequently intermittent, and "
        "dominated by calendar structure rather than by trend.",

        "Schmidt, Kabir and Hoque (2022) provide the closest published analogue in setting: "
        "daily item-level sales forecasting in an operating restaurant using real point-of-sale "
        "data. Their work establishes that the granularity is tractable and that machine-learning "
        "methods can be applied to it. Posch et al. (2022) address the same domain from a "
        "different direction, forecasting daily menu-item quantities in staff canteens and "
        "restaurants, and note a structural feature of the setting that matters for this "
        "project's assumptions: the restaurant industry is characterised by short ordering lead "
        "times and the absence of the complex supply chain that constrains retail. This supports "
        "modelling supplier lead time as short and deterministic, while also implying that the "
        "availability ceiling identified in Section 6.4 is a real operating constraint rather "
        "than an artefact of an unrealistic parameter.",

        "Two exogenous drivers recur in the daily-demand literature and are both engineered as "
        "features here. Badorf and Hoberg (2020) isolate the effect of daily weather on sales "
        "across a large panel of brick-and-mortar stores, establishing weather as a genuine "
        "driver of daily demand rather than a plausible-sounding one, and supplying the "
        "calibration basis for the temperature sensitivity built into this project's generator. "
        "Saputra and Kumar (2024) model holiday and event effects as exogenous regressors within a "
        "SARIMAX specification, which is the conventional statistical treatment and the "
        "calibration basis for the bank-holiday uplift used here. The methodological point these "
        "two share is that both effects are knowable in advance - a calendar indefinitely, a "
        "weather forecast for several days - which is precisely the information a trailing par "
        "level discards, and therefore the hypothesised source of any improvement this project "
        "might demonstrate.",

        "Huber and Stuckenschmidt (2020) are the closest published analogue in method rather "
        "than setting. They forecast daily retail demand using machine learning with calendric "
        "special days treated explicitly, which is structurally the same problem as the one "
        "addressed here and the nearest precedent for the feature set in Section 3.3. The M5 "
        "competition data (Makridakis, Spiliotis and Assimakopoulos 2022b) supplies the closest "
        "large public analogue in data structure: daily, hierarchical, SKU-level demand at store "
        "level, including the long tail of slow-moving lines that a single restaurant's range "
        "also contains. The structure of the M5 series, rather than the competition results, is "
        "what informs the weekly pattern and the proportion of intermittent lines in this "
        "project's generator.",

        "Perishability changes what a forecast is for, and van Donselaar et al. (2006) are the "
        "reference point. In replenishing short-shelf-life products in supermarkets, an order "
        "commits to a quantity that cannot be carried forward if it proves wrong. Forecast error "
        "on a non-perishable line produces a holding cost; on a three-day line it produces a "
        "write-off. This asymmetry is why forecast quality and inventory policy cannot be "
        "treated as separable problems here, an argument developed in Section 2.6 and borne out "
        "empirically in Section 6.3.",

        "Finally, the horizon over which to forecast is not arbitrary. Nikolopoulos et al. "
        "(2011) demonstrate that the level of temporal aggregation materially affects forecast "
        "quality, and that aggregating to a bucket equal to lead time plus review period - the "
        "protection period - is a principled choice rather than a convenience. This project "
        "forecasts over exactly that interval, which aligns the forecast with the quantity the "
        "inventory policy actually needs: not demand on a given day, but demand over the window "
        "the current order must cover.",
    ]),

    ("2.3 Statistical and machine-learning baselines", [
        "A forecasting result is only interpretable against a benchmark, and the choice of "
        "benchmark is a substantive methodological commitment rather than a formality. Hyndman "
        "and Athanasopoulos (2021) establish seasonal naive - the value observed on the same "
        "weekday of the previous week - as the standard reference for series with strong "
        "seasonal structure, and it serves two roles in this project: as a competing model in "
        "its own right, and as the denominator of MASE. For daily restaurant demand, where the "
        "day-of-week effect is the dominant signal, seasonal naive is a demanding benchmark "
        "rather than a straw man, and a model that fails to beat it has demonstrated nothing. "
        "This is the reason the model set spans a deliberate ladder from naive through seasonal "
        "naive and exponential smoothing to gradient boosting: each step must earn its "
        "additional complexity against the one below it.",

        "Exponential smoothing, and specifically the Holt-Winters formulation with additive "
        "trend and additive seasonality, is the natural statistical competitor at this "
        "granularity (Hyndman and Athanasopoulos 2021). It captures level, trend and weekly "
        "seasonality in a small number of estimated parameters and is robust on short series - "
        "but it cannot use exogenous information. A bank holiday, a promotion or a temperature "
        "forecast are invisible to it. That limitation is the analytical justification for "
        "admitting a feature-based model at all: the question is not whether machine learning is "
        "fashionable but whether the covariates it can accept carry information that a purely "
        "autoregressive method cannot access.",

        "The gradient-boosting family (Chen and Guestrin 2016) is the appropriate choice for "
        "that role. Its suitability here is not asserted on general grounds but on evidence from "
        "the closest available setting: the M5 accuracy competition (Makridakis, Spiliotis and "
        "Assimakopoulos 2022a) found that gradient-boosted trees dominated on daily hierarchical "
        "retail data, and further that models trained across many series outperformed models "
        "fitted to each series individually. That second finding is the direct justification for "
        "the global, cross-SKU model used here. Fitting a separate model per SKU on two years of "
        "daily data would leave each model estimating calendar effects from a handful of "
        "observations of each holiday; pooling across the range allows the weekend and "
        "bank-holiday structure to be learned once, from twenty series rather than one.",

        "The literature does not speak with one voice, and the counterpoint is worth engaging "
        "rather than omitting. Schmidt, Kabir and Hoque (2022) report that on a single "
        "restaurant's data, simpler linear models matched or exceeded heavier machine-learning "
        "methods at a one-day horizon. Two considerations reconcile this with the choice made "
        "here. First, horizon: the advantage of a feature-based model lies in anticipating "
        "conditions several days ahead that no recent observation reveals, and at a one-day "
        "horizon there is little for it to anticipate. This project forecasts over a seven-day "
        "protection period, where that advantage should be available. Second, the finding is a "
        "caution against assuming the heavier model wins, which is exactly why the full model "
        "ladder is evaluated empirically in Section 5.1 rather than assumed. The result there - "
        "a modest margin over ETS on sMAPE but a wider one on WAPE - is consistent with Schmidt, "
        "Kabir and Hoque's caution rather than a refutation of it.",

        "Prophet (Taylor and Letham 2018) was considered and not adopted. Its design goals - "
        "interpretable decomposition into trend, seasonality and holiday components, usable by "
        "analysts without forecasting expertise - align well with the explainability "
        "requirements set out in Section 2.8, and it handles holiday effects natively. It was "
        "rejected because its decomposition is additive in a fixed structural form, which offers "
        "less flexibility in representing interactions between calendar effects, promotions and "
        "weather than a tree ensemble, and because the interpretability requirement in this "
        "project is satisfied at the recommendation level rather than requiring an intrinsically "
        "interpretable forecaster.",
    ]),

    ("2.4 Intermittent demand", [
        "A restaurant range is not homogeneous. Alongside milk and chicken breast sit garnishes "
        "and specialities that sell on a minority of days and in small quantities. Applying a "
        "model designed for smooth, high-volume series to such a line is not merely suboptimal; "
        "it is a category error, because the quantity being estimated is different. For an "
        "intermittent series the useful estimate is a demand rate - expected units per period, "
        "accounting for both how often demand occurs and how large it is when it does - rather "
        "than a prediction of any particular day's sales.",

        "Croston (1972) is the foundational treatment. The method separates the series into two "
        "components, the size of demand when it occurs and the interval between occurrences, "
        "smooths each exponentially, and updates them only on periods in which demand is "
        "observed. The resulting rate is the ratio of smoothed size to smoothed interval. The "
        "separation is what makes the method appropriate: applying simple exponential smoothing "
        "to a series containing many zeros drags the estimate downwards on every empty period "
        "and produces a systematic understatement of the rate.",

        "Croston's estimator is nevertheless biased, and Syntetos and Boylan (2005) establish "
        "both the existence and the magnitude of that bias, proposing the correction now "
        "standard in the literature: multiplying the demand rate by a factor of one minus half "
        "the smoothing constant. This is the Syntetos-Boylan Approximation implemented in this "
        "project, and its inclusion rather than plain Croston is a deliberate choice traceable "
        "to that result rather than to convention.",

        "Deciding which SKUs should be routed to such a method requires a rule, and Syntetos, "
        "Boylan and Croston (2005) supply it. They categorise demand patterns as smooth, "
        "intermittent, erratic or lumpy according to two statistics - the average inter-demand "
        "interval and the squared coefficient of variation of demand sizes - and derive "
        "threshold values separating the regions on the basis of which estimator achieves lower "
        "mean square error. This categorisation is the basis of the routing rule used in Section "
        "3.5, which is the point at which a theoretical classification becomes an operational "
        "decision about which model serves which product.",

        "Babai et al. (2019) extend the treatment to the case where stock may become obsolete, "
        "proposing a forecasting method that accounts for inventory obsolescence and correcting "
        "further bias in Croston-type estimators under those conditions. Their setting is "
        "obsolescence in the spare-parts sense rather than perishability in the food sense, and "
        "the distinction matters: an obsolete part loses its value unpredictably at an unknown "
        "future point, whereas a perishable food item has a known, short and deterministic life. "
        "The parallel is nevertheless instructive, because both break the assumption underlying "
        "classical inventory theory that unsold stock retains its value into the next period - "
        "and it is that assumption, rather than any property of the forecast, which "
        "perishability destroys. Section 2.5 takes up the consequence.",
    ]),

    ("2.5 Inventory policy for perishables", [
        "Converting a forecast into an order requires a policy, and the relevant formulations "
        "are long established. Silver, Pyke and Peterson (1998) set out the periodic-review "
        "framework used here: at fixed review intervals, stock is raised to a target level sized "
        "to cover expected demand over the protection period plus a buffer against uncertainty. "
        "Zipkin (2000) provides the formal basis for the base-stock and (s,S) policies, "
        "establishing the conditions under which order-up-to structures are optimal under lead "
        "time. Chopra and Meindl (2016) frame the resulting trade-off in the terms an operator "
        "recognises: lead time, demand variability and the chosen service level jointly "
        "determine how much stock must be held, and any one can be traded against another.",

        "The protection period is the interval the current order must cover, and it is lead time "
        "plus review period rather than lead time alone. The distinction is not pedantic. Once "
        "an order is placed, the next opportunity to correct it is the following review, so the "
        "stock on hand must survive until the next delivery arrives, not merely until this one "
        "does. With a fixed delivery calendar this is a mechanical calculation, and it is the "
        "quantity the forecast in Section 2.2 is aligned to.",

        "Safety stock under the normal approximation follows from the service-level target: the "
        "buffer is the product of a standard normal quantile corresponding to the target and the "
        "standard deviation of demand over the protection period. The approximation is "
        "convenient and standard, and its assumptions should be stated plainly because they are "
        "not fully satisfied here. Daily restaurant demand for a given SKU is a count, is "
        "frequently right-skewed, and for slow movers is not remotely normal. The approximation "
        "is retained on the grounds of transparency and because the operational results do not "
        "turn on it, but the shift to quantile-based sizing is identified as future work in "
        "Section 7.4 for exactly this reason.",

        "Perishability then changes the structure of the problem rather than merely its "
        "parameters. Nahmias (1982) remains the standard review of fixed-life perishable "
        "inventory theory, and establishes the central difficulty: once stock ages, the state of "
        "the system is no longer a single number but a vector recording how much stock remains "
        "at each age, and the optimal policy becomes computationally intractable for any "
        "realistic shelf life. The literature's response has been to adopt heuristics with known "
        "behaviour rather than to pursue optimality, which is the position taken here - a "
        "base-stock policy with an explicit shelf-life cap, whose behaviour can be inspected, in "
        "preference to an optimisation whose behaviour cannot.",

        "The substantive consequence is that safety stock changes sign as a concept. In "
        "classical inventory theory, cover held beyond immediate need is insurance: it costs the "
        "holding rate and is eventually sold. For an item whose shelf life is shorter than the "
        "interval over which that cover would be consumed, it is not insurance but a guaranteed "
        "write-off - waste on a timer, incurring the full unit cost plus disposal. This is the "
        "most consequential theoretical point in the chapter for the design of the artefact, "
        "because it is what justifies capping the order quantity at what can plausibly be "
        "consumed within the remaining shelf life, and it is what explains the empirical finding "
        "in Section 5.8 that the waste benefit is concentrated in short-shelf-life lines.",

        "Issuing policy matters alongside ordering policy. Pierskalla and Roach (1972) establish "
        "the conditions under which issuing the oldest unit first minimises outdating, which is "
        "the theoretical basis for the first-expired-first-out rule implemented in the "
        "simulator. Modelling issue as FEFO rather than at random is a decision that materially "
        "affects simulated waste, and grounding it in this result rather than in operational "
        "folklore matters because the waste figures reported in Chapter 5 depend on it.",

        "Van Donselaar et al. (2006) close the loop between theory and practice, examining how "
        "perishable replenishment is actually conducted in food retail. Their work is the bridge "
        "between the formal treatments above and the operational reality this project models, "
        "and it supports treating shelf life as a first-class constraint in the ordering "
        "decision rather than as an afterthought applied to the result.",
    ]),

    ("2.6 Safety stock from forecast error", [
        "The conventional safety-stock formula in Silver, Pyke and Peterson (1998) sizes the "
        "buffer from the standard deviation of demand. This project departs from that "
        "convention, sizing it instead from the standard deviation of forecast error, and the "
        "departure requires justification because it is the decision that most directly produces "
        "the waste reduction reported in Chapter 5.",

        "Two distinct arguments support the departure, and they should be kept separate because "
        "they do different work. The first concerns which quantity is the correct one to "
        "measure. Prak, Teunter and Syntetos (2017) address the calculation of safety stocks "
        "when demand must itself be forecast, and show that the conventional treatment is "
        "incomplete: it treats the demand distribution as known, when in practice its parameters "
        "are estimated from data and those estimates carry error. Because that estimation "
        "uncertainty is ignored, the achieved service level does not match the target the buffer "
        "was sized for. Their analysis establishes that the quantity the buffer must protect "
        "against is the error in the forecast rather than the variability of the underlying "
        "series - which is, in any case, the quantity the inventory system actually faces, since "
        "the order is placed against the forecast and the shortfall that can arise is the gap "
        "between forecast and realisation. It should be noted that the correction Prak et al. "
        "derive acts to increase the required buffer relative to the naive formula, since it "
        "accounts for uncertainty the conventional treatment omits.",

        "The second argument concerns the magnitude of that quantity in this particular setting, "
        "and it points the other way. Daily restaurant demand for a fast-moving fresh item "
        "varies enormously across the week - a Saturday may be several times a Tuesday - so its "
        "raw standard deviation is large. But almost all of that variation is predictable. "
        "Sizing safety stock from raw demand variability would therefore inflate the buffer in "
        "proportion to a weekly swing the forecast already anticipates, and it would do so most "
        "aggressively on exactly the high-volume, short-shelf-life items where surplus stock "
        "becomes waste rather than carried inventory. Sizing from forecast error charges the "
        "buffer only for the variation that remains genuinely unpredictable once the calendar "
        "has been accounted for. The two arguments are compatible rather than contradictory: "
        "Prak et al. establish which quantity to measure, and the seasonal structure of this "
        "demand determines that the quantity so identified is substantially smaller than raw "
        "demand variability. This is the mechanism described in Section 6.2 and is why the "
        "forecast-driven policy achieves comparable availability from a materially smaller stock "
        "holding.",

        "It follows that forecasting and inventory policy are one design problem rather than two "
        "sequential ones. Syntetos and Boylan (2006) make the general form of this argument "
        "empirically, evaluating intermittent-demand estimators by their realised stock-control "
        "performance rather than by error metrics alone, and demonstrating that the ranking of "
        "methods by accuracy does not reliably predict their ranking by inventory outcome. That "
        "finding is the methodological warrant for this project's entire evaluation design: it "
        "is why Chapter 5 reports operational outcomes and not only sMAPE, and why an "
        "improvement in forecast accuracy is treated as a hypothesis about operational benefit "
        "rather than as evidence of it.",
    ]),

    ("2.7 ABC/XYZ segmentation", [
        "A twenty-line range is heterogeneous in two independent dimensions, and treating it "
        "uniformly wastes effort on the unimportant and under-serves the critical. The first "
        "dimension is economic: Slack, Brandon-Jones and Johnston (2016) set out the Pareto "
        "principle as applied to stock, where a minority of lines accounts for the large "
        "majority of annual purchase value and therefore warrants the majority of management "
        "attention. This is the ABC axis, and Section 5.3 confirms the expected distribution on "
        "this project's data.",

        "Value alone is insufficient, because two items of identical annual value may behave "
        "entirely differently - one selling steadily every day, the other in unpredictable "
        "bursts. The second dimension is behavioural, and the categorisation in Syntetos, Boylan "
        "and Croston (2005) discussed in Section 2.4 is what the XYZ axis operationalises: "
        "classification by demand variability and intermittency rather than by value. Crossing "
        "the two produces a grid in which each cell carries a different implication for how the "
        "item should be forecast and how much cover it warrants.",

        "Teunter, Babai and Syntetos (2010) supply the justification for acting on that grid "
        "rather than merely reporting it. They examine the differentiation of service-level "
        "targets across inventory classes and the cost consequences of doing so, and their "
        "analysis supports differentiating the target by class rather than applying a single "
        "figure across the range - which is the practice adopted in Section 3.6. Applying one "
        "service level uniformly means either over-serving cheap, stable items or under-serving "
        "expensive, volatile ones.",

        "Segmentation in this project does work beyond parameter-setting, however, and this is "
        "worth stating explicitly because it is where the design departs from the conventional "
        "use of ABC/XYZ. The grid also drives model selection, routing intermittent lines to the "
        "Syntetos-Boylan estimator and the remainder to the global gradient-boosting model, with "
        "the justification recorded per SKU. This converts what is normally a "
        "management-reporting device into an auditable routing rule, and it has a consequence "
        "visible in Section 6.5: Croston performs worse than seasonal naive when averaged across "
        "all twenty SKUs, yet remains the correct choice for the four it was designed for. A "
        "system reporting only aggregate accuracy would have discarded it.",
    ]),

    ("2.8 Explainability in operational decision support", [
        "This system recommends; a manager decides. That division of labour is a deliberate "
        "design commitment stated in Section 1.5, and it generates obligations that a fully "
        "automated system would not carry. A recommendation a human must accept or reject is "
        "worthless if the human cannot assess it, and it is dangerous if they cannot tell when "
        "not to trust it.",

        "The Information Commissioner's Office and The Alan Turing Institute (2020) set out the "
        "UK regulatory expectations for explaining AI-assisted decisions, and their taxonomy of "
        "explanation types structures what this system surfaces. Three are directly applicable "
        "here: the rationale explanation, covering why a particular recommendation was produced; "
        "the responsibility explanation, establishing who is accountable for the resulting "
        "decision, which in this design remains unambiguously the manager; and the data "
        "explanation, showing what information the recommendation was derived from. The guidance "
        "is not binding on a system processing no personal data, and Section 7.2 addresses that "
        "position, but it is the appropriate standard of professional practice for AI-assisted "
        "operational decisions in a UK context regardless.",

        "The technical framework against which this project's approach must be positioned is the "
        "additive feature-attribution family, of which Lundberg and Lee (2017) provide the "
        "unifying treatment. Their formulation attributes a prediction to its input features "
        "such that the attributions sum to the prediction, a property that makes the explanation "
        "both interpretable and complete. This project uses counterfactual ablation instead - "
        "measuring the change in forecast when a feature is withheld - and the comparison is the "
        "point. Ablation is computationally cheap, requires no additional library, and is "
        "straightforward to describe to a non-technical user. What it does not deliver is "
        "additivity: because the underlying model represents interactions between features, the "
        "individual ablation effects do not sum to the total, and a remainder exists. This "
        "project's explainability layer displays that remainder rather than normalising it away, "
        "which is a weaker but more accurate claim than an additive attribution would license. "
        "Positioning the choice against Lundberg and Lee is what makes the limitation legible "
        "rather than hidden.",

        "A second obligation follows from how people actually respond to algorithmic advice. "
        "Dietvorst, Simmons and Massey (2015) demonstrate algorithm aversion: having observed an "
        "algorithm err, people abandon it in favour of their own judgement even when the "
        "algorithm's overall performance remains superior. The finding is directly consequential "
        "for a system of this kind, because a forecast will certainly be wrong on some days and "
        "a manager will certainly notice. Two design decisions respond to it. The system "
        "displays its own recent error record alongside each recommendation, so that an "
        "individual miss is encountered in the context of overall performance rather than in "
        "isolation. And override is retained and recorded rather than discouraged, both because "
        "responsibility genuinely rests with the manager and because a system that cannot be "
        "overridden is more likely to be abandoned than one that can. Section 6.7 notes the "
        "corresponding limitation: the simulation models a manager who accepts every "
        "recommendation, so the realised benefit in practice would be some fraction of the "
        "simulated one.",
    ]),

    ("2.9 Evaluation methodology", [
        "Any evaluation of a forecasting system must respect the arrow of time, and the standard "
        "protocol for doing so is rolling-origin cross-validation (Hyndman and Athanasopoulos "
        "2021). A single train-test split on a time series wastes data and produces an estimate "
        "dependent on where the split happened to fall; random k-fold cross-validation is "
        "straightforwardly invalid, since it permits a model to be fitted on data postdating the "
        "period it is evaluated on. Rolling origin advances a forecast origin through the "
        "series, refitting at each step and evaluating over the subsequent horizon, which both "
        "uses the data efficiently and replicates the information state of a real forecaster at "
        "each decision point. This project's protocol - twelve expanding-window folds at a "
        "seven-day horizon - follows directly from that requirement, and the horizon is set to "
        "the protection period rather than chosen for convenience, for the reasons given in "
        "Section 2.2.",

        "The metric set requires equal care, and the case is made by Hyndman and Koehler (2006). "
        "Percentage-based errors are intuitive and comparable across series of different scales, "
        "which is why sMAPE is reported, but they are unstable or undefined on series containing "
        "zeros and they penalise over- and under-forecasting asymmetrically. Both problems bite "
        "here: this project's range includes intermittent lines with frequent zero-demand days, "
        "on which any positive forecast attracts the maximum sMAPE penalty regardless of how "
        "sensible the underlying demand rate is. That is not a hypothetical concern but an "
        "effect observed directly in Section 5.2, where the worst-scoring SKUs are without "
        "exception the intermittent ones.",

        "MASE is the response Hyndman and Koehler propose, scaling the error by the in-sample "
        "error of a naive benchmark. Being scale-free and defined on series with zeros, it is "
        "the appropriate headline measure across a heterogeneous range, and its interpretation "
        "is unusually direct: a value below one indicates performance better than the benchmark. "
        "WAPE is reported alongside because it weights errors by volume, answering a different "
        "and operationally relevant question - where the ordering errors are expensive. "
        "Reporting all three is not indecision. Each answers a distinct question and none is "
        "uniquely the accuracy of the system, which is precisely the argument made in Section "
        "6.3 for treating the operational results as the primary evidence.",

        "The literature is less prescriptive about evaluating a replenishment policy, and this "
        "is where the present work has to reason from first principles rather than from "
        "precedent. Syntetos and Boylan (2006) establish the necessary half of the argument - "
        "that forecasting methods should be judged by stock-control performance - but the "
        "comparison of two policies raises a further difficulty they do not address. Any "
        "replenishment policy can reduce stockouts by holding more stock and reduce waste by "
        "holding less. Two policies observed at whatever stock levels they happen to occupy "
        "therefore differ in their operating point as well as in their logic, and a comparison "
        "between them measures both. The matched-operating-point protocol adopted in Section "
        "3.11 exists to remove that confound, and its necessity is demonstrated empirically in "
        "Section 5.6, where a naive comparison at configured settings produces an apparently "
        "contradictory result.",
    ]),

    ("2.10 Gap and positioning", [
        "The literature reviewed above divides along a line that this project is built to cross.",

        "On one side are forecasting studies that establish accuracy and stop. Schmidt, Kabir "
        "and Hoque (2022) forecast restaurant demand at item level on real data and report error "
        "metrics, with no ordering decision downstream. Aci and Yergök (2023) do the same for a "
        "university refectory, comparing regression models on calendar and menu features and "
        "reporting predictive accuracy alone; the setting is food service and the features are "
        "close to those used here, but no stock decision follows from the forecast. The M5 accuracy competition (Makridakis, "
        "Spiliotis and Assimakopoulos 2022a) does the same at very large scale and with great "
        "methodological rigour, but its evaluation is entirely in units of forecast error; no "
        "operational key performance indicator is reported, and no order is placed. The implicit "
        "assumption is that a better forecast yields a better inventory outcome - an assumption "
        "that Syntetos and Boylan (2006) show does not reliably hold.",

        "On the other side are inventory studies that assume a demand distribution rather than "
        "forecasting one. The perishable inventory literature from Nahmias (1982) onwards is "
        "largely concerned with the structure of optimal policies given a known demand process, "
        "and Babai et al. (2019) bring forecasting closer to the inventory problem by accounting "
        "for obsolescence, but stop short of a policy simulation under perishability. Syntetos "
        "and Boylan (2006) are the closest precedent for the approach taken here, judging "
        "forecasting methods by realised stock-control performance rather than by error metrics "
        "- but for spare parts rather than perishables, and without controlling for the "
        "operating point at which each method is observed.",

        "The food-service literature that does span both sides is recent and is the work this "
        "project must be positioned against most carefully. Rodrigues et al. (2024) forecast "
        "short-term demand in food catering services explicitly as a waste-reduction "
        "intervention, and report reductions in wasted meals and in unmet demand against the "
        "operations' own baseline methods - which is, in structure, the same claim this project "
        "makes. Miguéis et al. (2022) forecast daily fresh-fish demand in grocery retail with "
        "the waste-versus-availability trade-off stated as the objective, and address demand "
        "censoring, a problem this project's synthetic data does not exhibit and which Section "
        "7.3 accordingly identifies as a limitation. Seyam et al. (2025) go furthest, observing "
        "that most food demand-forecasting studies assess predictive accuracy without analysing "
        "waste impact, and closing that gap by using predicted demand to replenish inventory "
        "dynamically and measuring the resulting waste.",

        "That last study substantially narrows the gap this project might otherwise have "
        "claimed, and the claim must be narrowed accordingly rather than defended. What remains "
        "unaddressed across all of this work is the decision layer itself. These studies convert "
        "a forecast into a quantity directly, or replenish to predicted demand for the following "
        "period. None converts the forecast into an order under the constraints an actual "
        "restaurant order faces simultaneously - supplier lead time, a fixed review calendar, "
        "remaining shelf life, pack size and minimum order quantity - with safety stock sized "
        "from forecast error rather than demand variance, under a first-expired-first-out "
        "issuing rule, in a simulator that tracks inventory by dated batch.",

        "Nor does any of it compare against a manual baseline at matched operating points. This "
        "is the methodological gap rather than the technical one, and it is the more "
        "consequential of the two. Where a published comparison reports a waste reduction "
        "against an incumbent method, it reports it at whatever availability the two methods "
        "happened to achieve; the reduction therefore confounds the policy with the stock level. "
        "Section 5.6 of this work demonstrates concretely how misleading that can be, and "
        "Section 5.7 demonstrates what the comparison yields once the confound is removed.",

        "Finally, the setting itself remains under-served. The evidence that ordering is manual "
        "and departs predictably from system advice comes from supermarkets (van Donselaar et "
        "al. 2010; van Donselaar et al. 2006). The food-service work above concerns catering "
        "operations and canteens, where production planning against a known cover count differs "
        "materially from ordering ingredients against uncertain a la carte demand. None of the sources "
        "reviewed studies replenishment in a small, independent quick-service or casual-dining site, "
        "which is the setting this project addresses; Section 3.13 sets out the limits of the "
        "search behind that statement.",

        "The gap this project occupies is therefore the conjunction of three things rather than "
        "any one of them: an end-to-end pipeline from daily item-level forecast to constrained, "
        "explained order recommendation under perishability; an evaluation conducted at matched "
        "operating points against a formalised manual baseline; and a food-service setting "
        "distinct from the catering and retail contexts in which the adjacent work sits.",
    ]),
]
