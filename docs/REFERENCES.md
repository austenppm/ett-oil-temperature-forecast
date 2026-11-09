# References

Sources that informed the design decisions of this PoC. Numbers quoted in the report
come from our own experiments unless a reference is cited next to them.

## Data
1. H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, W. Zhang. *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021. https://arxiv.org/abs/2012.07436
2. ETDataset (ETT-small), CC BY-ND 4.0. https://github.com/zhouhaoyi/ETDataset

## Transformer thermal physics and condition monitoring
3. IEC 60076-7:2018. *Power transformers – Part 7: Loading guide for mineral-oil-immersed power transformers* (first-order top-oil thermal model; losses ∝ load current²).
4. IEEE Std C57.91-2011. *IEEE Guide for Loading Mineral-Oil-Immersed Transformers and Step-Voltage Regulators.*
5. D. Tembo, F. Bragone, T. Laneryd, M. Barreau, K. Morozovska. *Data-Driven vs Traditional Approaches to Power Transformer's Top-Oil Temperature Estimation.* 2025. https://arxiv.org/abs/2501.16831 (data-driven models vs IEC 60076-7; quantile intervals under-cover; recommends conformalized quantile regression).
6. M. Djamali, S. Tenbohlen. *Malfunction Detection of the Cooling System in Air-Forced Power Transformers Using Online Thermal Monitoring.* IEEE Trans. Power Delivery, 2017 (residual between calculated and measured top-oil temperature for fault detection).
7. CIGRE Technical Brochure. *Digital twins for power transformers: concept and future perspectives.* Electra, 2026. https://electra.cigre.org/347-august-2026/technical-brochures/digital-twins-for-power-transformers-concept-and-future-perspectives

## Forecasting methodology and evaluation
8. H. Hewamalage, K. Ackermann, C. Bergmeir. *Forecast evaluation for data scientists: common pitfalls and best practices.* Data Mining and Knowledge Discovery 37, 2023. https://doi.org/10.1007/s10618-022-00894-5 (naive benchmarks, rolling-origin evaluation, leakage, MAPE near zero, ML models do not extrapolate).
9. Y. Cui, J. Xie, K. Zheng. *Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-Series Forecasting.* CIKM 2021. https://arxiv.org/abs/2103.16349
10. A. Zeng, M. Chen, L. Zhang, Q. Xu. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. https://arxiv.org/abs/2205.13504 (linear models on raw windows beat Transformers on ETT; basis of our "WindowLinear" baseline).
11. F. X. Diebold, R. S. Mariano. *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics, 1995; D. Harvey, S. Leybourne, P. Newbold. *Testing the equality of prediction mean squared errors.* International Journal of Forecasting, 1997 (small-sample correction).
12. L. Tashman. *Out-of-sample tests of forecasting accuracy: an analysis and review.* International Journal of Forecasting, 2000 (rolling-origin evaluation).
13. T. Januschowski et al. *Forecasting with trees.* International Journal of Forecasting, 2022. https://cdn.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf (gradient-boosted trees in the M5 competition).
14. J. Smith, K. F. Wallis. *A Simple Explanation of the Forecast Combination Puzzle.* Oxford Bulletin of Economics and Statistics, 2009 (equal-weight ensembles).

## Uncertainty quantification
15. Y. Romano, E. Patterson, E. Candès. *Conformalized Quantile Regression.* NeurIPS 2019. https://arxiv.org/abs/1905.03222
16. K. Stankeviciute, A. Alaa, M. van der Schaar. *Conformal Time-series Forecasting.* NeurIPS 2021; M. Zaffran et al. *Adaptive Conformal Predictions for Time Series.* ICML 2022 (exchangeability caveats for time series).

## Operating practice (context only; not used for numerical claims)
17. Typical top-oil alarm/trip settings for oil-immersed transformers are in the 80–110 °C range (IEEE C57.91 guidance and utility practice), far above the OT values in ETT. This is why the PoC uses a proxy alarm level and asks the client for the real one.
