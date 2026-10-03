# References used for dataset and methodology

Only sources that were actually opened (full text or abstract page) while preparing the project are listed. Items marked † were verified through an abstract page or bibliographic record only; their content beyond that page is **not** claimed.

1. P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, C. Varnier, "PRONOSTIA: An experimental platform for bearings accelerated degradation tests," *IEEE Int. Conf. on Prognostics and Health Management (PHM'12)*, Denver, CO, USA, 2012. (Author manuscript, HAL hal-00719503; read: platform, sensors, sampling, degradation patterns incl. "sudden degradations", mismatch of theoretical fault-frequency models.)
2. IEEE Reliability Society and FEMTO-ST Institute, "IEEE PHM 2012 Prognostic Challenge: Outline, Experiments, Scoring of results, Winners," 2012. (File `IEEEPHM2012-Challenge-Details.pdf` shipped with the dataset; read pp. 1-11: learning/test split, operating conditions, 25.6 kHz / 2560 samples / 10 s, file format, 20 g criterion, scoring equations, actual RULs of the test bearings.)
3. "IEEE PHM 2012 Data Challenge Dataset," repository README (dataset copy used in this project; states 6 learning + 11 test bearings and requests citation of [1]).
4. F. Huang, A. Sava, K. H. Adjallah, Z. Wang, "Fuzzy model identification based on mixture distribution analysis for bearings remaining useful life estimation using small training data set," arXiv:2012.04589. (Read title/abstract/introduction: small training data from few run-to-failure bearings; past-useful-life ratio.)
5. Z. Xu, Y. Guo, J. H. Saleh, "Remaining useful life prediction with uncertainty quantification: development of a highly accurate model for rotating machinery," arXiv:2109.11579. (Read title/abstract: benchmarks on the PHM12 bearing dataset.)
6. † Y. Lei, N. Li, L. Guo, N. Li, T. Yan, J. Lin, "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," *Mechanical Systems and Signal Processing*, vol. 104, pp. 799-834, 2018. doi:10.1016/j.ymssp.2017.11.016. (Publisher record: four processes: data acquisition, HI construction, health-stage division, RUL prediction.)
7. † X. Li, Q. Ding, J.-Q. Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks," *Reliability Engineering & System Safety*, vol. 172, pp. 1-11, 2018. (Abstract: time-window sample preparation on C-MAPSS.)
8. L. Basora, A. Viens, M. Arias Chao, X. Olive, "A benchmark on uncertainty quantification for deep learning prognostics," arXiv:2302.04730. (Read the section on piece-wise linear RUL degradation; turbofan data.)
9. † S. Jain, B. C. Wallace, "Attention is not Explanation," *NAACL-HLT 2019*, arXiv:1902.10186. (Abstract page.)
10. † M. M. R. Shamim et al., "Leakage-Robust Evaluation and Data-Scale Sensitivity of Attention-Enhanced Multi-Task Learning for Joint Fault Diagnosis and Remaining Useful Life Estimation," arXiv:2607.16493, July 2026. (Preprint, not peer-reviewed; abstract page only.)

## Statements that need a citation but could not be verified (**[CITATION NEEDED]**)
* Adam optimiser; Huber loss; LSTM / bidirectional LSTM; additive (Bahdanau-style) attention.
* A peer-reviewed CNN-LSTM-attention paper evaluated on PHM 2012.
* A paper reporting MAE/RMSE on PHM 2012.
* The origin of the piece-wise / capped RUL target for *bearings* (only the turbofan usage [8] was verified).

## Software
Python 3.12, NumPy, pandas, SciPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn. Exact versions: `artifacts/experiment_config.json`.
