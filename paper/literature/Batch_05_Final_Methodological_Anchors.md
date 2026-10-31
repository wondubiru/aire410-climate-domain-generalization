# Literature Matrix — Batch 05 Final Methodological Anchors

Verified papers in Batch 5: **19**

Batch 5 completes the 100-paper matrix by strengthening:

- remote-sensing domain adaptation/generalization reviews,
- remote-sensing augmentation,
- heterogeneous flood transfer,
- real-world flood external validation,
- explainability in flood monitoring,
- CNN and transformer architectural context,
- general robustness-oriented augmentation,
- foundational domain adaptation/generalization methods, and
- predictive uncertainty under distribution shift.

## 82. Toward generalizing to unseen domains in remote sensing: A survey

- **Citation key:** `liao2026dgsurvey`
- **Year:** 2026
- **Authors:** Muxin Liao; Meiying Liao; Zhihong Lu; Yuting Sun; Yinglong Wang; Yingqiong Peng
- **Venue:** Neurocomputing
- **DOI:** 10.1016/j.neucom.2026.134238
- **Literature block:** Remote-sensing domain-generalization survey
- **Dataset:** Survey of remote-sensing DG benchmarks
- **Platform/domain:** Remote sensing
- **Method:** Comprehensive survey of DG methods, benchmarks, datasets, and evaluation protocols
- **Evaluation:** Systematic literature review
- **Main finding:** Shows domain generalization remains comparatively underdeveloped in remote sensing and identifies evaluation under unseen domains as a central challenge.
- **Limitation:** Survey rather than task-specific empirical validation.
- **Relevance:** Current high-level reference for positioning our source-disjoint and external-domain evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.neucom.2026.134238

## 83. Domain Adaptation in Remote Sensing Image Classification: A Survey

- **Citation key:** `peng2022dasurvey`
- **Year:** 2022
- **Authors:** Jiangtao Peng; Yi Huang; Weiwei Sun; Na Chen; Yujie Ning; Qian Du
- **Venue:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **DOI:** 10.1109/JSTARS.2022.3220875
- **Literature block:** Remote-sensing domain-adaptation survey
- **Dataset:** Survey
- **Platform/domain:** Cross-domain remote sensing
- **Method:** Survey of shallow and deep domain-adaptation methods
- **Evaluation:** Literature review
- **Main finding:** Identifies acquisition conditions, scenes, time, and sensor differences as major causes of cross-domain distribution mismatch.
- **Limitation:** Primarily domain adaptation, where target-domain information may be available.
- **Relevance:** Helps distinguish domain adaptation from our harder source-only domain-generalization setting.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JSTARS.2022.3220875

## 84. Remote sensing scene classification under scarcity of labelled samples—A survey of the state-of-the-arts

- **Citation key:** `dutta2023scarcitysurvey`
- **Year:** 2023
- **Authors:** Suparna Dutta; Monidipa Das
- **Venue:** Computers & Geosciences
- **DOI:** 10.1016/j.cageo.2022.105295
- **Literature block:** Remote-sensing scene-classification survey
- **Dataset:** Review of scene-classification datasets
- **Platform/domain:** Remote-sensing scene imagery
- **Method:** Survey of data-level, model-level, and algorithm-level approaches under limited labels
- **Evaluation:** Systematic literature review
- **Main finding:** Shows scene classification is strongly affected by label scarcity, noisy samples, multimodality, and dataset limitations.
- **Limitation:** Focuses label scarcity rather than provenance-domain shift.
- **Relevance:** Provides methodological context for training on a moderate-size heterogeneous dataset.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.cageo.2022.105295

## 85. Transformers for Remote Sensing: A Systematic Review and Analysis

- **Citation key:** `wang2024transformersurvey`
- **Year:** 2024
- **Authors:** Ruikun Wang; Lei Ma; Guangjun He; Brian Alan Johnson; Ziyun Yan; Ming Chang; Ying Liang
- **Venue:** Sensors
- **DOI:** 10.3390/s24113495
- **Literature block:** Remote-sensing architecture survey
- **Dataset:** Systematic review
- **Platform/domain:** Remote sensing
- **Method:** Systematic analysis of transformer applications across remote-sensing tasks
- **Evaluation:** Systematic literature review
- **Main finding:** Finds transformers competitive across multiple remote-sensing tasks but highlights parameter and inference-cost trade-offs.
- **Limitation:** Broad transformer review rather than climate-specific cross-domain evaluation.
- **Relevance:** Positions our lightweight CNN/transfer-learning baseline relative to modern architectures.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/s24113495

## 86. A novel unsupervised adversarial domain adaptation network for remotely sensed scene classification

- **Citation key:** `liu2020adabdc`
- **Year:** 2020
- **Authors:** Wei Liu; Finlin Su
- **Venue:** International Journal of Remote Sensing
- **DOI:** 10.1080/01431161.2020.1736727
- **Literature block:** Cross-dataset remote-sensing adaptation
- **Dataset:** Four remote-sensing scene-classification benchmarks
- **Platform/domain:** High-resolution remote-sensing scenes
- **Method:** Adversarial domain adaptation with domain confusion
- **Evaluation:** Cross-dataset scene-classification transfer
- **Main finding:** Shows substantial degradation from scene/acquisition differences and reports improved transfer after adversarial domain alignment.
- **Limitation:** Uses unlabeled target-domain data and therefore addresses adaptation rather than true DG.
- **Relevance:** Strong remote-sensing precedent showing that cross-dataset shift can dominate classification error.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1080/01431161.2020.1736727

## 87. Improving remote sensing scene classification using quality-based data augmentation

- **Citation key:** `alharbi2023qss`
- **Year:** 2023
- **Authors:** Rowida Alharbi; Haikel Alhichri; Ridha Ouni; Yakoub Bazi; Maazen Alsabaan
- **Venue:** International Journal of Remote Sensing
- **DOI:** 10.1080/01431161.2023.2184213
- **Literature block:** Remote-sensing data augmentation
- **Dataset:** Remote-sensing scene datasets
- **Platform/domain:** Remote-sensing scenes
- **Method:** Quality-based Sample Selection augmentation using geometric transformations
- **Evaluation:** CNN scene-classification evaluation
- **Main finding:** Shows that augmentation quality and selection strategy can improve remote-sensing scene classification.
- **Limitation:** Primarily measures benchmark accuracy rather than source-disjoint robustness.
- **Relevance:** Direct methodological context for our conservative augmentation experiments.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1080/01431161.2023.2184213

## 88. Flood inundation monitoring using multi-source satellite imagery: a knowledge transfer strategy for heterogeneous image change detection

- **Citation key:** `zhao2024hfienet`
- **Year:** 2024
- **Authors:** Bofei Zhao; Haigang Sui; Junyi Liu; Weiyue Shi; Wentao Wang; Chuan Xu; Jindi Wang
- **Venue:** Remote Sensing of Environment
- **DOI:** 10.1016/j.rse.2024.114373
- **Literature block:** Heterogeneous flood-domain transfer
- **Dataset:** Multiple flood-event satellite datasets
- **Platform/domain:** Heterogeneous multi-source satellite imagery
- **Method:** Knowledge transfer and heterogeneous flood image change detection
- **Evaluation:** Multiple flood events with different scenarios
- **Main finding:** Demonstrates that heterogeneous sensor/source imagery requires explicit transfer mechanisms for robust flood inundation mapping.
- **Limitation:** Bi-temporal change detection differs from single-image scene classification.
- **Relevance:** Strong flood-specific evidence that source heterogeneity creates meaningful domain shift.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.rse.2024.114373

## 89. Post-flood disaster damaged houses classification based on dual-view image fusion and Concentration-Based Attention Module

- **Citation key:** `wu2024postflood`
- **Year:** 2024
- **Authors:** Luyuan Wu; Jingbo Tong; Zifa Wang; Jianhui Li; Meng Li; Hui Li; Yi Feng
- **Venue:** Sustainable Cities and Society
- **DOI:** 10.1016/j.scs.2024.105234
- **Literature block:** Post-flood image classification
- **Dataset:** Post-flood house imagery from the 2021 Zhengzhou flood disaster
- **Platform/domain:** Ground/post-disaster imagery
- **Method:** Dual-view CNN with ResNet-50 transfer learning and attention
- **Evaluation:** Post-flood house-damage classification
- **Main finding:** Shows transfer learning and attention can support fine-grained post-flood visual damage classification.
- **Limitation:** Single-event building-damage application with random train/validation/test allocation.
- **Relevance:** Useful contrast between strong within-event performance and our emphasis on external-event generalization.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.scs.2024.105234

## 90. Flood inundation monitoring with multi-source satellite imagery based on deep learning and explainable frameworks

- **Citation key:** `yu2026floodxai`
- **Year:** 2026
- **Authors:** Hongjie Yu; Yue-Ping Xu; Yintao Huang; Yen-Ming Chiang
- **Venue:** Journal of Hydrology
- **DOI:** 10.1016/j.jhydrol.2026.135409
- **Literature block:** Flood external validation and explainability
- **Dataset:** Sen1Floods11 plus real-world flood-event imagery
- **Platform/domain:** Sentinel-1, Sentinel-2, DEM, real-world flood scenes
- **Method:** U-Net multi-source flood segmentation plus Grad-CAM
- **Evaluation:** Standard benchmark evaluation followed by real-world flood monitoring
- **Main finding:** Reports stronger performance on the standard dataset than real-world flood scenes and demonstrates sensor/acquisition-dependent variation.
- **Limitation:** Segmentation rather than scene-level classification.
- **Relevance:** Exceptionally close precedent for our combination of external evaluation and Grad-CAM analysis.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.jhydrol.2026.135409

## 91. Deep Residual Learning for Image Recognition

- **Citation key:** `he2016resnet`
- **Year:** 2016
- **Authors:** Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun
- **Venue:** IEEE Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR.2016.90
- **Literature block:** CNN architecture foundation
- **Dataset:** ImageNet; CIFAR-10
- **Platform/domain:** General computer vision
- **Method:** Residual neural networks
- **Evaluation:** Large-scale image classification and detection
- **Main finding:** Introduces residual learning, enabling substantially deeper CNNs to train effectively.
- **Limitation:** General computer vision rather than remote sensing.
- **Relevance:** Foundational CNN reference and backbone used throughout remote-sensing transfer-learning literature.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR.2016.90

## 92. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows

- **Citation key:** `liu2021swin`
- **Year:** 2021
- **Authors:** Ze Liu; Yutong Lin; Yue Cao; Han Hu; Yixuan Wei; Zheng Zhang; Stephen Lin; Baining Guo
- **Venue:** IEEE/CVF International Conference on Computer Vision
- **DOI:** 10.1109/ICCV48922.2021.00986
- **Literature block:** Vision-transformer architecture
- **Dataset:** ImageNet; COCO; ADE20K
- **Platform/domain:** General computer vision
- **Method:** Hierarchical shifted-window vision transformer
- **Evaluation:** Classification, detection, and segmentation
- **Main finding:** Introduces an efficient hierarchical transformer backbone suitable for high-resolution visual tasks.
- **Limitation:** Not designed specifically for remote-sensing domain shift.
- **Relevance:** Modern architecture reference for discussing why future work could extend beyond CNN backbones.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ICCV48922.2021.00986

## 93. CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features

- **Citation key:** `yun2019cutmix`
- **Year:** 2019
- **Authors:** Sangdoo Yun; Dongyoon Han; Seong Joon Oh; Sanghyuk Chun; Junsuk Choe; Youngjoon Yoo
- **Venue:** IEEE/CVF International Conference on Computer Vision
- **DOI:** 10.1109/ICCV.2019.00612
- **Literature block:** Data augmentation / robustness
- **Dataset:** CIFAR; ImageNet
- **Platform/domain:** General computer vision
- **Method:** Patch-based image and label mixing
- **Evaluation:** Classification, localization, robustness, and OOD testing
- **Main finding:** Shows spatially mixing training images can improve classification, localization, corruption robustness, and OOD behavior.
- **Limitation:** Not remote-sensing-specific and mixed regions may alter semantic context.
- **Relevance:** Important augmentation baseline when discussing why stronger augmentation does not necessarily guarantee domain generalization.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ICCV.2019.00612

## 94. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty

- **Citation key:** `hendrycks2020augmix`
- **Year:** 2020
- **Authors:** Dan Hendrycks; Norman Mu; Ekin D. Cubuk; Barret Zoph; Justin Gilmer; Balaji Lakshminarayanan
- **Venue:** International Conference on Learning Representations
- **DOI:** No DOI recorded
- **Literature block:** Data augmentation / robustness
- **Dataset:** CIFAR-C; ImageNet-C
- **Platform/domain:** General computer vision
- **Method:** Stochastic augmentation chains with Jensen-Shannon consistency
- **Evaluation:** Clean accuracy, corruption robustness, and uncertainty calibration
- **Main finding:** Shows carefully structured augmentation can improve robustness to unforeseen corruptions and uncertainty.
- **Limitation:** Corruption robustness is not equivalent to semantic/source-domain generalization.
- **Relevance:** Useful contrast with our result that conservative augmentation did not consistently improve SD1.
- **Verification:** verified
- **Verification URL:** https://openreview.net/forum?id=S1gmrxHFvB

## 95. Domain-Adversarial Training of Neural Networks

- **Citation key:** `ganin2016dann`
- **Year:** 2016
- **Authors:** Yaroslav Ganin; Evgeniya Ustinova; Hana Ajakan; Pascal Germain; Hugo Larochelle; François Laviolette; Mario Marchand; Victor Lempitsky
- **Venue:** Journal of Machine Learning Research
- **DOI:** No DOI recorded
- **Literature block:** Domain adaptation foundation
- **Dataset:** Multiple visual/domain-adaptation benchmarks
- **Platform/domain:** General machine learning
- **Method:** Gradient-reversal domain-adversarial representation learning
- **Evaluation:** Source-labeled plus unlabeled-target domain adaptation
- **Main finding:** Establishes adversarial learning of features that remain task-discriminative while becoming domain-indistinguishable.
- **Limitation:** Requires target-domain samples and therefore is not domain generalization.
- **Relevance:** Foundational reference for understanding the distinction between DA and our source-only DG setting.
- **Verification:** verified
- **Verification URL:** https://www.jmlr.org/papers/v17/15-239.html

## 96. Deep CORAL: Correlation Alignment for Deep Domain Adaptation

- **Citation key:** `sun2016deepcoral`
- **Year:** 2016
- **Authors:** Baochen Sun; Kate Saenko
- **Venue:** European Conference on Computer Vision Workshops
- **DOI:** 10.1007/978-3-319-49409-8_35
- **Literature block:** Domain adaptation foundation
- **Dataset:** Standard domain-adaptation benchmarks
- **Platform/domain:** General computer vision
- **Method:** Second-order feature-statistics correlation alignment
- **Evaluation:** Unsupervised source-to-target domain adaptation
- **Main finding:** Shows alignment of feature correlations can reduce domain discrepancy in deep networks.
- **Limitation:** Uses target-domain data and is therefore unsuitable as a pure DG method.
- **Relevance:** Important methodological reference for feature-statistics approaches related conceptually to style-based DG.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1007/978-3-319-49409-8_35

## 97. Learning to Generalize: Meta-Learning for Domain Generalization

- **Citation key:** `li2018mldg`
- **Year:** 2018
- **Authors:** Da Li; Yongxin Yang; Yi-Zhe Song; Timothy M. Hospedales
- **Venue:** Proceedings of the AAAI Conference on Artificial Intelligence
- **DOI:** 10.1609/aaai.v32i1.11596
- **Literature block:** Domain generalization foundation
- **Dataset:** Cross-domain image benchmarks
- **Platform/domain:** General computer vision
- **Method:** Meta-learning that simulates train/test domain shift during optimization
- **Evaluation:** Training-domain meta-splits with unseen-domain testing
- **Main finding:** Demonstrates that explicitly simulating domain shift during learning can improve unseen-domain performance.
- **Limitation:** Generic benchmarks rather than remote-sensing imagery.
- **Relevance:** Foundational DG approach showing how training can explicitly optimize for unseen-domain behavior.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1609/aaai.v32i1.11596

## 98. SWAD: Domain Generalization by Seeking Flat Minima

- **Citation key:** `cha2021swad`
- **Year:** 2021
- **Authors:** Junbum Cha; Sanghyuk Chun; Kyungjae Lee; Han-Cheol Cho; Seunghyun Park; Yunsung Lee; Sungrae Park
- **Venue:** Advances in Neural Information Processing Systems
- **DOI:** No DOI recorded
- **Literature block:** Domain generalization optimization
- **Dataset:** PACS; VLCS; OfficeHome; TerraIncognita; DomainNet
- **Platform/domain:** General computer vision
- **Method:** Stochastic Weight Averaging Densely
- **Evaluation:** Five DomainBed domain-generalization benchmarks
- **Main finding:** Shows flatter minima can improve unseen-domain generalization over vanilla ERM.
- **Limitation:** Generic image domains rather than remote sensing.
- **Relevance:** Useful methodological contrast with our finding that several DG modifications do not consistently beat ERM.
- **Verification:** verified
- **Verification URL:** https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html

## 99. Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization

- **Citation key:** `rame2022fishr`
- **Year:** 2022
- **Authors:** Alexandre Rame; Corentin Dancette; Matthieu Cord
- **Venue:** Proceedings of the 39th International Conference on Machine Learning
- **DOI:** No DOI recorded
- **Literature block:** Domain generalization optimization
- **Dataset:** DomainBed benchmarks
- **Platform/domain:** General computer vision
- **Method:** Gradient-variance invariance across source domains
- **Evaluation:** Out-of-distribution DomainBed evaluation
- **Main finding:** Shows matching domain-level gradient variances can improve OOD generalization under controlled benchmarks.
- **Limitation:** General vision domains and increased training complexity.
- **Relevance:** Provides a stronger DG comparator and emphasizes the importance of controlled ERM comparisons.
- **Verification:** verified
- **Verification URL:** https://proceedings.mlr.press/v162/rame22a.html

## 100. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

- **Citation key:** `lakshminarayanan2017ensembles`
- **Year:** 2017
- **Authors:** Balaji Lakshminarayanan; Alexander Pritzel; Charles Blundell
- **Venue:** Advances in Neural Information Processing Systems
- **DOI:** No DOI recorded
- **Literature block:** Predictive uncertainty
- **Dataset:** Multiple regression/classification benchmarks; ImageNet
- **Platform/domain:** General machine learning
- **Method:** Deep ensembles
- **Evaluation:** Calibration plus in-distribution and out-of-distribution uncertainty
- **Main finding:** Shows independently trained ensembles produce strong predictive uncertainty and increased uncertainty on unfamiliar distributions.
- **Limitation:** Requires training and storing multiple models.
- **Relevance:** Provides an established future direction for converting our three-seed variability analysis into explicit predictive uncertainty estimation.
- **Verification:** verified
- **Verification URL:** https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
