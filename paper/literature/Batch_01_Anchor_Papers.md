# Literature Matrix — Batch 01 Anchor Papers

Verified papers in Batch 1: **21**

This anchor batch supports:

- climate/disaster aerial classification,
- UAV and external flood datasets,
- transfer learning in environmental remote sensing,
- remote-sensing domain shift and generalization,
- MixStyle and RSC,
- rigorous ERM/DG benchmarking,
- Grad-CAM, and
- shortcut learning.

## 1. Uncertainty-Aware Cross-Domain Few-Shot Scene Classification from Remote Sensing Imagery

- **Citation key:** `ning2026uncertainty`
- **Year:** 2026
- **Authors:** Zifan Ning; Can Li; He Chen; Guangyao Zhou; Shanghang Zhang; Lianlin Li; Yin Zhuang
- **Venue:** Remote Sensing
- **DOI:** 10.3390/rs18081233
- **Literature block:** Cross-domain remote sensing
- **Dataset:** Cross-domain few-shot RS benchmarks
- **Platform/domain:** Remote-sensing scene domains
- **Method:** Uncertainty-aware cross-domain framework
- **Main finding:** Uses predictive uncertainty to improve robustness and knowledge transfer under remote-sensing domain discrepancy.
- **Limitation:** Few-shot target-domain setting differs from our source-only DG protocol.
- **Relevance to our work:** Very recent evidence that domain discrepancy and uncertainty remain central RS classification problems.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/rs18081233

## 2. Analysing Satellite Imagery Classification under Spatial Domain Shift across Geographic Regions

- **Citation key:** `alemadi2025dsg`
- **Year:** 2025
- **Authors:** Sara A. Al-Emadi; Yin Yang; Ferda Ofli
- **Venue:** International Journal of Computer Vision
- **DOI:** 10.1007/s11263-025-02518-z
- **Literature block:** Remote-sensing domain generalization
- **Dataset:** DSGR
- **Platform/domain:** Satellite imagery across geographic regions
- **Method:** DG benchmark; ERM; modern DG methods; foundation models
- **Main finding:** Shows significant geographic distribution shift and reports that carefully configured ERM with strong representations can remain highly competitive with modern DG methods.
- **Limitation:** Satellite geographic shift differs from our composite UAV/satellite/source shift.
- **Relevance to our work:** One of the closest recent studies to our central finding that DG interventions do not automatically beat ERM.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1007/s11263-025-02518-z

## 3. CoDEx: Combining Domain Expertise for Spatial Generalization in Satellite Image Analysis

- **Citation key:** `kuriyal2025codex`
- **Year:** 2025
- **Authors:** Abhishek Kuriyal; Elliot Vincent; Mathieu Aubry; Loic Landrieu
- **Venue:** CVPR Workshops (EarthVision)
- **DOI:** 10.1109/CVPRW67362.2025.00206
- **Literature block:** Remote-sensing domain generalization
- **Dataset:** DynamicEarthNet; MUDS; OSCD; FMoW
- **Platform/domain:** Satellite imagery across spatial domains
- **Method:** Domain-expert ensemble / selection framework
- **Main finding:** Shows domain-expert specialization and selection can improve spatial generalization across satellite datasets.
- **Limitation:** Different task/data regime and a more specialized multi-expert architecture.
- **Relevance to our work:** Recent spatial-generalization comparison for satellite imagery.
- **Verification:** verified
- **Verification URL:** https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Kuriyal_CoDEx_Combining_Domain_Expertise_for_Spatial_Generalization_in_Satellite_Image_CVPRW_2025_paper.html

## 4. A Shift Reduction Domain Generalization Network for Hyperspectral Image Cross-Domain Classification

- **Citation key:** `qi2025srdgnet`
- **Year:** 2025
- **Authors:** Yunxiao Qi; Dongyang Liu; Junping Zhang; Ye Zhang
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2025.3599172
- **Literature block:** Remote-sensing domain generalization
- **Dataset:** Four public hyperspectral datasets
- **Platform/domain:** Cross-domain hyperspectral imagery
- **Method:** Shift Reduction Domain Generalization Network
- **Main finding:** Proposes explicit feature/domain-shift reduction for hyperspectral cross-domain classification.
- **Limitation:** Hyperspectral spectral-spatial setting differs from RGB aerial scene classification.
- **Relevance to our work:** Very recent remote-sensing DG evidence demonstrating continued interest in unseen-domain robustness.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2025.3599172

## 5. Optimizing Deep Learning Models for Climate-Related Natural Disaster Detection from UAV Images and Remote Sensing Data

- **Citation key:** `vanexel2025climate`
- **Year:** 2025
- **Authors:** Kim VanExel; Samendra Sherchan; Siyan Liu
- **Venue:** Journal of Imaging
- **DOI:** 10.3390/jimaging11020032
- **Literature block:** Climate/disaster classification
- **Dataset:** Climate Change Dataset; 6,334 aerial images
- **Platform/domain:** UAV + satellite/remote-sensing imagery
- **Method:** CNN; DenseNet201; VGG16; ResNet50
- **Main finding:** Demonstrates strong deep-learning performance for Flooded, Desert, and Neither classification on the Climate Change Dataset.
- **Limitation:** Does not provide the provenance-aware, source-disjoint, multi-seed, external-validation failure analysis used in our study.
- **Relevance to our work:** Original publication for the exact primary dataset and the direct baseline literature our work extends.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/jimaging11020032

## 6. Multi-Source Collaborative Domain Generalization for Cross-Scene Remote Sensing Image Classification

- **Citation key:** `han2024mscdg`
- **Year:** 2024
- **Authors:** Z. Han; C. Zhang; L. Gao; Z. Zeng; M. K. Ng; B. Zhang; J. Chanussot
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2024.3478385
- **Literature block:** Remote-sensing domain generalization
- **Dataset:** Three public multisource RS datasets
- **Platform/domain:** Multisource remote-sensing scenes
- **Method:** Multi-source collaborative DG with adversarial augmentation and model diversification
- **Main finding:** Reports improved cross-scene generalization through multi-source augmentation and diversification.
- **Limitation:** Uses a specialized multisource DG architecture and different data assumptions from our simpler interventions.
- **Relevance to our work:** Direct recent remote-sensing DG comparison.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2024.3478385

## 7. Semisupervised Cross-Domain Remote Sensing Scene Classification via Category-Level Feature Alignment Network

- **Citation key:** `li2024categoryalign`
- **Year:** 2024
- **Authors:** Yang Li; Zhang Li; Ang Su; Kun Wang; Zi Wang; Qifeng Yu
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2024.3392984
- **Literature block:** Remote-sensing domain adaptation
- **Dataset:** Cross-domain remote-sensing scene datasets
- **Platform/domain:** Cross-domain RS scenes
- **Method:** Category-level feature alignment network
- **Main finding:** Uses category-level alignment to reduce cross-domain feature discrepancies.
- **Limitation:** Uses labeled/unlabeled target-domain information, unlike our target-unseen source-disjoint DG setting.
- **Relevance to our work:** Recent comparison showing another strategy for cross-domain scene robustness.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2024.3392984

## 8. Transfer Learning in Environmental Remote Sensing

- **Citation key:** `ma2024transfer`
- **Year:** 2024
- **Authors:** Yuchi Ma; Shuo Chen; Stefano Ermon; David B. Lobell
- **Venue:** Remote Sensing of Environment
- **DOI:** 10.1016/j.rse.2023.113924
- **Literature block:** Transfer learning / domain shift
- **Dataset:** Review across environmental RS applications
- **Platform/domain:** Environmental remote sensing across sensors/domains
- **Method:** Review of transfer-learning approaches and domain shifts
- **Main finding:** Systematizes domain-shift types and transfer-learning strategies in environmental remote sensing.
- **Limitation:** Review rather than a controlled benchmark of our specific climate-scene task.
- **Relevance to our work:** Primary theoretical support for using pretrained backbones and discussing environmental domain shift.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.rse.2023.113924

## 9. A Lightweight Deep Learning Model for Aerial Image Classification and Its Application to UAV-based Disaster Management

- **Citation key:** `xinjie2024lightweight`
- **Year:** 2024
- **Authors:** Deng Xinjie; Burhan Khan; Fazal Ghaffar; Yit Hong Choo; Tao Zhou
- **Venue:** Procedia Computer Science
- **DOI:** 10.1016/j.procs.2024.09.395
- **Literature block:** UAV disaster classification
- **Dataset:** AIDER
- **Platform/domain:** UAV aerial imagery
- **Method:** Lightweight multi-scale/depthwise convolution model
- **Main finding:** Demonstrates a lightweight feature-extraction design for UAV disaster-management classification.
- **Limitation:** Does not center source-domain generalization or external cross-source validation.
- **Relevance to our work:** Recent disaster-classification comparison and deployment-oriented UAV context.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.procs.2024.09.395

## 10. Graph Embedding Interclass Relation-Aware Adaptive Network for Cross-Scene Classification of Multisource Remote Sensing Data

- **Citation key:** `yang2024geira`
- **Year:** 2024
- **Authors:** Teng Yang; Song Xiao; Jiahui Qu; Wenqian Dong; Qian Du; Yunsong Li
- **Venue:** IEEE Transactions on Image Processing
- **DOI:** 10.1109/TIP.2024.3422881
- **Literature block:** Remote-sensing domain adaptation
- **Dataset:** Multisource remote-sensing data
- **Platform/domain:** Cross-scene multisource remote sensing
- **Method:** Graph embedding and interclass relation-aware adaptation
- **Main finding:** Models class relationships and domain discrepancies for multisource cross-scene classification.
- **Limitation:** Unsupervised DA relies on target-domain samples.
- **Relevance to our work:** Supports the importance of multisource domain structure and class-aware cross-domain evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TIP.2024.3422881

## 11. Open-set Domain Adaptation for Scene Classification Using Multi-adversarial Learning

- **Citation key:** `zheng2024openset`
- **Year:** 2024
- **Authors:** Juepeng Zheng; Yibin Wen; Mengxuan Chen; Shuai Yuan; Weijia Li; Yi Zhao; Wenzhao Wu; Lixian Zhang; Runmin Dong; Haohuan Fu
- **Venue:** ISPRS Journal of Photogrammetry and Remote Sensing
- **DOI:** 10.1016/j.isprsjprs.2024.01.015
- **Literature block:** Remote-sensing domain adaptation
- **Dataset:** Three public remote-sensing scene datasets
- **Platform/domain:** Cross-domain remote-sensing scenes
- **Method:** Multi-Adversarial Open-Set Domain Adaptation Network
- **Main finding:** Addresses cross-domain remote-sensing classification when target domains include unknown classes.
- **Limitation:** Open-set DA assumes access to target-domain data and a different label-space problem from our DG protocol.
- **Relevance to our work:** Helps distinguish DG from broader real-world cross-domain adaptation settings.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.isprsjprs.2024.01.015

## 12. Cross-Domain Classification Based on Frequency Component Adaptation for Remote Sensing Images

- **Citation key:** `zhu2024fcan`
- **Year:** 2024
- **Authors:** Peng Zhu; Xiangrong Zhang; Xiao Han; Xina Cheng; Jing Gu; Puhua Chen; Licheng Jiao
- **Venue:** Remote Sensing
- **DOI:** 10.3390/rs16122134
- **Literature block:** Remote-sensing domain adaptation
- **Dataset:** AID; CLRS; MLRSN; RSSCN7; NWPU-RESISC45; NaSC-TG2
- **Platform/domain:** Cross-scene and cross-sensor remote sensing
- **Method:** Frequency Component Adaptation Network (FCAN)
- **Main finding:** Shows frequency-aware feature alignment can improve cross-domain remote-sensing scene classification.
- **Limitation:** Domain adaptation uses target-domain information; our primary DG setting assumes unseen target domains.
- **Relevance to our work:** Useful contrast between domain adaptation and true target-unseen domain generalization.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/rs16122134

## 13. Domain Generalization: A Survey

- **Citation key:** `zhou2023dgsurvey`
- **Year:** 2023
- **Authors:** Kaiyang Zhou; Ziwei Liu; Yu Qiao; Tao Xiang; Chen Change Loy
- **Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence
- **DOI:** 10.1109/TPAMI.2022.3195549
- **Literature block:** Domain generalization theory
- **Dataset:** Survey
- **Platform/domain:** Multiple application domains
- **Method:** DG taxonomy and review
- **Main finding:** Defines DG formally and organizes methods including alignment, meta-learning, augmentation, and ensembles.
- **Limitation:** Broad survey rather than remote-sensing-specific evidence.
- **Relevance to our work:** Primary reference for defining DG and distinguishing it from transfer learning and domain adaptation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TPAMI.2022.3195549

## 14. In Search of Lost Domain Generalization

- **Citation key:** `gulrajani2021domainbed`
- **Year:** 2021
- **Authors:** Ishaan Gulrajani; David Lopez-Paz
- **Venue:** International Conference on Learning Representations
- **DOI:** No DOI recorded
- **Literature block:** Domain generalization benchmarking
- **Dataset:** DomainBed: seven DG benchmarks
- **Platform/domain:** General computer-vision domains
- **Method:** DomainBed benchmark; ERM and 14 DG algorithms
- **Main finding:** Shows carefully implemented and tuned ERM is a very strong baseline and that many DG gains shrink under controlled evaluation.
- **Limitation:** Benchmarks are not remote-sensing-specific.
- **Relevance to our work:** Direct methodological support for our finding that DG-AUG-V2 and RSC do not consistently beat ERM.
- **Verification:** verified
- **Verification URL:** https://openreview.net/forum?id=lQdXeXDoWtI

## 15. FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding

- **Citation key:** `rahnemoonfar2021floodnet`
- **Year:** 2021
- **Authors:** Maryam Rahnemoonfar; Tashnim Chowdhury; Argho Sarkar; Debvrat Varshney; Mohammad Yari; Robin R. Murphy
- **Venue:** IEEE Access
- **DOI:** 10.1109/ACCESS.2021.3090981
- **Literature block:** External validation / flood datasets
- **Dataset:** FloodNet
- **Platform/domain:** Post-flood UAV imagery
- **Method:** Benchmark for image classification, semantic segmentation, and visual question answering
- **Main finding:** Introduces high-resolution post-Hurricane-Harvey UAV imagery and explicitly distinguishes natural water from flooded roads/buildings.
- **Limitation:** Established 2021 benchmark; classification taxonomy does not directly include our Desert class.
- **Relevance to our work:** Our independent external UAV benchmark and the source of the segmentation masks used for shortcut analysis.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ACCESS.2021.3090981

## 16. Domain Generalization with MixStyle

- **Citation key:** `zhou2021mixstyle`
- **Year:** 2021
- **Authors:** Kaiyang Zhou; Yongxin Yang; Yu Qiao; Tao Xiang
- **Venue:** International Conference on Learning Representations
- **DOI:** No DOI recorded
- **Literature block:** Domain generalization methods
- **Dataset:** Multiple DG benchmarks
- **Platform/domain:** General computer-vision domains
- **Method:** MixStyle
- **Main finding:** Mixes instance-level feature statistics to synthesize implicit domains and increase source-domain diversity.
- **Limitation:** Original benchmarks differ substantially from remote sensing and climate imagery.
- **Relevance to our work:** Foundational citation for the MixStyle variants we tested.
- **Verification:** verified
- **Verification URL:** https://openreview.net/forum?id=6xHJ37MVxxp

## 17. Shortcut Learning in Deep Neural Networks

- **Citation key:** `geirhos2020shortcut`
- **Year:** 2020
- **Authors:** Robert Geirhos; Jörn-Henrik Jacobsen; Claudio Michaelis; Richard Zemel; Wieland Brendel; Matthias Bethge; Felix A. Wichmann
- **Venue:** Nature Machine Intelligence
- **DOI:** 10.1038/s42256-020-00257-z
- **Literature block:** Shortcut learning / robustness
- **Dataset:** Perspective / multiple examples
- **Platform/domain:** General machine learning / vision
- **Method:** Shortcut-learning framework
- **Main finding:** Frames many benchmark-to-real-world failures as models learning predictive but non-robust shortcut decision rules.
- **Limitation:** General conceptual framework rather than remote-sensing specific quantitative evidence.
- **Relevance to our work:** Core conceptual framework for interpreting our ordinary-water/non-flooded-road contextual shortcut evidence.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1038/s42256-020-00257-z

## 18. Self-Challenging Improves Cross-Domain Generalization

- **Citation key:** `huang2020rsc`
- **Year:** 2020
- **Authors:** Zeyi Huang; Haohan Wang; Eric P. Xing; Dong Huang
- **Venue:** European Conference on Computer Vision
- **DOI:** 10.1007/978-3-030-58536-5_8
- **Literature block:** Domain generalization methods
- **Dataset:** Multiple DG benchmarks
- **Platform/domain:** General computer-vision domains
- **Method:** Representation Self-Challenging (RSC)
- **Main finding:** Suppresses dominant predictive features during training to encourage alternative features that may generalize better across domains.
- **Limitation:** Original evidence comes from standard DG benchmarks, not heterogeneous remote-sensing sources.
- **Relevance to our work:** Foundational citation for our RSC implementation and multi-seed evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1007/978-3-030-58536-5_8

## 19. EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion

- **Citation key:** `kyrkou2020emergencynet`
- **Year:** 2020
- **Authors:** Christos Kyrkou; Theocharis Theocharides
- **Venue:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **DOI:** 10.1109/JSTARS.2020.2969809
- **Literature block:** UAV disaster classification
- **Dataset:** AIDER
- **Platform/domain:** UAV/drone emergency imagery
- **Method:** Lightweight CNN with atrous convolutional feature fusion
- **Main finding:** Shows efficient UAV disaster classification can retain strong accuracy while substantially reducing computational requirements.
- **Limitation:** Focuses on efficient within-benchmark classification rather than source-disjoint robustness.
- **Relevance to our work:** Important UAV disaster-classification baseline and application motivation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JSTARS.2020.2969809

## 20. MobileNetV2: Inverted Residuals and Linear Bottlenecks

- **Citation key:** `sandler2018mobilenetv2`
- **Year:** 2018
- **Authors:** Mark Sandler; Andrew Howard; Menglong Zhu; Andrey Zhmoginov; Liang-Chieh Chen
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR.2018.00474
- **Literature block:** Model architecture / transfer learning
- **Dataset:** ImageNet and downstream vision tasks
- **Platform/domain:** General computer vision
- **Method:** MobileNetV2
- **Main finding:** Introduces inverted residual blocks and linear bottlenecks for efficient deep feature extraction.
- **Limitation:** Architecture paper does not address remote-sensing domain shift directly.
- **Relevance to our work:** Foundational citation for the pretrained backbone used throughout our primary transfer-learning experiments.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR.2018.00474

## 21. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization

- **Citation key:** `selvaraju2017gradcam`
- **Year:** 2017
- **Authors:** Ramprasaath R. Selvaraju; Michael Cogswell; Abhishek Das; Ramakrishna Vedantam; Devi Parikh; Dhruv Batra
- **Venue:** IEEE International Conference on Computer Vision
- **DOI:** 10.1109/ICCV.2017.74
- **Literature block:** Explainability
- **Dataset:** Multiple vision tasks/datasets
- **Platform/domain:** General computer vision
- **Method:** Grad-CAM
- **Main finding:** Uses target-class gradients flowing into convolutional features to produce spatial explanations for CNN predictions.
- **Limitation:** Saliency/localization does not establish causal dependence.
- **Relevance to our work:** Foundational method for our 27 frozen Grad-CAM heatmaps.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ICCV.2017.74
