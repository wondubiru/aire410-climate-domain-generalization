# Literature Matrix — Batch 03 Robustness, Explainability, and Foundation Models

Verified papers in Batch 3: **20**

Batch 3 strengthens:

- remote-sensing explainability,
- uncertainty and confidence calibration,
- real-world distribution-shift evaluation,
- OOD detection,
- geographic trustworthiness,
- self-supervised EO representation learning, and
- modern remote-sensing foundation models.

## 42. Opening the Black Box: A Systematic Review on Explainable Artificial Intelligence in Remote Sensing

- **Citation key:** `hohl2024xaireview`
- **Year:** 2024
- **Authors:** Adrian Höhl; Ivica Obadic; Miguel-Ángel Fernández-Torres; Hiba Najjar; Dario Augusto Borges Oliveira; Zeynep Akata; Andreas Dengel; Xiao Xiang Zhu
- **Venue:** IEEE Geoscience and Remote Sensing Magazine
- **DOI:** 10.1109/MGRS.2024.3467001
- **Literature block:** Remote-sensing explainability
- **Dataset:** Systematic review
- **Platform/domain:** Remote sensing / Earth observation
- **Method:** Systematic review of XAI methods, interpretation practices, and evaluation
- **Evaluation:** Systematic literature review
- **Main finding:** Identifies trends, scientific uses, evaluation methods, and major challenges for XAI in remote sensing.
- **Limitation:** Review paper; does not provide task-specific causal evidence for saliency maps.
- **Relevance:** Primary remote-sensing-specific reference for explaining why Grad-CAM should be interpreted cautiously.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/MGRS.2024.3467001

## 43. Self-Organising Explainable Multi-View Representation Learning for Remote Sensing Scene Classification

- **Citation key:** `gu2026explainablemultiview`
- **Year:** 2026
- **Authors:** Xiaowei Gu; Abdulrahman Kerim; Jinghao Zhang; Jungong Han; Qiang Shen; Peter M. Atkinson; Ce Zhang
- **Venue:** Applied Soft Computing
- **DOI:** 10.1016/j.asoc.2026.114579
- **Literature block:** Explainable scene classification
- **Dataset:** Optimal-31; WHU-RS; UCMerced; AID; RSI-CB256; PatternNet
- **Platform/domain:** Remote-sensing scene imagery
- **Method:** Transparent multi-view representation learning with evolving fuzzy neural encoders
- **Evaluation:** Six benchmark scene-classification datasets
- **Main finding:** Combines strong scene classification with transparent multi-view representation learning.
- **Limitation:** High benchmark accuracy does not directly establish source-disjoint or external-domain robustness.
- **Relevance:** Recent evidence connecting scene classification, generalization, and interpretability.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.asoc.2026.114579

## 44. From Faithfulness to Validity: Explainable Artificial Intelligence in Remote Sensing

- **Citation key:** `dritsas2026validity`
- **Year:** 2026
- **Authors:** E. Dritsas; M. Trigka
- **Venue:** IEEE Geoscience and Remote Sensing Magazine
- **DOI:** 10.1109/MGRS.2026.3717217
- **Literature block:** Remote-sensing explainability
- **Dataset:** Review / conceptual analysis
- **Platform/domain:** Remote sensing / Earth observation
- **Method:** Domain-constrained framework for explanation validity
- **Evaluation:** Review of RS-specific XAI assumptions and failure modes
- **Main finding:** Argues that explanations can appear faithful yet remain scientifically invalid when RS-specific spatial, spectral, temporal, and physical structure is ignored.
- **Limitation:** Conceptual/review contribution rather than controlled empirical evaluation.
- **Relevance:** Direct support for our cautious claim that Grad-CAM is explanatory evidence, not causal proof.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/MGRS.2026.3717217

## 45. Assessing Conformal Prediction for Remote Sensing: Application to Local Climate Zone Classification

- **Citation key:** `schweden2026conformal`
- **Year:** 2026
- **Authors:** Christoph Schweden; Peter Jung; Xiao Xiang Zhu
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2026.3719780
- **Literature block:** Uncertainty / calibration
- **Dataset:** So2Sat LCZ42
- **Platform/domain:** Satellite land-cover classification
- **Method:** Conformal prediction; uncertainty-aware training; active learning
- **Evaluation:** Prediction-set coverage, uncertainty, and active learning
- **Main finding:** Shows conformal prediction can provide statistically grounded prediction sets and useful uncertainty signals for remote-sensing classification.
- **Limitation:** Local Climate Zone classification differs from our climate-disaster taxonomy.
- **Relevance:** Supports future uncertainty/calibration analysis and the need to go beyond confidence scores.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2026.3719780

## 46. Uncertainty-Aware Dynamic Multi-Prototype Representation for Out-of-Distribution Detection in SAR Imagery

- **Citation key:** `zhou2026sarood`
- **Year:** 2026
- **Authors:** Xiaoyan Zhou; Tao Tang; Janne Heikkilä; Li Liu; Yuli Sun; Gangyao Kuang
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2026.3716820
- **Literature block:** OOD detection / uncertainty
- **Dataset:** SAR target-recognition/OOD datasets
- **Platform/domain:** Synthetic aperture radar imagery
- **Method:** Uncertainty-aware dynamic multi-prototype representation
- **Evaluation:** In-distribution classification + OOD detection
- **Main finding:** Explicitly models uncertainty and prototypes to improve recognition of out-of-distribution SAR samples.
- **Limitation:** SAR target recognition differs from RGB aerial scene classification.
- **Relevance:** Recent remote-sensing evidence that high confidence must be complemented by explicit OOD-awareness.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2026.3716820

## 47. Trustworthy Transfer Learning in GeoAI: Assessing Pretrained Models for Rooftop Solar PV Mapping Under Geographic Domain Shift

- **Citation key:** `hossain2026trustworthy`
- **Year:** 2026
- **Authors:** Tahsin Hossain; Tan Yigitcanlar; Niklas Tilly; Filip Biljecki
- **Venue:** Journal of Geovisualization and Spatial Analysis
- **DOI:** 10.1007/s41651-026-00293-0
- **Literature block:** Trustworthy geographic transfer
- **Dataset:** US-pretrained model transferred to Brisbane imagery
- **Platform/domain:** High-resolution aerial imagery
- **Method:** Transfer learning with independent geographic evaluation
- **Evaluation:** Development + independent test evaluation under geographic domain shift
- **Main finding:** Shows local transfer learning can improve performance while residual false positives, misses, and uncertainty remain important for trustworthy deployment.
- **Limitation:** Single rooftop-PV application and limited geographic test scope.
- **Relevance:** Very recent direct support for our argument that improved accuracy alone is insufficient evidence of robustness.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1007/s41651-026-00293-0

## 48. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift

- **Citation key:** `ovadia2019uncertainty`
- **Year:** 2019
- **Authors:** Yaniv Ovadia; Emily Fertig; Jie Ren; Zachary Nado; D. Sculley; Sebastian Nowozin; Joshua Dillon; Balaji Lakshminarayanan; Jasper Snoek
- **Venue:** Advances in Neural Information Processing Systems
- **DOI:** No DOI recorded
- **Literature block:** Uncertainty under dataset shift
- **Dataset:** Multiple classification benchmarks
- **Platform/domain:** General machine learning
- **Method:** Large-scale comparison of uncertainty-estimation methods
- **Evaluation:** Accuracy and calibration under increasing dataset shift
- **Main finding:** Shows predictive uncertainty and calibration can degrade substantially under dataset shift; ensembles are among the strongest approaches studied.
- **Limitation:** General vision/ML benchmark rather than remote sensing.
- **Relevance:** Foundational support for treating high confidence on shifted UAV imagery cautiously.
- **Verification:** verified
- **Verification URL:** https://papers.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

## 49. On Calibration of Modern Neural Networks

- **Citation key:** `guo2017calibration`
- **Year:** 2017
- **Authors:** Chuan Guo; Geoff Pleiss; Yu Sun; Kilian Q. Weinberger
- **Venue:** Proceedings of the 34th International Conference on Machine Learning
- **DOI:** No DOI recorded
- **Literature block:** Confidence calibration
- **Dataset:** Multiple image/text classification datasets
- **Platform/domain:** General machine learning
- **Method:** Calibration analysis; temperature scaling
- **Evaluation:** Expected calibration error and classification accuracy
- **Main finding:** Shows modern deep networks are often overconfident and that temperature scaling is an effective post-hoc calibration baseline.
- **Limitation:** In-distribution calibration does not guarantee calibration under domain shift.
- **Relevance:** Important theoretical reference for interpreting probability outputs and future calibration work.
- **Verification:** verified
- **Verification URL:** https://proceedings.mlr.press/v70/guo17a.html

## 50. WILDS: A Benchmark of in-the-Wild Distribution Shifts

- **Citation key:** `koh2021wilds`
- **Year:** 2021
- **Authors:** Pang Wei Koh; Shiori Sagawa; Henrik Marklund; Sang Michael Xie; Marvin Zhang; Akshay Balsubramani; Weihua Hu; Michihiro Yasunaga; Richard Lanas Phillips; Irena Gao; Tony Lee; Etienne David; Ian Stavness; Wei Guo; Berton Earnshaw; Imran Haque; Sara Beery; Jure Leskovec; Anshul Kundaje; Emma Pierson; Sergey Levine; Chelsea Finn; Percy Liang
- **Venue:** Proceedings of the 38th International Conference on Machine Learning
- **DOI:** No DOI recorded
- **Literature block:** Real-world distribution shift
- **Dataset:** WILDS; 10 real-world shift datasets
- **Platform/domain:** Multiple modalities including satellite/geographic data
- **Method:** Standardized OOD benchmark
- **Evaluation:** In-distribution vs naturally shifted test domains
- **Main finding:** Shows major ID-to-OOD performance gaps persist across real-world applications even when using robustness methods.
- **Limitation:** Broad benchmark; only a subset is geospatial.
- **Relevance:** Canonical evidence supporting source-disjoint and external evaluation instead of relying on IID testing.
- **Verification:** verified
- **Verification URL:** https://proceedings.mlr.press/v139/koh21a.html

## 51. Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data

- **Citation key:** `manas2021seco`
- **Year:** 2021
- **Authors:** Oscar Mañas; Alexandre Lacoste; Xavier Giró-i-Nieto; David Vazquez; Pau Rodríguez
- **Venue:** IEEE/CVF International Conference on Computer Vision
- **DOI:** 10.1109/ICCV48922.2021.00928
- **Literature block:** Self-supervised remote-sensing pretraining
- **Dataset:** Large unlabeled seasonal remote-sensing imagery
- **Platform/domain:** Satellite / Earth observation
- **Method:** Seasonal contrastive self-supervised learning
- **Evaluation:** Transfer to remote-sensing downstream tasks
- **Main finding:** Uses temporal/seasonal invariance to learn effective in-domain remote-sensing representations without labels.
- **Limitation:** Representation-learning objective differs from source-disjoint robustness evaluation.
- **Relevance:** Supports the argument that ImageNet is not the only useful source of transferable representations for EO.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ICCV48922.2021.00928

## 52. Geography-Aware Self-Supervised Learning

- **Citation key:** `ayush2021geoaware`
- **Year:** 2021
- **Authors:** Kumar Ayush; Burak Uzkent; Chenlin Meng; Kumar Tanmay; Marshall Burke; David Lobell; Stefano Ermon
- **Venue:** IEEE/CVF International Conference on Computer Vision
- **DOI:** 10.1109/ICCV48922.2021.01002
- **Literature block:** Geography-aware pretraining
- **Dataset:** Geo-located remote-sensing datasets
- **Platform/domain:** Satellite / geospatial imagery
- **Method:** Contrastive learning using temporal alignment and geolocation pretext tasks
- **Evaluation:** Classification, detection, and segmentation transfer
- **Main finding:** Shows spatiotemporal and geographic metadata can improve self-supervised remote-sensing representation learning.
- **Limitation:** Pretraining study rather than explicit source-domain generalization benchmark.
- **Relevance:** Supports geographic/source-aware representation learning.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ICCV48922.2021.01002

## 53. SatMAE: Pre-Training Transformers for Temporal and Multi-Spectral Satellite Imagery

- **Citation key:** `cong2022satmae`
- **Year:** 2022
- **Authors:** Yezhen Cong; Samar Khanna; Chenlin Meng; Patrick Liu; Erik Rozi; Yutong He; Marshall Burke; David Lobell; Stefano Ermon
- **Venue:** Advances in Neural Information Processing Systems
- **DOI:** 10.52202/068431-0015
- **Literature block:** Remote-sensing foundation/pretraining
- **Dataset:** Temporal and multispectral satellite imagery
- **Platform/domain:** Satellite imagery
- **Method:** Masked autoencoder with temporal/spectral encoding
- **Evaluation:** Classification and segmentation transfer evaluation
- **Main finding:** Demonstrates strong gains from adapting masked autoencoding to temporal and multispectral satellite data.
- **Limitation:** Does not specifically test our UAV/source-disjoint task.
- **Relevance:** Modern evidence for domain-specific EO pretraining.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.52202/068431-0015

## 54. CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

- **Citation key:** `fuller2023croma`
- **Year:** 2023
- **Authors:** Anthony Fuller; Koreen Millard; James Green
- **Venue:** Advances in Neural Information Processing Systems
- **DOI:** 10.52202/075280-0241
- **Literature block:** Multimodal remote-sensing foundation models
- **Dataset:** Aligned radar + optical remote-sensing data
- **Platform/domain:** SAR + multispectral optical imagery
- **Method:** Contrastive radar-optical masked autoencoder
- **Evaluation:** Classification, probing, clustering, and segmentation
- **Main finding:** Shows complementary contrastive and reconstruction objectives improve uni- and multimodal EO representations.
- **Limitation:** Multimodal sensor setting differs from our RGB UAV scene-classification pipeline.
- **Relevance:** Demonstrates how multimodal pretraining can improve general-purpose remote-sensing representations.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.52202/075280-0241

## 55. RemoteCLIP: A Vision Language Foundation Model for Remote Sensing

- **Citation key:** `liu2024remoteclip`
- **Year:** 2024
- **Authors:** Fan Liu; Delong Chen; Zhangqingyun Guan; Xiaocong Zhou; Jiale Zhu; Qiaolin Ye; Liyong Fu; Jun Zhou
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2024.3390838
- **Literature block:** Remote-sensing foundation models
- **Dataset:** Large unified image-caption remote-sensing corpus
- **Platform/domain:** Satellite + aerial + UAV imagery
- **Method:** Remote-sensing vision-language pretraining
- **Evaluation:** Zero-shot, few-shot, linear probing, retrieval, counting
- **Main finding:** Shows remote-sensing-specific vision-language pretraining substantially improves zero-shot and retrieval performance.
- **Limitation:** Foundation-model benchmark differs from our supervised three-class pipeline.
- **Relevance:** Important modern reference showing stronger pretrained representations can improve cross-domain generalization.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2024.3390838

## 56. SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery

- **Citation key:** `guo2024skysense`
- **Year:** 2024
- **Authors:** Xin Guo; Jiangwei Lao; Bo Dang; Yingying Zhang; Lei Yu; Lixiang Ru; Liheng Zhong; Ziyuan Huang; Kang Wu; Dingxiang Hu; Huimei He; Jian Wang; Jingdong Chen; Ming Yang; Yongjun Zhang; Yansheng Li
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR52733.2024.02613
- **Literature block:** Remote-sensing foundation models
- **Dataset:** 21.5 million multimodal temporal EO sequences
- **Platform/domain:** Optical + SAR + temporal EO imagery
- **Method:** Multi-modal spatiotemporal foundation model
- **Evaluation:** 16 datasets; 7 EO tasks
- **Main finding:** Reports broad generalization across sensors, modalities, temporal settings, and downstream tasks.
- **Limitation:** Massive pretraining/data scale is far beyond our course-project compute setting.
- **Relevance:** Shows the current direction toward representations that explicitly account for domain/modality diversity.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR52733.2024.02613

## 57. SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding

- **Citation key:** `bastani2023satlaspretrain`
- **Year:** 2023
- **Authors:** Favyen Bastani; Piper Wolters; Ritwik Gupta; Joe Ferdinando; Aniruddha Kembhavi
- **Venue:** IEEE/CVF International Conference on Computer Vision
- **DOI:** 10.1109/ICCV51070.2023.01538
- **Literature block:** Large-scale EO pretraining
- **Dataset:** SatlasPretrain; Sentinel-2 + NAIP; 302M labels; 137 categories
- **Platform/domain:** Satellite + high-resolution aerial imagery
- **Method:** Large-scale multi-task pretraining
- **Evaluation:** Multiple downstream remote-sensing tasks
- **Main finding:** Shows remote-sensing-specific pretraining improves average downstream performance substantially over ImageNet.
- **Limitation:** Broad multi-task pretraining rather than a controlled source-disjoint classification study.
- **Relevance:** Supports the idea that in-domain EO pretraining could be a stronger future baseline than ImageNet MobileNetV2.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/ICCV51070.2023.01538

## 58. GEO-Bench: Toward Foundation Models for Earth Monitoring

- **Citation key:** `lacoste2023geobench`
- **Year:** 2023
- **Authors:** Alexandre Lacoste; Nils Lehmann; Pau Rodriguez; Evan David Sherwin; Hannah Kerner; Björn Lütjens; Jeremy Andrew Irvin; David Dao; Hamed Alemohammad; Alexandre Drouin; Mehmet Gunturkun; Gabriel Huang; David Vazquez; Dava Newman; Yoshua Bengio; Stefano Ermon; Xiao Xiang Zhu
- **Venue:** Advances in Neural Information Processing Systems Datasets and Benchmarks Track
- **DOI:** 10.52202/075280-2223
- **Literature block:** Foundation-model benchmarking
- **Dataset:** GEO-Bench; six classification + six segmentation tasks
- **Platform/domain:** Earth observation
- **Method:** Standardized benchmark for pretrained EO representations
- **Evaluation:** Aggregated evaluation across 12 EO downstream tasks
- **Main finding:** Provides a standardized methodology for evaluating foundation-model representations across diverse EO tasks.
- **Limitation:** Downstream benchmark aggregation differs from our source-disjoint climate-scene protocol.
- **Relevance:** Supports rigorous, multi-dataset evaluation rather than single-benchmark headline accuracy.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.52202/075280-2223

## 59. AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities

- **Citation key:** `astruc2025anysat`
- **Year:** 2025
- **Authors:** Guillaume Astruc; Nicolas Gonthier; Clément Mallet; Loic Landrieu
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR52734.2025.01819
- **Literature block:** Remote-sensing foundation models
- **Dataset:** GeoPlex; five multimodal datasets; 11 sensors
- **Platform/domain:** Aerial + satellite + multiresolution + multimodal EO
- **Method:** Scale-adaptive JEPA-based multimodal foundation model
- **Evaluation:** GeoPlex + six external EO datasets
- **Main finding:** Shows one model can generalize across highly heterogeneous resolutions, scales, modalities, and external datasets.
- **Limitation:** Much larger pretraining regime and broader task scope than our TensorFlow classification system.
- **Relevance:** Highly relevant modern evidence that domain diversity should be built directly into EO representation learning.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR52734.2025.01819

## 60. RobSense: A Robust Multi-Modal Foundation Model for Remote Sensing with Static, Temporal, and Incomplete Data Adaptability

- **Citation key:** `do2025robsense`
- **Year:** 2025
- **Authors:** Minh Kha Do; Kang Han; Phu Lai; Khoa T. Phan; Wei Xiang
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR52734.2025.00696
- **Literature block:** Robust remote-sensing foundation models
- **Dataset:** Satlas-based global Sentinel-1/Sentinel-2 pretraining
- **Platform/domain:** Multispectral + SAR; static/temporal; incomplete inputs
- **Method:** Multimodal masked autoencoding + temporal alignment + latent reconstruction
- **Evaluation:** Classification, segmentation, change detection; complete and missing-data settings
- **Main finding:** Maintains strong performance across diverse input types and increasingly incomplete sensor observations.
- **Limitation:** Robustness is primarily modality/input-completeness robustness rather than our source-provenance shift.
- **Relevance:** Modern example of explicitly designing EO models for robustness rather than only IID accuracy.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR52734.2025.00696

## 61. Change-Aware Sampling and Contrastive Learning for Satellite Images

- **Citation key:** `mall2023caco`
- **Year:** 2023
- **Authors:** Utkarsh Mall; Bharath Hariharan; Kavita Bala
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR52729.2023.00509
- **Literature block:** Geographic sampling / self-supervision
- **Dataset:** Large spatiotemporal satellite imagery corpus
- **Platform/domain:** Satellite imagery
- **Method:** Change-aware contrastive loss + geographically informed data sampling
- **Evaluation:** Downstream segmentation and change-detection tasks
- **Main finding:** Shows geographic sampling strategy and temporal-change awareness materially affect representation quality.
- **Limitation:** Self-supervised representation study rather than explicit DG testing.
- **Relevance:** Strong support for our finding that source/geographic dataset composition affects learned representations.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR52729.2023.00509
