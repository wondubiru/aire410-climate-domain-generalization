# Literature Matrix — Batch 02 Balanced Papers

Verified papers in Batch 2: **20**

Batch 2 strengthens:

- flood and disaster benchmark coverage,
- mainstream aerial/satellite scene-classification datasets,
- transfer learning and in-domain pretraining,
- geographically diverse external flood datasets, and
- recent remote-sensing benchmark literature.

**Note on xBD:** the canonical CVPRW paper is retained without a DOI rather than substituting the separate CMU/SEI report DOI.

## 22. Survey on Remote Sensing Scene Classification: From Traditional Methods to Large Generative AI Models

- **Citation key:** `huang2026scenesurvey`
- **Year:** 2026
- **Authors:** Qionghao Huang; Can Hu
- **Venue:** Journal of King Saud University - Computer and Information Sciences
- **DOI:** 10.1007/s44443-026-00694-7
- **Block:** Remote-sensing scene surveys
- **Dataset:** Review
- **Platform/domain:** Remote-sensing scene classification
- **Method:** Survey covering handcrafted features, CNNs, ViTs, foundation models, VLMs, and generative AI
- **Evaluation:** Literature review
- **Main finding:** Identifies cross-domain generalization, interpretability, multimodal learning, and standardized evaluation as important current research directions.
- **Limitation:** Broad survey rather than task-specific empirical evidence.
- **Relevance:** Current high-level reference positioning our work within the newest remote-sensing scene-classification landscape.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1007/s44443-026-00694-7

## 23. AIFloodSense: A Global Aerial Imagery Dataset for Semantic Segmentation and Understanding of Flooded Environments

- **Citation key:** `simantiris2026aifloodsense`
- **Year:** 2026
- **Authors:** Georgios Simantiris; Konstantinos Bacharidis; Apostolos Papanikolaou; Petros Giannakakis; Costas Panagiotakis
- **Venue:** Remote Sensing
- **DOI:** 10.3390/rs18060938
- **Block:** Flood generalization benchmarks
- **Dataset:** AIFloodSense; 470 aerial images; 230 flood events; 64 countries
- **Platform/domain:** Globally diverse aerial flood imagery
- **Method:** Classification, segmentation, and VQA benchmark
- **Evaluation:** Diversity-aware flood benchmarking and external testing
- **Main finding:** Shows geographically/event-diverse flood data can generalize better externally than larger but less diverse datasets.
- **Limitation:** Compact dataset and different annotation/tasks from our three-class benchmark.
- **Relevance:** Extremely relevant recent evidence supporting our argument that diversity and domain composition matter more than headline within-dataset accuracy.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/rs18060938

## 24. MEET: A Million-Scale Dataset for Fine-Grained Geospatial Scene Classification With Zoom-Free Remote Sensing Imagery

- **Citation key:** `li2025meet`
- **Year:** 2025
- **Authors:** Yansheng Li; Yuning Wu; Gong Cheng; Chao Tao; Bo Dang; Yu Wang; Jiahao Zhang; Chuge Zhang; Yiting Liu; Xu Tang; Jiayi Ma; Yongjun Zhang
- **Venue:** IEEE/CAA Journal of Automatica Sinica
- **DOI:** 10.1109/JAS.2025.125324
- **Block:** Modern scene benchmarks
- **Dataset:** MEET; >1.03 million samples; 80 categories
- **Platform/domain:** Fine-grained remote-sensing scenes
- **Method:** Context-aware transformer + million-scale benchmark
- **Evaluation:** Fine-grained geospatial scene classification
- **Main finding:** Demonstrates the importance of surrounding spatial context for fixed-resolution geospatial recognition.
- **Limitation:** Focuses fine-grained contextual recognition rather than cross-source robustness.
- **Relevance:** Recent large-scale evidence that scene context can strongly influence remote-sensing classification.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JAS.2025.125324

## 25. STURM-Flood: A Curated Dataset for Deep Learning-Based Flood Extent Mapping Leveraging Sentinel-1 and Sentinel-2 Imagery

- **Citation key:** `notarangelo2025sturm`
- **Year:** 2025
- **Authors:** Nicla Notarangelo; Charlotte Wirion; Frankwin van Winsen
- **Venue:** Big Earth Data
- **DOI:** 10.1080/20964471.2025.2458714
- **Block:** Flood mapping datasets
- **Dataset:** STURM-Flood
- **Platform/domain:** Sentinel-1 SAR + Sentinel-2 optical imagery
- **Method:** Curated multi-sensor DL-ready flood dataset
- **Evaluation:** Flood extent mapping with Copernicus EMS references
- **Main finding:** Provides curated multi-sensor flood data aimed at robust and reproducible flood mapping.
- **Limitation:** Pixel-level extent mapping rather than scene-level climate classification.
- **Relevance:** Recent evidence for multi-source/multi-sensor dataset design and external flood evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1080/20964471.2025.2458714

## 26. In-Domain Self-Supervised Learning Improves Remote Sensing Image Scene Classification

- **Citation key:** `dimitrovski2024indomainssl`
- **Year:** 2024
- **Authors:** Ivica Dimitrovski; Ivan Kitanovski; Nikola Simidjievski; Dragi Kocev
- **Venue:** IEEE Geoscience and Remote Sensing Letters
- **DOI:** 10.1109/LGRS.2024.3352926
- **Block:** Pretraining / representation learning
- **Dataset:** Million-AID pretraining + 14 downstream RS datasets
- **Platform/domain:** Remote-sensing imagery
- **Method:** iBOT self-supervised in-domain pretraining with ViTs
- **Evaluation:** Comparison of ImageNet vs in-domain SSL pretraining
- **Main finding:** Shows in-domain remote-sensing pretraining consistently improves downstream performance compared with generic pretraining strategies.
- **Limitation:** Focuses representation transfer rather than source-disjoint DG.
- **Relevance:** Provides modern context for why source-domain-aware pretraining may outperform generic ImageNet transfer.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/LGRS.2024.3352926

## 27. DAM-Net: Flood Detection from SAR Imagery Using Differential Attention Metric-Based Vision Transformers

- **Citation key:** `saleh2024damnet`
- **Year:** 2024
- **Authors:** Tamer Saleh; Xingxing Weng; Shimaa Holail; Chen Hao; Gui-Song Xia
- **Venue:** ISPRS Journal of Photogrammetry and Remote Sensing
- **DOI:** 10.1016/j.isprsjprs.2024.05.018
- **Block:** Flood mapping benchmark
- **Dataset:** S1GFloods; 5,360 pre/post image pairs; 46 events; 6 continents
- **Platform/domain:** Sentinel-1 SAR
- **Method:** Differential-attention vision transformer for bi-temporal flood detection
- **Evaluation:** Cross-event SAR flood-change detection
- **Main finding:** Introduces S1GFloods and demonstrates strong flood change-detection results across globally distributed events.
- **Limitation:** Bi-temporal SAR segmentation differs from single-image RGB scene classification.
- **Relevance:** Recent globally diverse flood benchmark illustrating why cross-event diversity is essential.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.isprsjprs.2024.05.018

## 28. RescueNet: A High Resolution UAV Semantic Segmentation Dataset for Natural Disaster Damage Assessment

- **Citation key:** `rahnemoonfar2023rescuenet`
- **Year:** 2023
- **Authors:** Maryam Rahnemoonfar; Tashnim Chowdhury; Robin Murphy
- **Venue:** Scientific Data
- **DOI:** 10.1038/s41597-023-02799-4
- **Block:** UAV disaster datasets
- **Dataset:** RescueNet
- **Platform/domain:** High-resolution post-disaster UAV imagery
- **Method:** Dataset with scene classification and dense semantic annotations
- **Evaluation:** Post-disaster scene understanding and segmentation
- **Main finding:** Provides detailed UAV annotations for buildings, roads, pools, trees, and other post-disaster scene elements.
- **Limitation:** Collected primarily after Hurricane Michael and focuses damage assessment rather than source-domain shift.
- **Relevance:** Recent UAV disaster benchmark useful for discussing external scene understanding and annotation richness.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1038/s41597-023-02799-4

## 29. On Creating Benchmark Dataset for Aerial Image Interpretation: Reviews, Guidances, and Million-AID

- **Citation key:** `long2021millionaid`
- **Year:** 2021
- **Authors:** Yang Long; Gui-Song Xia; Shengyang Li; Wen Yang; Michael Ying Yang; Xiao Xiang Zhu; Liangpei Zhang; Deren Li
- **Venue:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **DOI:** 10.1109/JSTARS.2021.3070368
- **Block:** Aerial scene benchmarks
- **Dataset:** Million-AID; approximately one million images
- **Platform/domain:** Large-scale aerial imagery
- **Method:** Dataset-construction guidance and large-scale benchmark
- **Evaluation:** Aerial scene-classification benchmarking
- **Main finding:** Demonstrates the importance of scale, taxonomy design, annotation quality, and diversity in aerial benchmarks.
- **Limitation:** Scale alone does not guarantee domain robustness.
- **Relevance:** Strong dataset-design reference and supports our provenance/dataset-composition emphasis.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JSTARS.2021.3070368

## 30. Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1

- **Citation key:** `bonafilia2020sen1floods11`
- **Year:** 2020
- **Authors:** Derrick Bonafilia; Beth Tellman; Tyler Anderson; Erica Issenberg
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
- **DOI:** 10.1109/CVPRW50498.2020.00113
- **Block:** Flood mapping benchmark
- **Dataset:** Sen1Floods11; 4,831 Sentinel-1 chips; 11 flood events
- **Platform/domain:** Sentinel-1 SAR; global flood mapping
- **Method:** Dataset + fully convolutional flood-segmentation baselines
- **Evaluation:** Georeferenced flood/permanent-water segmentation
- **Main finding:** Introduces a geographically diverse Sentinel-1 flood benchmark spanning multiple biomes, ecoregions, continents, and flood events.
- **Limitation:** SAR pixel-level flood mapping differs from our RGB three-class scene classification task.
- **Relevance:** Important global flood benchmark demonstrating the need for geographically diverse external flood evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPRW50498.2020.00113

## 31. RSI-CB: A Large-Scale Remote Sensing Image Classification Benchmark Using Crowdsourced Data

- **Citation key:** `li2020rsicb`
- **Year:** 2020
- **Authors:** Haifeng Li; Xin Dou; Chao Tao; Zhixiang Wu; Jie Chen; Jian Peng; Min Deng; Ling Zhao
- **Venue:** Sensors
- **DOI:** 10.3390/s20061594
- **Block:** Remote-sensing scene benchmarks
- **Dataset:** RSI-CB
- **Platform/domain:** Worldwide remote-sensing imagery + crowdsourced labels
- **Method:** Large-scale crowdsourced benchmark; classical and DCNN baselines
- **Evaluation:** Scene classification and transferability experiments
- **Main finding:** Shows that geographically broad crowdsourced benchmarks can improve diversity and support transfer studies.
- **Limitation:** Some labeling depends on POI/crowdsourced supervision and manual filtering.
- **Relevance:** Supports our emphasis on geographic/source diversity and benchmark composition.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/s20061594

## 32. Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery

- **Citation key:** `gupta2019xbd`
- **Year:** 2019
- **Authors:** Ritwik Gupta; Bryce Goodman; Nirav Patel; Ricky Hosfelt; Sandra Sajeev; Eric Heim; Jigar Doshi; Keane Lucas; Howie Choset; Matthew Gaston
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
- **DOI:** No DOI recorded
- **Block:** Disaster satellite datasets
- **Dataset:** xBD
- **Platform/domain:** Pre/post-disaster satellite imagery
- **Method:** Large-scale building localization and damage classification benchmark
- **Evaluation:** Cross-disaster building-damage assessment
- **Main finding:** Introduces a large disaster dataset spanning multiple disaster types, countries, and environmental conditions.
- **Limitation:** Building-damage/change-detection objective differs from our scene-level climate taxonomy.
- **Relevance:** Supports the value of broad event/source diversity for robust disaster-imagery models.
- **Verification:** verified
- **Verification URL:** https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html

## 33. EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification

- **Citation key:** `helber2019eurosat`
- **Year:** 2019
- **Authors:** Patrick Helber; Benjamin Bischke; Andreas Dengel; Damian Borth
- **Venue:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **DOI:** 10.1109/JSTARS.2019.2918242
- **Block:** Satellite scene benchmarks
- **Dataset:** EuroSAT; 27,000 Sentinel-2 images; 10 classes
- **Platform/domain:** Sentinel-2 multispectral imagery
- **Method:** Deep CNN land-use/land-cover benchmark
- **Evaluation:** Within-dataset LULC classification
- **Main finding:** Provides a publicly available Sentinel-2 benchmark and demonstrates high CNN classification performance.
- **Limitation:** High within-dataset accuracy does not directly establish cross-source robustness.
- **Relevance:** Illustrates the strong benchmark performance that can be achieved on homogeneous satellite datasets.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JSTARS.2019.2918242

## 34. A Survey of Remote Sensing Image Classification Based on CNNs

- **Citation key:** `song2019cnnsurvey`
- **Year:** 2019
- **Authors:** Jia Song; Shaohua Gao; Yunqiang Zhu; Chenyan Ma
- **Venue:** Big Earth Data
- **DOI:** 10.1080/20964471.2019.1657720
- **Block:** Remote-sensing classification surveys
- **Dataset:** Review
- **Platform/domain:** Remote-sensing imagery
- **Method:** CNN-based classification survey
- **Evaluation:** Literature review
- **Main finding:** Reviews the transition toward CNN-based representation learning for remote-sensing classification.
- **Limitation:** Predates modern transformers, foundation models, and current DG methods.
- **Relevance:** Useful historical review for establishing the CNN classification evolution preceding our study.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1080/20964471.2019.1657720

## 35. BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding

- **Citation key:** `sumbul2019bigearthnet`
- **Year:** 2019
- **Authors:** Gencer Sumbul; Marcela Charfuelan; Begum Demir; Volker Markl
- **Venue:** IEEE International Geoscience and Remote Sensing Symposium
- **DOI:** 10.1109/IGARSS.2019.8900532
- **Block:** Satellite scene benchmarks
- **Dataset:** BigEarthNet; 590,326 Sentinel-2 image patches
- **Platform/domain:** Sentinel-2 multispectral imagery
- **Method:** Large-scale multi-label benchmark
- **Evaluation:** Multi-label land-cover interpretation
- **Main finding:** Introduces one of the first very large Sentinel-2 archives for deep remote-sensing representation learning.
- **Limitation:** Multi-label LULC differs from our RGB three-class climate-scene task.
- **Relevance:** Important large-scale satellite benchmark and pretraining/data-diversity reference.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/IGARSS.2019.8900532

## 36. DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images

- **Citation key:** `demir2018deepglobe`
- **Year:** 2018
- **Authors:** Ilke Demir; Krzysztof Koperski; David Lindenbaum; Guan Pang; Jing Huang; Saikat Basu; Forest Hughes; Devis Tuia; Ramesh Raskar
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
- **DOI:** 10.1109/CVPRW.2018.00031
- **Block:** Satellite benchmark datasets
- **Dataset:** DeepGlobe
- **Platform/domain:** Satellite imagery
- **Method:** Benchmark challenge for classification, detection, and segmentation
- **Evaluation:** Standardized Earth-observation benchmark competitions
- **Main finding:** Establishes large-scale satellite understanding tasks and standardized evaluation protocols.
- **Limitation:** Benchmark tasks differ from our specific climate three-class problem.
- **Relevance:** Important precedent for benchmark-driven satellite image evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPRW.2018.00031

## 37. PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval

- **Citation key:** `zhou2018patternnet`
- **Year:** 2018
- **Authors:** Weixun Zhou; Shawn Newsam; Congmin Li; Zhenfeng Shao
- **Venue:** ISPRS Journal of Photogrammetry and Remote Sensing
- **DOI:** 10.1016/j.isprsjprs.2018.01.004
- **Block:** Remote-sensing benchmarks
- **Dataset:** PatternNet; 38 classes; 800 images per class
- **Platform/domain:** High-resolution remote-sensing imagery
- **Method:** Large-scale image retrieval benchmark and deep-feature evaluation
- **Evaluation:** Remote-sensing image retrieval benchmark
- **Main finding:** Addresses scale and saturation limitations of earlier remote-sensing benchmark collections.
- **Limitation:** Designed primarily for retrieval rather than climate-scene classification.
- **Relevance:** Useful evidence that dataset scale/diversity strongly influence learned remote-sensing representations.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.isprsjprs.2018.01.004

## 38. Remote Sensing Image Scene Classification: Benchmark and State of the Art

- **Citation key:** `cheng2017resisc45`
- **Year:** 2017
- **Authors:** Gong Cheng; Junwei Han; Xiaoqiang Lu
- **Venue:** Proceedings of the IEEE
- **DOI:** 10.1109/JPROC.2017.2675998
- **Block:** Remote-sensing scene benchmarks
- **Dataset:** NWPU-RESISC45; 31,500 images; 45 classes
- **Platform/domain:** High-resolution remote-sensing imagery
- **Method:** Benchmark dataset plus broad scene-classification review
- **Evaluation:** Comparison of representative scene-classification methods
- **Main finding:** Introduces NWPU-RESISC45 with substantial within-class variation and between-class similarity.
- **Limitation:** Benchmark partitions primarily assess within-dataset classification rather than explicit source shift.
- **Relevance:** Foundational benchmark and taxonomy for remote-sensing scene classification.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JPROC.2017.2675998

## 39. Towards Better Exploiting Convolutional Neural Networks for Remote Sensing Scene Classification

- **Citation key:** `nogueira2017convnets`
- **Year:** 2017
- **Authors:** Keiller Nogueira; Otavio A. B. Penatti; Jefersson A. dos Santos
- **Venue:** Pattern Recognition
- **DOI:** 10.1016/j.patcog.2016.07.001
- **Block:** Transfer learning / scene classification
- **Dataset:** Three remote-sensing scene datasets
- **Platform/domain:** Aerial and remote-sensing scenes
- **Method:** CNN training, fine-tuning, and fixed feature extraction
- **Evaluation:** Comparison of CNN exploitation strategies
- **Main finding:** Finds fine-tuning pretrained CNNs generally provides strong remote-sensing scene representations.
- **Limitation:** Predates modern DG protocols and large-scale contemporary remote-sensing benchmarks.
- **Relevance:** Foundational evidence motivating transfer learning rather than training every architecture from scratch.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.patcog.2016.07.001

## 40. AID: A Benchmark Data Set for Performance Evaluation of Aerial Scene Classification

- **Citation key:** `xia2017aid`
- **Year:** 2017
- **Authors:** Gui-Song Xia; Jingwen Hu; Fan Hu; Baoguang Shi; Xiang Bai; Yanfei Zhong; Liangpei Zhang; Xiaoqiang Lu
- **Venue:** IEEE Transactions on Geoscience and Remote Sensing
- **DOI:** 10.1109/TGRS.2017.2685945
- **Block:** Remote-sensing scene benchmarks
- **Dataset:** AID
- **Platform/domain:** High-resolution aerial imagery
- **Method:** Large aerial-scene benchmark with classical and deep-learning baselines
- **Evaluation:** Aerial scene classification benchmark
- **Main finding:** Provides a large and diverse aerial benchmark designed to overcome saturation of earlier small scene datasets.
- **Limitation:** Conventional random benchmark evaluation does not directly measure unseen-source robustness.
- **Relevance:** Canonical aerial scene-classification dataset and methodological precedent for our task.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/TGRS.2017.2685945

## 41. Transferring Deep Convolutional Neural Networks for the Scene Classification of High-Resolution Remote Sensing Imagery

- **Citation key:** `hu2015transfercnn`
- **Year:** 2015
- **Authors:** Fan Hu; Gui-Song Xia; Jingwen Hu; Liangpei Zhang
- **Venue:** Remote Sensing
- **DOI:** 10.3390/rs71114680
- **Block:** Transfer learning / scene classification
- **Dataset:** High-resolution remote-sensing scene datasets
- **Platform/domain:** High-resolution remote sensing
- **Method:** Transfer of pretrained deep CNN representations
- **Evaluation:** Transfer-learning scene classification
- **Main finding:** Shows pretrained CNN representations can transfer effectively to high-resolution remote-sensing scenes.
- **Limitation:** Does not test modern source-disjoint external-domain evaluation.
- **Relevance:** Early foundational remote-sensing transfer-learning paper.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/rs71114680
