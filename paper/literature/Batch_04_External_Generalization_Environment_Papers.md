# Literature Matrix — Batch 04 External Generalization, Flood, and Desert Environment Papers

Verified papers in Batch 4: **20**

Batch 4 strengthens:

- real-world geographic/domain-shift evaluation,
- cross-dataset scene classification,
- global and multisensor benchmark design,
- independent-event flood validation,
- UAV disaster classification,
- desert and desertification recognition, and
- cross-temporal environmental domain adaptation.

## 62. Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery

- **Citation key:** `alemadi2025rwds`
- **Year:** 2025
- **Authors:** Sara A. Al-Emadi; Yin Yang; Ferda Ofli
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR52734.2025.00777
- **Literature block:** Real-world remote-sensing distribution shift
- **Dataset:** RWDS; three domain-generalization benchmark datasets
- **Platform/domain:** Satellite imagery; climate zones, disasters, and geographic regions
- **Method:** Benchmarking modern object detectors under real-world spatial distribution shifts
- **Evaluation:** Source-domain training with evaluation on unseen geographic/climate/disaster domains
- **Main finding:** Demonstrates substantial detector degradation under realistic geographic and environmental shifts.
- **Limitation:** Object detection differs from our scene-level three-class classification task.
- **Relevance:** Direct support for evaluating remote-sensing models under real source/geographic shifts rather than IID splits.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR52734.2025.00777

## 63. Remote-Sensing Cross-Domain Scene Classification: A Dataset and Benchmark

- **Citation key:** `liu2022mrssc`
- **Year:** 2022
- **Authors:** Kang Liu; Jian Yang; Shengyang Li
- **Venue:** Remote Sensing
- **DOI:** 10.3390/rs14184635
- **Literature block:** Cross-domain scene classification
- **Dataset:** MRSSC; 26,710 images; 7 classes; 4 domains
- **Platform/domain:** Multimodal remote-sensing imagery
- **Method:** Cross-domain remote-sensing scene-classification benchmark
- **Evaluation:** Scene classification across distinct remote-sensing domains/modalities
- **Main finding:** Shows major performance challenges arise when scene classifiers must transfer across imaging domains.
- **Limitation:** Its domains and class taxonomy differ from our UAV/satellite climate-scene composition.
- **Relevance:** One of the most direct benchmark precedents for our source-aware cross-domain scene evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/rs14184635

## 64. LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation

- **Citation key:** `wang2021loveda`
- **Year:** 2021
- **Authors:** Junjue Wang; Zhuo Zheng; Ailong Ma; Xiaoyan Lu; Yanfei Zhong
- **Venue:** Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks
- **DOI:** No DOI recorded
- **Literature block:** Geographic domain adaptation benchmark
- **Dataset:** LoveDA; 5,987 high-resolution images; urban and rural domains
- **Platform/domain:** High-spatial-resolution remote sensing
- **Method:** Semantic-segmentation and unsupervised domain-adaptation benchmark
- **Evaluation:** Urban-to-rural and rural-to-urban transfer
- **Main finding:** Demonstrates that geographical environment changes and class-distribution differences substantially challenge transferability.
- **Limitation:** Segmentation rather than scene classification.
- **Relevance:** Strong precedent for defining domains through geographic/environmental differences.
- **Verification:** verified
- **Verification URL:** https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4e732ced3463d06de0ca9a15b6153677-Abstract-round2.html

## 65. OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping

- **Citation key:** `xia2023openearthmap`
- **Year:** 2023
- **Authors:** Junshi Xia; Naoto Yokoya; Bruno Adriano; Clifford Broni-Bediako
- **Venue:** IEEE/CVF Winter Conference on Applications of Computer Vision
- **DOI:** 10.1109/WACV56688.2023.00619
- **Literature block:** Global geographic generalization benchmark
- **Dataset:** OpenEarthMap; 5,000 aerial/satellite images; 97 regions; 44 countries; 6 continents
- **Platform/domain:** High-resolution aerial and satellite imagery
- **Method:** Land-cover segmentation plus unsupervised domain-adaptation benchmarking
- **Evaluation:** Global multi-region evaluation
- **Main finding:** Provides a geographically diverse benchmark explicitly designed to support worldwide model generalization.
- **Limitation:** Land-cover segmentation differs from our scene-classification objective.
- **Relevance:** Excellent example of geographic diversity and external-region evaluation.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/WACV56688.2023.00619

## 66. Cross-Dataset Generalization of ConvNeXt-Tiny for Remote Sensing Scene Classification

- **Citation key:** `nafiiyah2026convnextcross`
- **Year:** 2026
- **Authors:** Nur Nafiiyah; Agus Harjoko; Achmad Nizar Hidayanto
- **Venue:** Engineering, Technology & Applied Science Research
- **DOI:** 10.48084/etasr.17607
- **Literature block:** Cross-dataset scene generalization
- **Dataset:** NWPU-RESISC45 and AID
- **Platform/domain:** High-resolution aerial/remote-sensing scenes
- **Method:** ImageNet-pretrained ConvNeXt-Tiny compared with ResNet50, DenseNet121, and MobileNetV2
- **Evaluation:** In-dataset and cross-dataset evaluation without target fine-tuning
- **Main finding:** Shows that strong IID scene-classification performance does not automatically translate into equal cross-dataset robustness.
- **Limitation:** Evaluation uses shared classes between two conventional scene benchmarks rather than disaster imagery.
- **Relevance:** Very direct modern precedent for our argument that cross-dataset evaluation exposes information hidden by random test splits.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.48084/etasr.17607

## 67. Functional Map of the World

- **Citation key:** `christie2018fmow`
- **Year:** 2018
- **Authors:** Gordon Christie; Neil Fendley; James Wilson; Ryan Mukherjee
- **Venue:** IEEE Conference on Computer Vision and Pattern Recognition
- **DOI:** 10.1109/CVPR.2018.00646
- **Literature block:** Global satellite scene dataset
- **Dataset:** fMoW; more than one million images; more than 200 countries; 63 categories
- **Platform/domain:** Temporal satellite imagery plus metadata
- **Method:** Large-scale functional scene recognition benchmark
- **Evaluation:** Temporal/geographic satellite-image classification
- **Main finding:** Demonstrates the scale and geographic diversity needed for real-world satellite scene understanding.
- **Limitation:** Functional land-use labels differ from climate and disaster categories.
- **Relevance:** Important global-scale example showing how geography, time, and metadata affect satellite recognition.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPR.2018.00646

## 68. So2Sat LCZ42: A Benchmark Data Set for the Classification of Global Local Climate Zones

- **Citation key:** `zhu2020so2sat`
- **Year:** 2020
- **Authors:** Xiao Xiang Zhu; Jingliang Hu; Chunping Qiu; Yilei Shi; Jian Kang; Lichao Mou; Hossein Bagheri; Matthias Haberle; Yuansheng Hua; Rong Huang; Lloyd Hughes; Hao Li; Yao Sun; Guichen Zhang; Shiyao Han; Michael Schmitt; Yuanyuan Wang
- **Venue:** IEEE Geoscience and Remote Sensing Magazine
- **DOI:** 10.1109/MGRS.2020.2964708
- **Literature block:** Global climate-zone benchmark
- **Dataset:** So2Sat LCZ42; approximately 500,000 patches; 42 major urban regions plus additional areas
- **Platform/domain:** Sentinel-1 SAR + Sentinel-2 multispectral imagery
- **Method:** Global local-climate-zone classification benchmark
- **Evaluation:** Multi-city global classification
- **Main finding:** Provides globally distributed multisensor data for studying scalable climate-zone classification.
- **Limitation:** Urban climate-zone taxonomy differs from our three climate-scene classes.
- **Relevance:** Supports geographically distributed training/evaluation and climate-related remote-sensing classification.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/MGRS.2020.2964708

## 69. SEN12MS – A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion

- **Citation key:** `schmitt2019sen12ms`
- **Year:** 2019
- **Authors:** Michael Schmitt; Lloyd H. Hughes; Chunping Qiu; Xiao Xiang Zhu
- **Venue:** ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences
- **DOI:** 10.5194/isprs-annals-IV-2-W7-153-2019
- **Literature block:** Multisensor global remote-sensing datasets
- **Dataset:** SEN12MS; 180,662 georeferenced patch triplets
- **Platform/domain:** Sentinel-1 SAR + Sentinel-2 multispectral + MODIS land cover
- **Method:** Curated global multisensor dataset
- **Evaluation:** Data distributed across inhabited continents and meteorological seasons
- **Main finding:** Addresses limitations in spatial coverage, diversity, and sample count of earlier remote-sensing datasets.
- **Limitation:** Dataset contribution rather than a dedicated cross-domain classification experiment.
- **Relevance:** Supports our emphasis on dataset composition, source diversity, and seasonal/geographic coverage.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019

## 70. Dynamic World, Near real-time global 10 m land use land cover mapping

- **Citation key:** `brown2022dynamicworld`
- **Year:** 2022
- **Authors:** Christopher F. Brown; Steven P. Brumby; Brookie Guzder-Williams; Tanya Birch; Samantha Brooks Hyde; Joseph Mazzariello; Wanda Czerwinski; Valerie J. Pasquarella; Robert Haertel; Simon Ilyushchenko; Kurt Schwehr; Mikaela Weisse; Fred Stolle; Craig Hanson; Oliver Guinan; Rebecca Moore; Alexander M. Tait
- **Venue:** Scientific Data
- **DOI:** 10.1038/s41597-022-01307-4
- **Literature block:** Global operational land-cover classification
- **Dataset:** Global Sentinel-2 training and validation imagery
- **Platform/domain:** Sentinel-2
- **Method:** Deep-learning global near-real-time land-cover classification
- **Evaluation:** Global stratified training and unseen validation data
- **Main finding:** Demonstrates operational deep-learning classification at global scale using geographically and ecologically diverse training/validation data.
- **Limitation:** Land-cover taxonomy and global mapping differ from our source-disjoint disaster-scene objective.
- **Relevance:** Strong example of deliberately designing data collection for global ecological/geographic diversity.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1038/s41597-022-01307-4

## 71. Towards global flood mapping onboard low cost satellites with machine learning

- **Citation key:** `mateogarcia2021worldfloods`
- **Year:** 2021
- **Authors:** Gonzalo Mateo-Garcia; Joshua Veitch-Michaelis; Lewis Smith; Silviu Vlad Oprea; Guy Schumann; Yarin Gal; Atılım Güneş Baydin; Dietmar Backes
- **Venue:** Scientific Reports
- **DOI:** 10.1038/s41598-021-86650-z
- **Literature block:** Flood external generalization
- **Dataset:** WorldFloods; 119 global flood events
- **Platform/domain:** Sentinel-2 / satellite flood imagery
- **Method:** CNN flood segmentation with onboard low-power inference
- **Evaluation:** Training on global events and testing on independent flood locations
- **Main finding:** Demonstrates flood models on geographically independent events rather than only random image partitions.
- **Limitation:** Pixel-level segmentation rather than scene-level classification.
- **Relevance:** Excellent precedent for our external FloodNet evaluation philosophy.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1038/s41598-021-86650-z

## 72. Satellite imaging reveals increased proportion of population exposed to floods

- **Citation key:** `tellman2021globalflood`
- **Year:** 2021
- **Authors:** B. Tellman; J. A. Sullivan; C. Kuhn; A. J. Kettner; C. S. Doyle; G. R. Brakenridge; T. A. Erickson; D. A. Slayback
- **Venue:** Nature
- **DOI:** 10.1038/s41586-021-03695-w
- **Literature block:** Global flood observations
- **Dataset:** Global Flood Database; 913 flood events, 2000–2018
- **Platform/domain:** MODIS satellite observations
- **Method:** Large-scale satellite-based flood extent analysis
- **Evaluation:** Global multi-event flood mapping with independent accuracy assessment
- **Main finding:** Provides globally observed flood extent across hundreds of heterogeneous real disaster events.
- **Limitation:** Not a deep-learning scene-classification study.
- **Relevance:** Provides strong evidence for the enormous variability of real flood environments and events.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1038/s41586-021-03695-w

## 73. SpaceNet 8 - The Detection of Flooded Roads and Buildings

- **Citation key:** `hansch2022spacenet8`
- **Year:** 2022
- **Authors:** Ronny Hänsch; Jacob Arndt; Dalton Lunga; Matthew Gibb; Tyler Pedelose; Arnold Boedihardjo; Desiree Petrie; Todd M. Bacastow
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
- **DOI:** 10.1109/CVPRW56347.2022.00153
- **Literature block:** Flood disaster benchmark
- **Dataset:** SpaceNet 8; flooded buildings and roads; approximately 850 km²
- **Platform/domain:** Pre/post-event satellite imagery
- **Method:** Building, road-network, and flood-damage benchmark
- **Evaluation:** Infrastructure extraction plus flood-state assessment
- **Main finding:** Combines flood detection with operationally important building and road understanding.
- **Limitation:** Task is structured detection/segmentation rather than our three-class scene classification.
- **Relevance:** Supports disaster imagery evaluation using real post-event environmental context.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/CVPRW56347.2022.00153

## 74. Advancing efficiency and accuracy in aerial image classification for disaster incidents with SqueezeViT and UAVDisaster36K benchmark

- **Citation key:** `shianios2026uavdisaster36k`
- **Year:** 2026
- **Authors:** Demetris Shianios; Christos Kyrkou
- **Venue:** Neurocomputing
- **DOI:** 10.1016/j.neucom.2026.133282
- **Literature block:** UAV disaster scene classification
- **Dataset:** UAVDisaster36K; approximately 36,000 UAV disaster images
- **Platform/domain:** UAV aerial imagery
- **Method:** SqueezeViT plus ConvNet/ViT benchmark comparisons
- **Evaluation:** Disaster image classification with performance/efficiency benchmarking
- **Main finding:** Provides a substantially larger modern UAV disaster classification benchmark and compares accuracy with deployment efficiency.
- **Limitation:** Does not isolate satellite-to-UAV domain transfer.
- **Relevance:** Directly relevant recent UAV disaster scene-classification benchmark.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.neucom.2026.133282

## 75. Deep Learning Image-Based Classification for Post-Earthquake Damage Level Prediction Using UAVs

- **Citation key:** `alsaaran2025earthquakeuav`
- **Year:** 2025
- **Authors:** Norah Alsaaran; Adel Soudani
- **Venue:** Sensors
- **DOI:** 10.3390/s25175406
- **Literature block:** UAV disaster classification
- **Dataset:** Post-earthquake UAV imagery
- **Platform/domain:** UAV imagery
- **Method:** Lightweight MobileNetV3-Small damage classification
- **Evaluation:** Three-level post-earthquake damage classification
- **Main finding:** Shows lightweight pretrained CNNs can provide efficient real-time UAV disaster classification.
- **Limitation:** Single disaster type and damage taxonomy.
- **Relevance:** Useful comparison for lightweight transfer learning in UAV disaster imagery.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.3390/s25175406

## 76. DBE-SSDNet: A Dual-Branch Desert Segmentation Model With Structured State-Space Duality for Satellite Remote Sensing

- **Citation key:** `wang2026dbessdnet`
- **Year:** 2026
- **Authors:** Zhaobin Wang; Rui Wang; Yongke Lv; Yaonan Zhang; Ye Zhu
- **Venue:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **DOI:** 10.1109/JSTARS.2026.3689109
- **Literature block:** Desert mapping / geographic generalization
- **Dataset:** Landsat 8 imagery from multiple desert systems
- **Platform/domain:** Satellite desert imagery
- **Method:** Dual-branch state-space desert segmentation network
- **Evaluation:** Multi-region desert boundary testing
- **Main finding:** Reports strong performance across distinct desert boundary conditions and geographic regions.
- **Limitation:** Segmentation and boundary extraction differ from scene-level Desert classification.
- **Relevance:** Recent evidence for explicitly testing desert models across different geographical conditions.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1109/JSTARS.2026.3689109

## 77. Global perspectives on sand dune patterns: Scale-adaptable classification using Landsat imagery and deep learning strategies

- **Citation key:** `zheng2024globaldunes`
- **Year:** 2024
- **Authors:** Zhijia Zheng; Xiuyuan Zhang; Jiajun Li; Eslam Ali; Jinsongdi Yu; Shihong Du
- **Venue:** ISPRS Journal of Photogrammetry and Remote Sensing
- **DOI:** 10.1016/j.isprsjprs.2024.10.002
- **Literature block:** Global desert scene understanding
- **Dataset:** Global Landsat sand-dune imagery
- **Platform/domain:** Landsat desert imagery
- **Method:** Deep-learning scale-adaptable dune-pattern classification
- **Evaluation:** Large-scale classification of globally distributed sand-dune patterns
- **Main finding:** Demonstrates deep-learning classification across geographically diverse global desert environments.
- **Limitation:** Dune morphology classes differ from our broad Desert scene label.
- **Relevance:** Provides direct literature support for the heterogeneity within the apparently simple Desert category.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.isprsjprs.2024.10.002

## 78. An end-to-end pipeline based on a Two-Dimensional Convolutional Neural Network for monitoring desertification in Africa using remote sensing images

- **Citation key:** `chouikhi2025africadesert`
- **Year:** 2025
- **Authors:** Farah Chouikhi; Ali Ben Abbes; Imed Riadh Farah
- **Venue:** Environmental Modelling & Software
- **DOI:** 10.1016/j.envsoft.2025.106622
- **Literature block:** Desertification classification
- **Dataset:** MODIS imagery across Africa, 2015–2023
- **Platform/domain:** Continental-scale satellite imagery
- **Method:** 2D-CNN desertification monitoring pipeline
- **Evaluation:** Multi-year African desertification classification
- **Main finding:** Demonstrates CNN-based desertification monitoring over large spatial and temporal extents.
- **Limitation:** MODIS resolution and desertification labels differ from our RGB scene-level dataset.
- **Relevance:** Supports the Desert component of our climate-scene classification literature.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.envsoft.2025.106622

## 79. Detecting the signs of desertification with Landsat imagery: A semi-supervised anomaly detection approach

- **Citation key:** `harrou2024desertification`
- **Year:** 2024
- **Authors:** Fouzi Harrou; Benamar Bouyeddou; Nabil Zerrouki; Abdelkader Dairi; Ying Sun; Yacine Zerrouki
- **Venue:** Results in Engineering
- **DOI:** 10.1016/j.rineng.2024.102037
- **Literature block:** Desertification detection
- **Dataset:** Multitemporal Landsat imagery from Biskra, Algeria
- **Platform/domain:** Landsat / arid environment
- **Method:** Semi-supervised anomaly detection with OCSVM and alternative detectors
- **Evaluation:** Desertification detection plus bootstrap confidence intervals
- **Main finding:** Shows desertification can be detected from multispectral temporal signals without requiring large labeled desertification datasets.
- **Limitation:** Regional anomaly-detection task rather than cross-domain scene classification.
- **Relevance:** Broadens our Desert literature beyond conventional supervised CNN classification.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.rineng.2024.102037

## 80. Remotely sensed desertification modeling using ensemble of machine learning algorithms

- **Citation key:** `boali2024desertensemble`
- **Year:** 2024
- **Authors:** Abdolhossein Boali; Hamid Reza Asgari; Ali Mohammadian Behbahani; Abdolrassoul Salmanmahiny; Babak Naimi
- **Venue:** Remote Sensing Applications: Society and Environment
- **DOI:** 10.1016/j.rsase.2024.101149
- **Literature block:** Desertification / model uncertainty
- **Dataset:** Remote-sensing desertification indicators
- **Platform/domain:** Arid-region remote sensing
- **Method:** Ensemble machine-learning desertification modeling
- **Evaluation:** Comparison and ensemble of multiple desertification models
- **Main finding:** Uses model ensembles to reduce uncertainty in remote-sensing desertification assessment.
- **Limitation:** Environmental modeling rather than RGB scene recognition.
- **Relevance:** Useful contrast with single-model classification and supports discussion of uncertainty.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1016/j.rsase.2024.101149

## 81. Cross-temporal classification of sand dune/sheet in the Belt and Road Initiative drylands based on deep learning domain adaptation

- **Citation key:** `zhan2026sanddomain`
- **Year:** 2026
- **Authors:** Hao Zhan; Yingqi Xue; Xiaoqiang Yuan; Jinsongdi Yu; Zhijia Zheng
- **Venue:** International Journal of Digital Earth
- **DOI:** 10.1080/17538947.2026.2654903
- **Literature block:** Desert domain adaptation
- **Dataset:** Dryland imagery across 22 Belt and Road Initiative countries and multiple years
- **Platform/domain:** Large-scale multitemporal dryland remote sensing
- **Method:** Fourier-based spectral adaptation plus TransUNet self-training
- **Evaluation:** Cross-temporal classification under spectral domain shift
- **Main finding:** Directly demonstrates that spectral/domain shifts across time impair desert mapping and motivates domain-adaptation strategies.
- **Limitation:** Temporal domain adaptation differs from our source/provenance-domain shift.
- **Relevance:** Highly relevant evidence that even within dryland/desert imagery, distribution shift materially changes classification behavior.
- **Verification:** verified
- **Verification URL:** https://doi.org/10.1080/17538947.2026.2654903
