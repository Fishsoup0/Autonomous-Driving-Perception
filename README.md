<div align="center">

[![image](assets/image.jpg)](https://arxiv.org/abs/2408.16530)

# A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions

[Yu Wang](https://scholar.google.com/citations?user=EsZuitIAAAAJ&hl=zh-TW) · Shaohua Wang · Yicheng Li · Mingchun Liu

[![arXiv](https://img.shields.io/badge/arXiv-2408.16530-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2408.16530)
[![Awesome](https://awesome.re/badge.svg)](https://github.com/Fishsoup0/Autonomous-Driving-Perception)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Fishsoup0/Autonomous-Driving-Perception?style=flat&color=orange)](https://github.com/Fishsoup0/Autonomous-Driving-Perception/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/Fishsoup0/Autonomous-Driving-Perception)](https://github.com/Fishsoup0/Autonomous-Driving-Perception/commits/main)

*A curated collection of papers, datasets, and toolboxes on 3D perception for autonomous driving — covering camera / LiDAR / fusion detection, occupancy prediction, end-to-end driving, world models, V2X cooperative perception, and robustness / test-time adaptation.*

**⭐ If you find this repository helpful, please consider giving it a star and [citing our survey](#citation)!**

</div>

---

## 🔥 News

- **[2026-07]** 🚀 Major refresh: restructured the repository, expanded all paper lists with CVPR / ICCV / NeurIPS / ICLR 2024–2025 works, and added new sections on **World Models**, **VLM / LLM for Driving**, and **Robustness & Test-Time Adaptation**.
- **[2026-07]** Added `CONTRIBUTING.md` — community PRs are welcome!
- **[2024-08]** 📄 Our survey is released on [arXiv](https://arxiv.org/abs/2408.16530).

---

## 📖 Table of Contents

- [About the Survey](#about-the-survey)
- [Datasets](#datasets)
- [Simulators & Toolboxes](#simulators--toolboxes)
- [Papers](#papers)
  - [Camera-Based 3D Object Detection](#camera-based-3d-object-detection)
  - [LiDAR-Based 3D Object Detection](#lidar-based-3d-object-detection)
  - [Fusion-Based 3D Object Detection](#fusion-based-3d-object-detection)
  - [3D Occupancy Prediction](#3d-occupancy-prediction)
  - [Temporal & Streaming Perception](#temporal--streaming-perception)
  - [End-to-End Autonomous Driving](#end-to-end-autonomous-driving)
  - [World Models for Autonomous Driving](#world-models-for-autonomous-driving)
  - [VLM / LLM for Autonomous Driving](#vlm--llm-for-autonomous-driving)
  - [V2X Cooperative Perception](#v2x-cooperative-perception)
  - [Robustness, Domain Adaptation & Test-Time Adaptation](#robustness-domain-adaptation--test-time-adaptation)
- [Related Surveys](#related-surveys)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## About the Survey

This repository accompanies the review paper **"A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions"**, which provides an extensive overview of 3D object perception for autonomous driving systems, covering camera-based, LiDAR-based, and fusion detection techniques, and discussing future directions such as temporal perception, occupancy grids, end-to-end learning frameworks, and cooperative perception.

<details>
<summary><b>Abstract (click to expand)</b></summary>

In recent years, 3D object perception has become a crucial component in the development of autonomous driving systems, providing essential environmental awareness. However, as perception tasks in autonomous driving evolve, their variants have increased, leading to diverse insights from industry and academia. Currently, there is a lack of comprehensive surveys that collect and summarize these perception tasks and their developments from a broader perspective. This review extensively summarizes traditional 3D object detection methods, focusing on camera-based, LiDAR-based, and fusion detection techniques. We provide a comprehensive analysis of the strengths and limitations of each approach, highlighting advancements in accuracy and robustness. Furthermore, we discuss future directions, including methods to improve accuracy such as temporal perception, occupancy grids, and end-to-end learning frameworks. We also explore cooperative perception methods that extend the perception range through collaborative communication. By providing a holistic view of the current state and future developments in 3D object perception, we aim to offer a more comprehensive understanding of perception tasks for autonomous driving.

</details>

**Key contributions:**

1. A holistic view of the evolution and future trends in 3D object perception for autonomous driving.
2. A comprehensive summary, classification, and analysis of the latest camera-based, LiDAR-based, and fusion-based 3D object detection methods.
3. A panoramic compilation of perception methods, datasets, and evaluation metrics to promote research insights.

---

## Datasets

### General Autonomous Driving Perception

| Dataset | Type | Sensors | Highlights | Link |
|---|---|---|---|---|
| **KITTI** | Real | LiDAR + Camera | Classic benchmark for 2D/3D detection, stereo, depth | [Website](http://www.cvlibs.net/datasets/kitti/) |
| **nuScenes** | Real | LiDAR + Camera + Radar | 360° perception, tracking, sensor fusion | [Website](https://www.nuscenes.org/) |
| **Waymo Open Dataset** | Real | LiDAR + Camera | Large-scale 3D detection, tracking, segmentation | [Website](https://waymo.com/open/) |
| **Argoverse 2** | Real | LiDAR + Camera | Long-range detection, HD maps, forecasting | [Website](https://www.argoverse.org/av2.html) |
| **KITTI-360** | Real | LiDAR + Camera | 360° scene reconstruction, mapping, localization | [Website](http://www.cvlibs.net/datasets/kitti-360/) |

### 3D Occupancy & Semantic Scene Understanding

| Dataset | Type | Sensors | Highlights | Link |
|---|---|---|---|---|
| **Occ3D** (nuScenes / Waymo) | Real | LiDAR + Camera | Standard benchmark for 3D occupancy prediction | [GitHub](https://github.com/Tsinghua-MARS-Lab/Occ3D) |
| **SemanticKITTI** | Real | LiDAR | Semantic segmentation, scene completion, SLAM | [Website](http://www.semantic-kitti.org/) |
| **OpenScene** | Real | Camera | Large-scale occupancy benchmark built on nuPlan | [GitHub](https://github.com/OpenDriveLab/OpenScene) |

### V2X / Cooperative Perception

| Dataset | Type | Sensors | Highlights | Link |
|---|---|---|---|---|
| **OPV2V** | Sim | LiDAR + Camera | First large-scale V2V benchmark (CARLA) | [Website](https://mobility-lab.seas.ucla.edu/opv2v/) |
| **V2XSet** | Sim | LiDAR + Camera | V2X with realistic noise simulation | [GitHub](https://github.com/DerrickXuNu/v2x-vit) |
| **V2X-Sim** | Sim | LiDAR + Camera | Multi-agent collaborative perception | [Website](https://ai4ce.github.io/V2X-Sim/) |
| **DAIR-V2X** | Real | LiDAR + Camera | First real-world vehicle-infrastructure dataset | [GitHub](https://github.com/AIR-THU/DAIR-V2X) |
| **V2V4Real** | Real | LiDAR + Camera | Real-world V2V cooperative perception (CVPR 2023) | [GitHub](https://github.com/ucla-mobility/V2V4Real) |
| **TUMTraf V2X** | Real | LiDAR + Camera | V2I cooperative perception (CVPR 2024) | [Website](https://tum-traffic-dataset.github.io/tumtraf-v2x/) |
| **RCooper** | Real | LiDAR + Camera | Roadside cooperative perception (CVPR 2024) | [GitHub](https://github.com/AIR-THU/DAIR-RCooper) |

---

## Simulators & Toolboxes

| Name | Description | Link |
|---|---|---|
| **CARLA** | Open-source simulator with rich sensor suites; the de-facto standard for closed-loop AD research | [Website](https://carla.org/) |
| **SVL Simulator** (LGSVL) | High-fidelity simulation, integrates with Autoware / Apollo (archived but still widely referenced) | [GitHub](https://github.com/lgsvl/simulator) |
| **AirSim** | Microsoft's cross-platform simulator on Unreal Engine (archived) | [GitHub](https://github.com/microsoft/AirSim) |
| **OpenPCDet** | Widely-used toolbox for LiDAR-based 3D detection | [GitHub](https://github.com/open-mmlab/OpenPCDet) |
| **MMDetection3D** | OpenMMLab's general 3D detection platform | [GitHub](https://github.com/open-mmlab/mmdetection3d) |
| **OpenCOOD** | Standard framework for V2X cooperative detection | [GitHub](https://github.com/DerrickXuNu/OpenCOOD) |

---

## Papers

> Format: **Method** — Venue Year | [Paper] | [Code]. Within each section, papers are ordered chronologically. 🔥 marks recent (2024+) works.

### Camera-Based 3D Object Detection

**Monocular**

- **Mono3D** — CVPR 2016 | [Paper](https://arxiv.org/abs/1608.07711) — pioneering monocular 3D proposals
- **M3D-RPN** — ICCV 2019 | [Paper](https://arxiv.org/abs/1907.06038) — monocular 3D region proposal network
- **FCOS3D** — ICCVW 2021 | [Paper](https://arxiv.org/abs/2104.10956) | [Code](https://github.com/open-mmlab/mmdetection3d) — fully convolutional single-stage monocular detection
- **MonoDTR** — CVPR 2022 | [Paper](https://arxiv.org/abs/2203.10981) — depth-aware transformer
- **MonoDETR** — ICCV 2023 | [Paper](https://arxiv.org/abs/2203.13310) | [Code](https://github.com/ZrrSkywalker/MonoDETR) — depth-guided DETR for monocular 3D detection

**Multi-View / BEV**

- **DETR3D** — CoRL 2021 | [Paper](https://arxiv.org/abs/2110.06922) | [Code](https://github.com/WangYueFt/detr3d) — sparse 3D queries from multi-view images
- **BEVDet** — arXiv 2021 | [Paper](https://arxiv.org/abs/2112.11790) | [Code](https://github.com/HuangJunJie2017/BEVDet) — LSS-based BEV detection framework
- **PETR** — ECCV 2022 | [Paper](https://arxiv.org/abs/2203.05625) | [Code](https://github.com/megvii-research/PETR) — 3D position embedding for multi-view detection
- **BEVFormer** — ECCV 2022 | [Paper](https://arxiv.org/abs/2203.17270) | [Code](https://github.com/fundamentalvision/BEVFormer) — spatiotemporal transformer for BEV representation
- **BEVDepth** — AAAI 2023 | [Paper](https://arxiv.org/abs/2206.10092) | [Code](https://github.com/Megvii-BaseDetection/BEVDepth) — reliable depth supervision for BEV detection
- **SparseBEV** — ICCV 2023 | [Paper](https://arxiv.org/abs/2308.09244) | [Code](https://github.com/MCG-NJU/SparseBEV) — fully sparse high-performance detector
- **StreamPETR** — ICCV 2023 | [Paper](https://arxiv.org/abs/2303.11926) | [Code](https://github.com/exiawsh/StreamPETR) — object-centric temporal modeling
- 🔥 **Far3D** — AAAI 2024 | [Paper](https://arxiv.org/abs/2308.09616) | [Code](https://github.com/megvii-research/Far3D) — long-range surround-view 3D detection

### LiDAR-Based 3D Object Detection

- **VoxelNet** — CVPR 2018 | [Paper](https://arxiv.org/abs/1711.06396) — end-to-end voxel feature learning
- **SECOND** — Sensors 2018 | [Paper](https://www.mdpi.com/1424-8220/18/10/3337) | [Code](https://github.com/traveller59/second.pytorch) — sparse 3D convolution
- **PointRCNN** — CVPR 2019 | [Paper](https://arxiv.org/abs/1812.04244) | [Code](https://github.com/sshaoshuai/PointRCNN) — point-based two-stage detection
- **PointPillars** — CVPR 2019 | [Paper](https://arxiv.org/abs/1812.05784) — fast pillar encoding, industry workhorse
- **PV-RCNN** — CVPR 2020 | [Paper](https://arxiv.org/abs/1912.13192) | [Code](https://github.com/open-mmlab/OpenPCDet) — point-voxel feature aggregation
- **CenterPoint** — CVPR 2021 | [Paper](https://arxiv.org/abs/2006.11275) | [Code](https://github.com/tianweiy/CenterPoint) — center-based detection and tracking
- **Voxel R-CNN** — AAAI 2021 | [Paper](https://arxiv.org/abs/2012.15712) — accurate voxel-based two-stage detection
- **DSVT** — CVPR 2023 | [Paper](https://arxiv.org/abs/2301.06051) | [Code](https://github.com/Haiyang-W/DSVT) — dynamic sparse voxel transformer
- **VoxelNeXt** — CVPR 2023 | [Paper](https://arxiv.org/abs/2303.11301) | [Code](https://github.com/dvlab-research/VoxelNeXt) — fully sparse voxel detection
- 🔥 **SAFDNet** — CVPR 2024 | [Paper](https://arxiv.org/abs/2403.05817) | [Code](https://github.com/zhanggang001/HEDNet) — simple and effective fully sparse detector
- 🔥 **Voxel Mamba** — NeurIPS 2024 | [Paper](https://arxiv.org/abs/2406.10700) — group-free state space models for point clouds
- 🔥 **LION** — NeurIPS 2024 | [Paper](https://arxiv.org/abs/2407.18232) — linear group RNN backbone for 3D detection

### Fusion-Based 3D Object Detection

- **MV3D** — CVPR 2017 | [Paper](https://arxiv.org/abs/1611.07759) — pioneering multi-view LiDAR-camera fusion
- **AVOD** — IROS 2018 | [Paper](https://arxiv.org/abs/1712.02294) — aggregated view proposal fusion
- **TransFusion** — CVPR 2022 | [Paper](https://arxiv.org/abs/2203.11496) | [Code](https://github.com/XuyangBai/TransFusion) — transformer-based robust LiDAR-camera fusion
- **BEVFusion** — ICRA 2023 | [Paper](https://arxiv.org/abs/2205.13542) | [Code](https://github.com/mit-han-lab/bevfusion) — unified multi-task multi-sensor BEV fusion
- **CMT** — ICCV 2023 | [Paper](https://arxiv.org/abs/2301.01283) | [Code](https://github.com/junjie18/CMT) — cross-modal transformer, robust to sensor failure
- **UniTR** — ICCV 2023 | [Paper](https://arxiv.org/abs/2308.07732) | [Code](https://github.com/Haiyang-W/UniTR) — unified modality-agnostic transformer backbone
- 🔥 **IS-Fusion** — CVPR 2024 | [Paper](https://arxiv.org/abs/2403.15241) | [Code](https://github.com/yinjunbo/IS-Fusion) — instance-scene collaborative fusion
- 🔥 **RCBEVDet** — CVPR 2024 | [Paper](https://arxiv.org/abs/2403.16440) | [Code](https://github.com/VDIGPKU/RCBEVDet) — radar-camera BEV fusion

### 3D Occupancy Prediction

- **MonoScene** — CVPR 2022 | [Paper](https://arxiv.org/abs/2112.00726) | [Code](https://github.com/astra-vision/MonoScene) — monocular semantic scene completion
- **TPVFormer** — CVPR 2023 | [Paper](https://arxiv.org/abs/2302.07817) | [Code](https://github.com/wzzheng/TPVFormer) — tri-perspective view representation
- **SurroundOcc** — ICCV 2023 | [Paper](https://arxiv.org/abs/2303.09551) | [Code](https://github.com/weiyithu/SurroundOcc) — multi-camera dense occupancy
- **OccFormer** — ICCV 2023 | [Paper](https://arxiv.org/abs/2304.05316) | [Code](https://github.com/zhangyp15/OccFormer) — dual-path transformer for occupancy
- **Occ3D** — NeurIPS 2023 | [Paper](https://arxiv.org/abs/2304.14365) | [Code](https://github.com/Tsinghua-MARS-Lab/Occ3D) — large-scale occupancy benchmark
- **FB-OCC** — CVPRW 2023 | [Paper](https://arxiv.org/abs/2307.01492) | [Code](https://github.com/NVlabs/FB-BEV) — forward-backward view transformation, challenge winner
- **FlashOcc** — arXiv 2023 | [Paper](https://arxiv.org/abs/2311.12058) | [Code](https://github.com/Yzichen/FlashOCC) — fast and memory-efficient channel-to-height occupancy
- 🔥 **SelfOcc** — CVPR 2024 | [Paper](https://arxiv.org/abs/2311.12754) | [Code](https://github.com/huang-yh/SelfOcc) — self-supervised vision-based occupancy
- 🔥 **Cam4DOcc** — CVPR 2024 | [Paper](https://arxiv.org/abs/2311.17663) | [Code](https://github.com/haomo-ai/Cam4DOcc) — camera-only 4D occupancy forecasting benchmark
- 🔥 **SparseOcc** — ECCV 2024 | [Paper](https://arxiv.org/abs/2312.17118) | [Code](https://github.com/MCG-NJU/SparseOcc) — fully sparse occupancy prediction
- 🔥 **GaussianFormer** — ECCV 2024 | [Paper](https://arxiv.org/abs/2405.17429) | [Code](https://github.com/huang-yh/GaussianFormer) — 3D Gaussians as scene representation
- 🔥 **GaussianWorld** — CVPR 2025 | [Paper](https://arxiv.org/abs/2412.10373) — Gaussian world model for streaming occupancy
- 🔥 **S2GO** — arXiv 2025 | [Paper](https://arxiv.org/abs/2506.05473) — streaming sparse Gaussian occupancy prediction

### Temporal & Streaming Perception

- **Towards Streaming Perception** — ECCV 2020 (Best Paper Honorable Mention) | [Paper](https://arxiv.org/abs/2005.10420) — streaming accuracy metric
- **StreamYOLO** — CVPR 2022 | [Paper](https://arxiv.org/abs/2203.12338) | [Code](https://github.com/yancie-yjr/StreamYOLO) — real-time streaming detection
- **StreamPETR** — ICCV 2023 | [Paper](https://arxiv.org/abs/2303.11926) | [Code](https://github.com/exiawsh/StreamPETR) — efficient temporal multi-view 3D detection
- 🔥 **PTT** — CVPR 2024 | [Paper](https://arxiv.org/abs/2312.08371) | [Code](https://github.com/KuanchihHuang/PTT) — point-trajectory transformer for temporal 3D detection

### End-to-End Autonomous Driving

- **CIL** — ICRA 2018 | [Paper](https://arxiv.org/abs/1710.02410) — conditional imitation learning
- **ChauffeurNet** — RSS 2019 | [Paper](https://arxiv.org/abs/1812.03079) — imitating the best, synthesizing the worst
- **Learning by Cheating** — CoRL 2019 | [Paper](https://arxiv.org/abs/1912.12294) — privileged-agent distillation
- **UniAD** — CVPR 2023 (Best Paper) | [Paper](https://arxiv.org/abs/2212.10156) | [Code](https://github.com/OpenDriveLab/UniAD) — planning-oriented unified framework
- **VAD** — ICCV 2023 | [Paper](https://arxiv.org/abs/2303.12077) | [Code](https://github.com/hustvl/VAD) — vectorized scene representation for planning
- 🔥 **BEV-Planner** — CVPR 2024 | [Paper](https://arxiv.org/abs/2312.03031) | [Code](https://github.com/NVlabs/BEV-Planner) — is ego status all you need?
- 🔥 **SparseDrive** — ICRA 2025 | [Paper](https://arxiv.org/abs/2405.19620) | [Code](https://github.com/swc-17/SparseDrive) — sparse scene representation end-to-end driving
- 🔥 **DiffusionDrive** — CVPR 2025 | [Paper](https://arxiv.org/abs/2411.15139) | [Code](https://github.com/hustvl/DiffusionDrive) — truncated diffusion policy for planning
- 🔥 **RecogDrive** — arXiv 2025 | [Paper](https://arxiv.org/abs/2506.08052) — reinforced cognitive framework for end-to-end driving

### World Models for Autonomous Driving

- **GAIA-1** — arXiv 2023 | [Paper](https://arxiv.org/abs/2309.17080) — generative world model for driving (Wayve)
- **DriveDreamer** — ECCV 2024 | [Paper](https://arxiv.org/abs/2309.09777) | [Code](https://github.com/JeffWang987/DriveDreamer) — real-world-driven world model
- 🔥 **Drive-WM** — CVPR 2024 | [Paper](https://arxiv.org/abs/2311.17918) | [Code](https://github.com/BraveGroup/Drive-WM) — multiview visual forecasting and planning
- 🔥 **OccWorld** — ECCV 2024 | [Paper](https://arxiv.org/abs/2311.16038) | [Code](https://github.com/wzzheng/OccWorld) — 3D occupancy world model
- 🔥 **GenAD** — CVPR 2024 (Highlight) | [Paper](https://arxiv.org/abs/2403.09630) | [Code](https://github.com/OpenDriveLab/DriveAGI) — generalized video prediction from web-scale driving data
- 🔥 **DriveWorld** — CVPR 2024 | [Paper](https://arxiv.org/abs/2405.04390) — 4D pre-training via world models
- 🔥 **Vista** — NeurIPS 2024 | [Paper](https://arxiv.org/abs/2405.17398) | [Code](https://github.com/OpenDriveLab/Vista) — high-fidelity generalizable driving world model
- 🔥 **OccSora** — arXiv 2024 | [Paper](https://arxiv.org/abs/2405.20337) | [Code](https://github.com/wzzheng/OccSora) — 4D occupancy generation as world simulation
- 🔥 **GaussianAD** — arXiv 2024 | [Paper](https://arxiv.org/abs/2412.10371) — Gaussian-centric end-to-end driving

### VLM / LLM for Autonomous Driving

- **LMDrive** — CVPR 2024 | [Paper](https://arxiv.org/abs/2312.07488) | [Code](https://github.com/opendilab/LMDrive) — closed-loop driving with LLMs
- **DriveLM** — ECCV 2024 | [Paper](https://arxiv.org/abs/2312.14150) | [Code](https://github.com/OpenDriveLab/DriveLM) — graph visual question answering for driving
- 🔥 **DriveVLM** — CoRL 2024 | [Paper](https://arxiv.org/abs/2402.12289) — VLM-based scene understanding and planning
- 🔥 **EMMA** — arXiv 2024 | [Paper](https://arxiv.org/abs/2410.23262) — end-to-end multimodal model for driving (Waymo)

### V2X Cooperative Perception

**Methods**

- **V2VNet** — ECCV 2020 | [Paper](https://arxiv.org/abs/2008.07519) — joint perception and prediction via V2V
- **V2X-ViT** — ECCV 2022 | [Paper](https://arxiv.org/abs/2203.10638) | [Code](https://github.com/DerrickXuNu/v2x-vit) — heterogeneous multi-agent vision transformer
- **Where2comm** — NeurIPS 2022 | [Paper](https://arxiv.org/abs/2209.12836) | [Code](https://github.com/MediaBrain-SJTU/Where2comm) — communication-efficient collaboration via spatial confidence
- **CoBEVT** — CoRL 2022 | [Paper](https://arxiv.org/abs/2207.02202) | [Code](https://github.com/DerrickXuNu/CoBEVT) — sparse transformer for cooperative BEV
- **CoAlign** — ICRA 2023 | [Paper](https://arxiv.org/abs/2211.07214) | [Code](https://github.com/yifanlu0227/CoAlign) — robust to pose errors
- **HM-ViT** — ICCV 2023 | [Paper](https://arxiv.org/abs/2304.10628) — hetero-modal V2V cooperative perception
- **FFNet** — NeurIPS 2023 | [Paper](https://arxiv.org/abs/2303.10552) | [Code](https://github.com/haibao-yu/FFNet-VIC3D) — flow-based feature fusion against latency
- 🔥 **HEAL** — ICLR 2024 | [Paper](https://arxiv.org/abs/2401.13964) | [Code](https://github.com/yifanlu0227/HEAL) — extensible heterogeneous collaboration
- 🔥 **UniV2X** — AAAI 2025 | [Paper](https://arxiv.org/abs/2404.00717) | [Code](https://github.com/AIR-THU/UniV2X) — end-to-end driving through V2X cooperation
- 🔥 **V2X-R** — CVPR 2025 | [Paper](https://arxiv.org/abs/2411.08402) — cooperative LiDAR-4D radar fusion with denoising diffusion
- 🔥 **STAMP** — ICLR 2025 | [Paper](https://openreview.net/forum?id=8NdNniulYE) — scalable task- and model-agnostic collaboration
- 🔥 **CoSDH** — CVPR 2025 | [Paper](https://arxiv.org/abs/2503.03430) — supply-demand-aware communication-efficient collaboration
- 🔥 **CoopTrack** — ICCV 2025 (Highlight) | [Paper](https://arxiv.org/abs/2507.19239) — end-to-end cooperative sequential perception

**Datasets & Benchmarks** — see the [V2X datasets table](#v2x--cooperative-perception) above (OPV2V, V2XSet, V2X-Sim, DAIR-V2X, V2V4Real, TUMTraf V2X, RCooper).

### Robustness, Domain Adaptation & Test-Time Adaptation

- **ST3D** — CVPR 2021 | [Paper](https://arxiv.org/abs/2103.05346) | [Code](https://github.com/CVMI-Lab/ST3D) — self-training for UDA in 3D detection
- **3D Common Corruptions Benchmark** — CVPR 2023 | [Paper](https://arxiv.org/abs/2303.11040) — benchmarking robustness of 3D detectors to corruptions
- **Robo3D** — ICCV 2023 | [Paper](https://arxiv.org/abs/2303.17597) | [Code](https://github.com/ldkong1205/Robo3D) — robust and reliable 3D perception benchmark
<!-- TODO: add your own V2X-TTA (IEEE IoT Journal) and PseudoTTA papers here with links once you have the DOIs -->

---

## Related Surveys

- **3D Object Detection for Autonomous Driving: A Comprehensive Survey** — IJCV 2023 | [Paper](https://arxiv.org/abs/2206.09474)
- **3D Object Detection from Images for Autonomous Driving: A Survey** — TPAMI 2023 | [Paper](https://arxiv.org/abs/2202.02980)
- **A Survey on Occupancy Perception for Autonomous Driving: The Information Fusion Perspective** — Information Fusion 2025 | [GitHub](https://github.com/HuaiyuanXu/3D-Occupancy-Perception)
- **End-to-End Autonomous Driving: Challenges and Frontiers** — TPAMI 2024 | [Paper](https://arxiv.org/abs/2306.16927) | [GitHub](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)

---

## Contributing

Contributions are welcome! If you have a paper, dataset, or toolbox that fits this collection:

1. Fork the repository and create a branch.
2. Follow the entry format: `**Method** — Venue Year | [Paper](link) | [Code](link) — one-line description`.
3. Open a Pull Request, or simply [open an issue](https://github.com/Fishsoup0/Autonomous-Driving-Perception/issues) with the paper link.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Citation

If you find this survey and repository useful in your research, please consider citing:

```bibtex
@article{wang2024comprehensive,
  title   = {A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions},
  author  = {Wang, Yu and Wang, Shaohua and Li, Yicheng and Liu, Mingchun},
  journal = {IEEE Sensors Journal},
  year    = {2025},
  note    = {TODO: fill in volume, number, pages, and DOI of the published version},
}
```

arXiv preprint: [arXiv:2408.16530](https://arxiv.org/abs/2408.16530)

---

## License

This repository is released under the [MIT License](LICENSE).

---

<div align="center">

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Fishsoup0/Autonomous-Driving-Perception&type=Date)](https://star-history.com/#Fishsoup0/Autonomous-Driving-Perception&Date)

</div>
