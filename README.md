

<p align="center">
    <a href="https://arxiv.org/abs/2408.16530">
        <img width="350" alt="image" src="assets/image.jpg">
    </a>
</p>

<p align="center">
    <strong>Yu Wang</strong>
    .
    <strong>Shaohua Wang</strong>
    .
    <strong>Yicheng Li</strong>
        .
    <strong>Mingchun Liu</strong>
</p>

<p align="center">
    <a href='https://arxiv.org/abs/2408.16530'>
        <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
</p>

# A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions

This repository is associated with the review paper titled “A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions,” which provides an extensive overview of recent advancements in 3D object perception for autonomous driving systems. The review covers key approaches including Camera-Based Detection, LiDAR-Based Detection, and Fusion Detection Techniques.

We provide a thorough analysis of the strengths and limitations of each method, highlighting advancements in accuracy and robustness. Furthermore, the review discusses future directions such as Temporal Perception, Occupancy Grids, End-to-End Learning Frameworks, and Cooperative Perception methods, which extend the perception range through collaborative communication.

This repository will be actively maintained with continuous updates on the latest advancements in 3D object detection for autonomous driving. By offering a comprehensive view of the current state and future developments, we aim to be a valuable resource for researchers and practitioners in this field.

## Overview

- **Paper Title**: A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions
- **Authors**: [Yu Wang](https://scholar.google.com/citations?user=EsZuitIAAAAJ&hl=zh-TW), Shaohua Wang, Yicheng Li，Mingchun Liu
- **Link to Paper**: [arXiv](https://arxiv.org/abs/2408.16530)

## Abstract

In recent years, 3D object perception has become a crucial component in the development of autonomous driving systems, providing essential environmental awareness. However, as perception tasks in autonomous driving evolve, their variants have increased, leading to diverse insights from industry and academia. Currently, there is a lack of comprehensive surveys that collect and summarize these perception tasks and their developments from a broader perspective. This review extensively summarizes traditional 3D object detection methods, focusing on camera-based, LiDAR-based, and fusion detection techniques. We provide a comprehensive analysis of the strengths and limitations of each approach, highlighting advancements in accuracy and robustness. Furthermore, we discuss future directions, including methods to improve accuracy such as temporal perception, occupancy grids, and end-to-end learning frameworks. We also explore cooperative perception methods that extend the perception range through collaborative communication. By providing a holistic view of the current state and future developments in 3D object perception, we aim to offer a more comprehensive understanding of perception tasks for autonomous driving. 

## Key Contributions

1. To our knowledge, this is the first time that different development trends in autonomous driving environmental perception have been summarized and analyzed, providing a holistic view of the evolution and future trends in 3D object perception.

2. We provide a comprehensive summary, classification, and analysis of the latest methods in camera-based, LiDAR-based, and fusion-based 3D object detection.

3. We offer a panoramic view of perception in autonomous driving environments, not only summarizing the perception methods comprehensively but also compiling datasets and evaluation metrics used by different methods to promote research insights.


# Dataset Resources

## Vehicle-Road Collaboration Datasets

1. **OPV2V **
   - **Type**: Simulated, Lidar (L), Camera (C)
   - **Use Cases**: V2V communication.
   - **Link**: [OPV2V Dataset](https://github.com/DerrickXuNu/OpenCOOD)

2. **V2X-Sim **
   - **Type**: Simulated, Lidar (L), Camera (C), GPS/IMU
   - **Use Cases**: V2V, V2I communication.
   - **Link**: [V2X-Sim Dataset](https://github.com/AIR-THU/DAIR-V2X)

3. **V2XSet **
   - **Type**: Simulated, Lidar (L), Camera (C)
   - **Use Cases**: V2V, V2I communication.
   - **Link**: [V2XSet Dataset](https://github.com/AIR-THU/DAIR-V2X)

4. **Rope3D **
   - **Type**: Real, Camera (C), GPS/IMU
   - **Use Cases**: Camera-based localization.
   - **Link**: [Rope3D Dataset](https://github.com/MasterHow/Rope3D)

5. **DAIR-V2X **
   - **Type**: Real, Lidar (L), Camera (C), GPS/IMU
   - **Use Cases**: V2I communication, sensor fusion.
   - **Link**: [DAIR-V2X Dataset](https://thudair.baai.ac.cn/index)

6. **V2X-Real **
   - **Type**: Real, Lidar (L), Camera (C), GPS/IMU
   - **Use Cases**: V2V, V2I, I2I communication.
   - **Link**: [V2X-Real Dataset](https://github.com/AIR-THU/DAIR-V2X)

## 3D Occupancy Datasets

1. **Occ3D **
   - **Type**: Real, Lidar (L), Camera (C)
   - **Use Cases**: 3D occupancy, semantic segmentation.
   - **Link**: [Occ3D Dataset](https://github.com/alexklwong/occ3d)

2. **Semantic KITTI **
   - **Type**: Real, Lidar (L)
   - **Use Cases**: Semantic segmentation, SLAM.
   - **Link**: [Semantic KITTI Dataset](http://www.semantic-kitti.org/)

3. **KITTI-360 **
   - **Type**: Real, Lidar (L), Camera (C)
   - **Use Cases**: Mapping, localization.
   - **Link**: [KITTI-360 Dataset](http://www.cvlibs.net/datasets/kitti-360/)

# Simulators

## CARLA Simulator
- **Description**: CARLA is an open-source simulator for autonomous driving research. It provides realistic environments and supports various sensors, including cameras, lidar, and GPS. It's widely used for training and testing autonomous driving models in a controlled and customizable environment.
- **Link**: [CARLA Simulator](https://carla.org/)

## LGSVL Simulator
- **Description**: LGSVL (now known as SVL Simulator) is an autonomous vehicle simulator developed by LG Electronics. It offers high-fidelity simulations and integrates with popular frameworks like Autoware and Apollo. It's ideal for testing vehicle perception, planning, and control algorithms in complex scenarios.
- **Link**: [SVL Simulator](https://www.svlsimulator.com/)

## Microsoft AirSim
- **Description**: AirSim is a cross-platform open-source simulator developed by Microsoft. It supports the simulation of drones, cars, and other vehicles in 3D environments. AirSim is compatible with Unreal Engine and offers APIs for deep integration with machine learning frameworks.
- **Link**: [Microsoft AirSim](https://github.com/microsoft/AirSim)

# High-Quality Papers on 3D Object Detection

## Camera-Based 3D Object Detection

1. **Mono3D: Monocular 3D Object Detection for Autonomous Driving**
   - **Description**: Introduces Mono3D, a method for predicting 3D object proposals from a single image using a multi-view approach. Pioneering work in monocular 3D object detection.
   - **Link**: [Mono3D](https://arxiv.org/abs/1608.07711)
   - **Year**: 2016

2. **M3D-RPN: Monocular 3D Region Proposal Network for Object Detection**
   - **Description**: Presents a region proposal network that directly predicts 3D bounding boxes from monocular images, advancing the state-of-the-art in monocular 3D detection.
   - **Link**: [M3D-RPN](https://arxiv.org/abs/1907.06038)
   - **Year**: 2019

3. **MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer**
   - **Description**: A recent method that leverages depth-aware transformers for monocular 3D object detection, achieving state-of-the-art results.
   - **Link**: [MonoDTR](https://arxiv.org/abs/2203.10981)
   - **Year**: 2022

## Lidar-Based 3D Object Detection

1. **VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection**
   - **Description**: One of the first methods to directly apply deep learning on raw point clouds, using voxelization and 3D convolutional networks for 3D object detection.
   - **Link**: [VoxelNet](https://arxiv.org/abs/1711.06396)
   - **Year**: 2018

2. **PointPillars: Fast Encoders for Object Detection from Point Clouds**
   - **Description**: Uses vertical columns (pillars) to encode point cloud data, enabling fast and efficient 3D object detection, widely used in real-time applications.
   - **Link**: [PointPillars](https://arxiv.org/abs/1812.05784)
   - **Year**: 2019

3. **CenterPoint: Center-based 3D Object Detection and Tracking**
   - **Description**: A more recent approach that represents objects as points and shows strong performance on both detection and tracking tasks.
   - **Link**: [CenterPoint](https://arxiv.org/abs/2006.11275)
   - **Year**: 2020

## Fusion-Based 3D Object Detection

1. **MV3D: Multi-View 3D Object Detection Network**
   - **Description**: Combines RGB images and lidar point clouds for robust 3D object detection, a pioneering work in multi-sensor fusion.
   - **Link**: [MV3D](https://arxiv.org/abs/1611.07759)
   - **Year**: 2017

2. **AVOD: Aggregated View Object Detection in Autonomous Driving**
   - **Description**: A two-stage object detection framework that combines lidar point clouds with RGB images, achieving high accuracy in 3D detection.
   - **Link**: [AVOD](https://arxiv.org/abs/1712.02294)
   - **Year**: 2018

3. **TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers**
   - **Description**: A recent method that uses transformers to achieve robust lidar-camera fusion for 3D object detection, pushing the boundaries of fusion techniques.
   - **Link**: [TransFusion](https://arxiv.org/abs/2203.11496)
   - **Year**: 2022


# High-Quality Papers on 3D Occupancy Prediction

1. **OccNet: Occupancy Networks for 3D Reconstruction in a Single Forward Pass**
   - **Description**: Introduces Occupancy Networks (OccNet), a deep learning framework for predicting continuous occupancy values in 3D space, enabling high-quality and detailed 3D reconstruction from sparse inputs.
   - **Link**: [OccNet](https://arxiv.org/abs/1905.09662)
   - **Year**: 2019

2. **Predicting Sharp and Accurate Occupancy Grids Using Variational Autoencoders**
   - **Description**: Proposes the use of variational autoencoders (VAEs) to predict sharp and accurate 3D occupancy grids from lidar data, improving the reliability of occupancy predictions in autonomous driving.
   - **Link**: [Predicting Sharp Occupancy Grids](https://arxiv.org/abs/2007.09718)
   - **Year**: 2020

3. **Deep Occupancy Flow: 3D Motion Prediction from Partial Observations**
   - **Description**: A recent work that combines occupancy prediction with motion flow estimation, enabling the prediction of dynamic 3D occupancy grids from partial observations, particularly useful for autonomous driving.
   - **Link**: [Deep Occupancy Flow](https://arxiv.org/abs/2203.17292)
   - **Year**: 2022
  
# High-Quality Papers on Streaming Perception

1. **StreamYOLO: Real-Time Object Detection for Streaming Perception**
   - **Description**: StreamYOLO is designed for real-time object detection in streaming perception scenarios, optimizing latency and accuracy by employing a novel feature alignment mechanism. It’s particularly suited for applications requiring continuous perception, such as autonomous driving.
   - **Link**: [StreamYOLO](https://arxiv.org/abs/2301.02078)
   - **Year**: 2023

2. **Towards Streaming Perception**
   - **Description**: This paper introduces the concept of streaming perception, focusing on real-time processing of sensor data to maintain continuous perception in dynamic environments. The authors propose new benchmarks and models to handle the challenges of streaming data.
   - **Link**: [Towards Streaming Perception](https://arxiv.org/abs/2207.02174)
   - **Year**: 2022

3. **STM: SpatioTemporal Modeling for Efficient Online Video Object Detection**
   - **Description**: STM (SpatioTemporal Modeling) enhances online video object detection by integrating spatial and temporal features to maintain high accuracy and efficiency in streaming perception scenarios, addressing the challenges of real-time processing.
   - **Link**: [STM](https://arxiv.org/abs/1912.05083)
   - **Year**: 2020
  
# High-Quality Papers on End-to-End Autonomous Driving

1. **End-to-End Driving via Conditional Imitation Learning**
   - **Description**: This paper introduces a framework for end-to-end autonomous driving using conditional imitation learning. The model learns to predict driving actions directly from sensory input based on high-level commands, effectively bridging the gap between perception and control.
   - **Link**: [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
   - **Year**: 2018

2. **ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst**
   - **Description**: ChauffeurNet combines imitation learning with data augmentation techniques to handle challenging driving scenarios. The model learns end-to-end driving policies by imitating expert drivers while also synthesizing difficult driving situations to improve robustness.
   - **Link**: [ChauffeurNet](https://arxiv.org/abs/1812.03079)
   - **Year**: 2018

3. **Learning by Cheating: Imitating Features from Graphical Environments for Real-World Reinforcement Learning**
   - **Description**: This paper presents a novel approach where a model is first trained in a simulated environment using rich graphical features, and then fine-tuned in the real world. This method allows for the transfer of end-to-end driving skills from simulation to reality, leveraging the advantages of both environments.
   - **Link**: [Learning by Cheating](https://arxiv.org/abs/1912.12294)
   - **Year**: 2020
# High-Quality Papers on Vehicle-Road Collaboration (V2X Communication)

1. **Cooperative Perception with V2X Communication: Exploring the Design Space**
   - **Description**: This paper explores the design space for cooperative perception using V2X communication, analyzing the trade-offs between different communication strategies and their impact on perception performance in autonomous driving scenarios. It provides a comprehensive study of how vehicle-road collaboration can enhance situational awareness.
   - **Link**: [Cooperative Perception with V2X Communication](https://arxiv.org/abs/2004.07074)
   - **Year**: 2020

2. **V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Prediction**
   - **Description**: V2VNet introduces a framework for vehicle-to-vehicle (V2V) communication that enables joint perception and prediction across multiple vehicles. This approach improves the accuracy and robustness of autonomous driving systems by sharing sensor data and predictive models between vehicles.
   - **Link**: [V2VNet](https://arxiv.org/abs/2008.07519)
   - **Year**: 2020

3. **V2XSet: An Extended Dataset for Vehicle-to-Everything (V2X) Cooperative Perception**
   - **Description**: V2XSet extends existing datasets to include V2X scenarios, providing a rich resource for training and evaluating vehicle-road collaboration algorithms. The paper discusses the challenges of V2X communication and presents baseline results for cooperative perception tasks.
   - **Link**: [V2XSet](https://arxiv.org/abs/2203.10168)
   - **Year**: 2022

4. **V2X-Sim: A Simulation Dataset for Multi-Agent Collaborative Perception**
   - **Description**: V2X-Sim is a simulation dataset designed for multi-agent collaborative perception in V2X scenarios. It provides diverse and challenging environments to train and evaluate vehicle-road collaboration models, focusing on the interaction between vehicles and infrastructure in a simulated setting.
   - **Link**: [V2X-Sim](https://arxiv.org/abs/2109.12434)
   - **Year**: 2021

## Related Work

# CVPR 2024 Papers Autonomous Driving

This repository is continuously updated. We prioritize including articles that have already been submitted to arXiv.

We kindly invite you to our platform, Auto Driving Heart, for paper interpretation and sharing. If you would like to promote your paper, please feel free to contact me.



### 1) End to End | 端到端自动驾驶

**Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?**

- Paper: https://arxiv.org/pdf/2312.03031.pdf
- Code: https://github.com/NVlabs/BEV-Planner

**Visual Point Cloud Forecasting enables Scalable Autonomous Driving**

- Paper: https://arxiv.org/pdf/2312.17655.pdf
- Code: https://github.com/OpenDriveLab/ViDAR

**PlanKD: Compressing End-to-End Motion Planner for Autonomous Driving**

- Paper: https://arxiv.org/pdf/2403.01238.pdf
- Code: https://github.com/tulerfeng/PlanKD

**VLP: Vision Language Planning for Autonomous Driving**

- Paper：https://arxiv.org/abs/2401.05577

### 2）LLM Agent | 大语言模型智能体

**ChatSim: Editable Scene Simulation for Autonomous Driving via LLM-Agent Collaboration**

- Paper: https://arxiv.org/pdf/2402.05746.pdf
- Code: https://github.com/yifanlu0227/ChatSim

**LMDrive: Closed-Loop End-to-End Driving with Large Language Models**

- Paper: https://arxiv.org/pdf/2312.07488.pdf
- Code: https://github.com/opendilab/LMDrive

**MAPLM: A Real-World Large-Scale Vision-Language Dataset for Map and Traffic Scene Understanding**

- Code: https://github.com/LLVM-AD/MAPLM

**One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

- Paper：https://arxiv.org/pdf/2403.01849.pdf
- Code：https://github.com/TreeLLi/APT

**PromptKD: Unsupervised Prompt Distillation for Vision-Language Models**

- Paper：https://arxiv.org/pdf/2403.02781

**RegionGPT: Towards Region Understanding Vision Language Model**

- Paper：https://arxiv.org/pdf/2403.02330

**Towards Learning a Generalist Model for Embodied Navigation**

- Paper: https://arxiv.org/pdf/2312.02010.pdf
- Code: https://github.com/zd11024/NaviLLM

### 3）SSC: Semantic Scene Completion | 语义场景补全

**Symphonize 3D Semantic Scene Completion with Contextual Instance Queries**

- Paper: https://arxiv.org/pdf/2306.15670.pdf
- Code: https://github.com/hustvl/Symphonies

**PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness**

- Paper: https://arxiv.org/pdf/2312.02158.pdf
- Code: https://github.com/astra-vision/PaSCo

**SemCity: Semantic Scene Generationwith Triplane Diffusion**

- Paper: https://arxiv.org/pdf/2403.07773.pdf
- Code: https://github.com/zoomin-lee/SemCity

### 4）OCC: Occupancy Prediction | 占用感知

**SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction**

- Paper: https://arxiv.org/pdf/2311.12754.pdf
- Code: https://github.com/huang-yh/SelfOcc

**Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications**

- Paper: https://arxiv.org/pdf/2311.17663.pdf
- Code: https://github.com/haomo-ai/Cam4DOcc

**PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation**

- Paper: https://arxiv.org/pdf/2306.10013.pdf
- Code: https://github.com/Robertwyq/PanoOcc

### 5) World Model | 世界模型

**Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving**

- Paper: https://arxiv.org/pdf/2311.17918.pdf
- Code: https://github.com/BraveGroup/Drive-WM

### 6）车道线检测

**Lane2Seq: Towards Unified Lane Detection via Sequence Generation**

- Paper：https://arxiv.org/abs/2402.17172

### 7）Pre-training | 预训练

**UniPAD: A Universal Pre-training Paradigm for Autonomous Driving**

- Paper: https://arxiv.org/pdf/2310.08370.pdf
- Code: https://github.com/Nightmare-n/UniPAD

### 8）AIGC | 人工智能内容生成

**Panacea: Panoramic and Controllable Video Generation for Autonomous Driving**

- Paper: https://arxiv.org/pdf/2311.16813.pdf
- Code: https://github.com/wenyuqing/panacea

**SemCity: Semantic Scene Generation with Triplane Diffusion**

- Paper:
- Code: https://github.com/zoomin-lee/SemCity

**BerfScene: Bev-conditioned Equivariant Radiance Fields for Infinite 3D Scene Generation**

- Paper: https://arxiv.org/pdf/2312.02136.pdf
- Code: https://github.com/zqh0253/BerfScene

### 9）3D Object Detection | 三维目标检测

**PTT: Point-Trajectory Transformer for Efficient Temporal 3D Object Detection**

- Paper: https://arxiv.org/pdf/2312.08371.pdf
- Code: https://github.com/KuanchihHuang/PTT

**SeaBird: Segmentation in Bird’s View with Dice Loss Improves Monocular 3D Detection of Large Objects**

- Paper: https://arxiv.org/pdf/2403.20318
- Code: https://github.com/abhi1kumar/SeaBird

**VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection**

- Code: https://github.com/skmhrk1209/VSRD

**CaKDP: Category-aware Knowledge Distillation and Pruning Framework for Lightweight 3D Object Detection**

- Code: https://github.com/zhnxjtu/CaKDP

**CN-RMA: Combined Network with Ray Marching Aggregation for 3D Indoors Object Detection from Multi-view Images**

- Paper：https://arxiv.org/abs/2403.04198
- Code：https://github.com/SerCharles/CN-RMA

**UniMODE: Unified Monocular 3D Object Detection**

- Paper：https://arxiv.org/abs/2402.18573

**Enhancing 3D Object Detection with 2D Detection-Guided Query Anchors**

- Paper：https://arxiv.org/abs/2403.06093
- Code：https://github.com/nullmax-vision/QAF2D

**SAFDNet: A Simple and Effective Network for Fully Sparse 3D Object Detection**

- Paper：https://arxiv.org/abs/2403.05817
- Code：https://github.com/zhanggang001/HEDNet

**RadarDistill: Boosting Radar-based Object Detection Performance via Knowledge Distillation from LiDAR Features**

- Paper：https://arxiv.org/pdf/2403.05061

**IS-Fusion: Instance-Scene Collaborative Fusion for Multimodal 3D Object Detection**

- Paper: https://arxiv.org/pdf/2403.15241.pdf
- Code: https://github.com/yinjunbo/IS-Fusion

**RCBEVDet: Radar-camera Fusion in Bird’s Eye View for 3D Object Detection**

- Paper: https://arxiv.org/pdf/2403.16440.pdf
- Code: https://github.com/VDIGPKU/RCBEVDet

**MonoCD: Monocular 3D Object Detection with Complementary Depths**

- Paper: 
- Code: https://github.com/dragonfly606/MonoCD

### 10）Stereo Matching | 双目立体匹配

**MoCha-Stereo: Motif Channel Attention Network for Stereo Matching**

- Code: https://github.com/ZYangChen/MoCha-Stereo

**Learning Intra-view and Cross-view Geometric Knowledge for Stereo Matching**

- Paper：https://arxiv.org/abs/2402.19270
- Code：https://github.com/DFSDDDDD1199/ICGNet

**Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching**

- Paper：https://arxiv.org/abs/2403.00486
- Code：https://github.com/Windsrain/Selective-Stereo

**Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching**

- Paper: https://arxiv.org/pdf/2306.15612.pdf
- Code: https://github.com/xxxupeng/ADL

**Neural Markov Random Field for Stereo Matching**

- Paper: https://arxiv.org/pdf/2403.11193.pdf
- Code: https://github.com/aeolusguan/NMRF

### 11）Cooperative Perception | 协同感知

**RCooper: A Real-world Large-scale Dataset for Roadside Cooperative Perception**

- Code: https://github.com/ryhnhao/RCooper

### 12）SLAM

**SNI-SLAM: SemanticNeurallmplicit SLAM**

- Paper: https://arxiv.org/pdf/2311.11016.pdf

**CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition**

- Paper：https://arxiv.org/abs/2402.19231
- Code：https://github.com/Lu-Feng/CricaVPR

**Implicit Event-RGBD Neural SLAM**

- Paper: https://arxiv.org/pdf/2311.11013.pdf
- Code: https://github.com/DelinQu/EN-SLAM


### 13）Scene Flow Estimation | 场景流估计

**DifFlow3D: Toward Robust Uncertainty-Aware Scene Flow Estimation with Iterative Diffusion-Based Refinement**

- Paper: https://arxiv.org/pdf/2311.17456.pdf
- Code: https://github.com/IRMVLab/DifFlow3D

**3DSFLabeling: Boosting 3D Scene Flow Estimation by Pseudo Auto Labeling**

- Paper: https://arxiv.org/pdf/2402.18146.pdf
- Code: https://github.com/jiangchaokang/3DSFLabelling

**Regularizing Self-supervised 3D Scene Flows with Surface Awareness and Cyclic Consistency**

- Paper: https://arxiv.org/pdf/2312.08879.pdf
- Code: https://github.com/vacany/sac-flow

### 14）Point Cloud | 点云

**Point Transformer V3: Simpler, Faster, Stronger**

- Paper: https://arxiv.org/pdf/2312.10035.pdf
- Code: https://github.com/Pointcept/PointTransformerV3

**Rethinking Few-shot 3D Point Cloud Semantic Segmentation**

- Paper: https://arxiv.org/pdf/2403.00592.pdf
- Code: https://github.com/ZhaochongAn/COSeg

**PDF: A Probability-Driven Framework for Open World 3D Point Cloud Semantic Segmentation**

- Code: https://github.com/JinfengX/PointCloudPDF

**Weakly Supervised Point Cloud Semantic Segmentation via Artificial Oracle**

- Paper: 
- Code: https://github.com/jihun1998/AO

**GLiDR: Topologically Regularized Graph Generative Network for Sparse LiDAR Point Clouds**

- Paper: 
- Code: https://github.com/GLiDR-CVPR2024/GLiDR


### 15)  Efficient Network

**Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications**

- Paper: https://arxiv.org/pdf/2401.06197.pdf

**RepViT: Revisiting Mobile CNN From ViT Perspective**

- Paper: https://arxiv.org/pdf/2307.09283.pdf
- Code: https://github.com/THU-MIG/RepViT

### 16) Segmentation

**OMG-Seg: Is One Model Good Enough For All Segmentation?**

- Paper: https://arxiv.org/pdf/2401.10229.pdf
- Code: https://github.com/lxtGH/OMG-Seg

**Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation**

- Paper: https://arxiv.org/pdf/2312.04265.pdf
- Code: https://github.com/w1oves/Rein

**SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation**

- Paper：https://arxiv.org/abs/2311.15707

**SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation**

- Paper：https://arxiv.org/abs/2311.15537

**Style Blind Domain Generalized Semantic Segmentation via Covariance Alignment and Semantic Consistence Contrastive Learning**

- Paper：https://arxiv.org/abs/2403.06122

### 17）Radar | 毫米波雷达

**DART: Doppler-Aided Radar Tomography**

- Code: https://github.com/thetianshuhuang/dart

**RadSimReal: Bridging the Gap Between Synthetic and Real Data in Radar Object Detection With Simulation**

- Code: https://github.com/yuvalHG/RadSimReal

### 18）Nerf与Gaussian Splatting

**DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes**

- Paper: https://arxiv.org/pdf/2312.07920.pdf
- Code: https://github.com/VDIGPKU/DrivingGaussian

**Dynamic LiDAR Re-simulation using Compositional Neural Fields**

- Paper: https://arxiv.org/pdf/2312.05247.pdf
- Code: https://github.com/prs-eth/Dynamic-LiDAR-Resimulation

**NARUTO: Neural Active Reconstruction from Uncertain Target Observations**

- Paper：https://arxiv.org/abs/2402.18771

**DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization**

- Paper：https://arxiv.org/abs/2403.06912

### 19）MOT: Muti-object Tracking | 多物体跟踪

**Delving into the Trajectory Long-tail Distribution for Muti-object Tracking**

- Code: https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT

**DeconfuseTrack:Dealing with Confusion for Multi-Object Tracking**

- Paper：https://arxiv.org/abs/2403.02767

### 20）Multi-label Atomic Activity Recognition

**Action-slot: Visual Action-centric Representations for Multi-label Atomic Activity Recognition in Traffic Scenes**

- Paper: https://arxiv.org/pdf/2311.17948.pdf
- Code: https://github.com/HCIS-Lab/Action-slot

### 21) Motion Prediction | 运动预测

**SmartRefine: An Scenario-Adaptive Refinement Framework for Efficient Motion Prediction**

- Code: https://github.com/opendilab/SmartRefine

### 22) Trajectory Prediction | 轨迹预测

**Test-Time Training of Trajectory Prediction via Masked Autoencoder and Actor-specific Token Memory**

- Paper: https://arxiv.org/pdf/2403.10052.pdf
- Code: https://github.com/daeheepark/T4P

**Producing and Leveraging Online Map Uncertainty in Trajectory Prediction**

- Paper: https://arxiv.org/pdf/2403.16439.pdf
- Code: https://github.com/alfredgu001324/MapUncertaintyPrediction

### 23) Depth Estimation | 深度估计

**AFNet: Adaptive Fusion of Single-View and Multi-View Depth for Autonomous Driving**

- Paper: https://arxiv.org/pdf/2403.07535.pdf
- Code: https://github.com/Junda24/AFNet

### 24) Event Camera | 事件相机

**Seeing Motion at Nighttime with an Event Camera**

- Paper: 
- Code: https://github.com/Liu-haoyue/NER-Net?tab=readme-ov-file

## Citation

If you find this review useful in your research, please consider citing:
@misc{wang2024comprehensivereview3dobject,
      title={A Comprehensive Review of 3D Object Detection in Autonomous Driving: Technological Advances and Future Directions}, 
      author={Yu Wang and Shaohua Wang and Yicheng Li and Mingchun Liu},
      year={2024},
      eprint={2408.16530},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.16530}, 
}
