# [Robustness Evaluation in Hand Pose Estimation Models using Metamorphic Testing](https://ieeexplore.ieee.org/document/10190429)

In this work, we adopt metamorphic testing to evaluate the robustness of hand pose estimation on 
four state-of-the-art models: [MediaPipe hands](https://github.com/google/mediapipe), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose),
[BodyHands](https://github.com/cvlab-stonybrook/BodyHands), and [NSRM hand](https://github.com/HowieMa/NSRMhand).

Occlusions, illumination variations and motion blur are indetified as the main obstacles to the performance of existing hand pose estimation models. 
Considering their influence on the HPE models, we transform the source test case obtianed from two public hand pose datasets: 
[FreiHand](https://github.com/lmb-freiburg/freihand) and [CMU Panoptic Hand](http://domedb.perception.cs.cmu.edu/handdb.html) 
to construct the corresponding follow-up test cases, and propose the following metamorphic relations: 

![MR_Summary](https://user-images.githubusercontent.com/86390633/212545001-72f63b9f-97c0-441d-b292-39b6a5806504.png)

The experimental results are uploaded and placed at the corresponding folders of this repositories at: 
[Source test cases](https://github.com/mpuu00001/Robustness-Evaluation-in-Hand-Pose-Estimation/tree/main/Source%20test%20cases), 
[MR1](https://github.com/mpuu00001/Robustness-Evaluation-in-Hand-Pose-Estimation/tree/main/MR1), 
[MR2](https://github.com/mpuu00001/Robustness-Evaluation-in-Hand-Pose-Estimation/tree/main/MR2),
[MR3](https://github.com/mpuu00001/Robustness-Evaluation-in-Hand-Pose-Estimation/tree/main/MR3), and 
[MR4](https://github.com/mpuu00001/Robustness-Evaluation-in-Hand-Pose-Estimation/tree/main/MR4).

Here are samples of hands in different test cases:

<img src="https://github.com/user-attachments/assets/723eb412-4c83-4505-a6d7-9f39d6e4d22b" alt="Presentation1" width="350"/>

## License
The code is released under the MIT license.
