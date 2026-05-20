# Deep-Learning-Final-Project | Comparing-VGG-Style-and-MobileNetV2-Style-CNN-Architectures

Abstract
The deployment of Deep Learning models in resource-constrained environments necessitates a fundamental trade-off between predictive accuracy and computational efficiency. This project investigates the architectural efficiency of Convolutional Neural Networks (CNNs) by implementing and contrasting two distinct paradigms: a standard dense 3-layer VGG-style model and a lightweight MobileNet-style model leveraging depthwise separable convolutions. Evaluated on the CIFAR-10 dataset, the MobileNet-style architecture achieved a 91.7% reduction in parameters and an 87.2% reduction in theoretical floating-point operations (FLOPs) compared to the VGG-style model, while maintaining a competitive test accuracy of 82.25% (vs. 86.57%). However, empirical evaluation on an NVIDIA T4 GPU revealed a paradox: the theoretical efficiency of the lightweight model did not translate to faster inference latency, exposing the memory-bandwidth limitations of depthwise convolutions on standard parallel hardware. This report critically analyzes these trade-offs and proposes Hardware-Aware Neural Architecture Search (HW-NAS) for future edge-AI applications.

1. Introduction

The rapid growth of deep learning applications has created an increasing demand for models that balance high predictive accuracy with computational efficiency. Convolutional Neural Networks (CNNs) have become the standard architecture for image classification tasks, yet they vary dramatically in their design philosophy ranging from deep, parameter-heavy models that maximize accuracy to lightweight models engineered for speed and minimal resource consumption. However, traditional architectures achieve high accuracy through massive over-parameterization, resulting in substantial memory footprints and high power consumption. This structural inefficiency poses a significant barrier to deploying AI on edge devices, such as Unmanned Aerial Vehicles (UAVs), IoT sensors, and mobile platforms, which operate under strict thermal, battery, and compute constraints.
To bridge this gap, researchers have developed lightweight architectures that optimize the fundamental convolutional operations. The primary objective of this project is to implement, train, and critically evaluate the performance trade-offs between a traditional dense CNN (VGG-style) and a lightweight parameter-efficient CNN (MobileNet-style) architectures on the CIFAR-10 benchmark dataset: a VGG-style model characterized by stacked convolutional layers with increasing filter depths, and a MobileNet-style model that employs depth wise separable convolutions to dramatically reduce computational cost.
The central research question guiding this project is: Can a significantly lighter CNN achieve competitive classification accuracy compared to a heavier, more parameter-rich model and what does the answer reveal about the practical design of CNNs for resource-constrained environments such as mobile devices, embedded systems, and edge computing hardware?
Both models were trained for 15 epochs on CIFAR-10 using identical training configurations to ensure a fair comparison. Performance was evaluated not only on classification accuracy but also on the number of parameters, floating-point operations (FLOPs), training time, and inference latency.

2. Theoretical Background
2.1 The CIFAR-10 Dataset

CIFAR-10 (Krizhevsky, 2009) is a widely used benchmark for image classification, consisting of 60,000 color images of 32×32 pixels distributed across 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is split into 50,000 training images and 10,000 test images. Its modest resolution and well-defined class structure make it ideal for evaluating and comparing CNN architectures without requiring expensive computational resources.
  
2.2 VGG-Style Architectures

VGGNet, introduced by Simonyan and Zisserman (2014), demonstrated that network depth achieved through stacking multiple 3×3 convolutional layers is a critical component for strong classification performance. The VGG design philosophy favors simplicity and regularity: each convolutional block doubles the number of feature maps while halving spatial dimensions through max-pooling. This approach results in powerful feature extraction but at the cost of a large number of parameters and high FLOPs, making vanilla VGG models computationally expensive for real-time or resource-limited deployment.


2.3 MobileNet and Depthwise Separable Convolutions

Howard et al. (2017) introduced MobileNet as a family of lightweight CNNs designed specifically for mobile and embedded vision applications. The key innovation is the replacement of standard convolutions with depthwise separable convolutions, which factorize a single convolution into two steps: a depthwise convolution that applies a single filter per input channel, followed by a pointwise (1×1) convolution that combines the outputs. This decomposition reduces computation by a factor of approximately 8 to 9 times compared to a standard convolution of equivalent output, while achieving similar representational capacity. Sandler et al. (2018) further advanced this with MobileNetV2, which introduced inverted residuals and linear bottlenecks for improved gradient flow.
 

2.4 Key Efficiency Metrics
Evaluating a CNN architecture requires looking beyond accuracy alone. The following metrics are central to this study:
• Parameters: The total number of learnable weights in the model. More parameters generally imply higher memory requirements.
• FLOPs (Floating-Point Operations): A measure of computational workload, calculated as 2 × MACs (Multiply-Accumulate Operations). Higher FLOPs correspond to slower inference on fixed hardware.
• Training Time: Total wall-clock time to train the model for a fixed number of epochs.
• Inference Latency: Average time to process a single batch of test images, directly reflecting deployment responsiveness.
3. Methodology
3.1 Environment and Setup

All experiments were conducted on Google Colaboratory using an NVIDIA T4 GPU (CUDA 12.8). The implementation used PyTorch 2.10.0 with torchvision for dataset handling. The thop library (version 0.1.1) was used for FLOPs profiling. Python 3.12 was the runtime environment.
3.2 Data Preprocessing and Augmentation

The CIFAR-10 dataset was downloaded via torchvision.datasets.CIFAR10. The training set was subjected to standard data augmentation to improve generalization: random cropping with 4-pixel padding and random horizontal flipping. Both training and test sets were normalised using the CIFAR-10 channel-wise mean (0.4914, 0.4822, 0.4465) and standard deviation (0.2023, 0.1994, 0.2010). A batch size of 128 was used for both training and testing data loaders with 2 worker threads.
3.3 Model 1( VGG-Style CNN)

The VGG-style model consists of three convolutional blocks followed by a fully connected classifier. Each block follows the pattern:


The filter counts double with each block (64 → 128 → 256), and max-pooling progressively reduces the spatial dimensions from 32×32 to 4×4. The classifier applies Dropout(0.5) for regularisation before two fully connected layers (4096→512→10). This design results in 3.25 million parameters and 311.57 million FLOPs.
3.4 Model 2 (MobileNet-Style CNN)
The MobileNet-style model replaces standard convolutions with depthwise separable convolution blocks. Each block performs a depthwise convolution (one filter per channel) followed by a pointwise 1×1 convolution that projects channels into a higher-dimensional space. An initial standard 3×3 convolution projects the input from 3 to 32 channels. Six depthwise separable blocks then progressively expand and down-sample the feature maps (32→64→128→128→256→256→512), using stride-2 in three of the blocks to halve spatial dimensions. A global average pooling layer replaces the large fully connected head, reducing the final representation to a 512-dimensional vector fed into a linear classifier. This design results in only 274,250 parameters (0.27M) and 39.87 million FLOPs approximately 12 times fewer parameters and 8× fewer FLOPs than the VGG-style model.
3.5 Training Configuration
Both models were trained with identical hyperparameters to ensure a fair comparison:
• Optimizer: Adam with an initial learning rate of 0.001
• Learning Rate Schedule: StepLR halved every 5 epochs (gamma = 0.5)
• Loss Function: Cross-Entropy Loss
• Epochs: 15
• Batch Size: 128
FLOPs were measured using thop.profile() on a single 32×32×3 dummy input tensor. Inference time was measured as the average wall-clock time per batch across 50 test batches using torch.no_grad() to disable gradient computation.
4. Experimental Results and Analysis
4.1 Model Size and Computational Cost

Metric	VGG-style CNN	MobileNet-style CNN
Parameters	3.25M	0.27M
FLOPs	311.57M	39.87M
Best Test Accuracy	86.57%	82.25%
Final Test Accuracy	85.86%	82.25%
Total Training Time	324.8 s	329.5 s
Avg Inference Time	5.03 ms	6.67 ms
Table 1: Summary comparison of VGG-style CNN vs. MobileNet-style CNN on CIFAR-10
The VGG-style model is substantially larger than the MobileNet-style model approximately 12 times more parameters (3.25M vs. 0.27M) and 7.8× more FLOPs (311.57M vs. 39.87M). This directly reflects the architectural difference between standard convolutions (VGG) and depthwise separable convolutions (MobileNet). The efficiency comparison is visualised in Figure 4.
 
Fig 4: Computational Efficiency Comparison Parameters (M), FLOPs (M), and Avg Inference Time (ms)
4.2 Training Accuracy and Loss

Both models showed consistent improvement across all 15 epochs. The VGG-style model reached a best test accuracy of 86.57% (epoch 14), while the MobileNet-style model reached 82.25% (epoch 15). The 4.32 percentage point gap in peak test accuracy is notable given the 12 times difference in parameter count.

The VGG model's training accuracy (87.34% at epoch 15) slightly exceeded its test accuracy (85.86%), indicating mild overfitting in the later epochs. The MobileNet model showed more convergent training and test curves, suggesting better generalization behavior for its capacity level.
 
Figure 5: Accuracy and Loss over 15 Epochs  VGG-style CNN (blue) vs. MobileNet-style CNN (red). Solid lines = test; dashed lines = train.
4.3 Epoch-by-Epoch Results

Ep.	VGG Train Acc	VGG Test Acc	VGG Train Loss	VGG Test Loss	Mob Train Acc	Mob Test Acc	Mob Train Loss	Mob Test Loss
1	39.90%	55.19%	1.6267	1.2492	45.35%	56.79%	1.4870	1.2085
2	57.78%	65.14%	1.1753	0.9626	60.04%	64.49%	1.1077	1.0021
3	65.12%	70.22%	0.9873	0.8480	66.03%	67.19%	0.9525	0.9623
4	69.83%	70.46%	0.8711	0.8654	69.95%	72.21%	0.8387	0.7935
5	72.95%	72.01%	0.7872	0.8547	73.37%	73.25%	0.7486	0.7690
6	77.49%	77.89%	0.6543	0.6478	76.80%	77.05%	0.6519	0.6608
7	79.51%	81.94%	0.6056	0.5200	78.66%	77.52%	0.6122	0.6508
8	80.55%	81.54%	0.5759	0.5325	79.50%	78.86%	0.5843	0.6199
9	81.83%	80.52%	0.5386	0.5604	80.53%	79.28%	0.5599	0.6128
10	82.89%	82.58%	0.5092	0.5072	81.30%	80.18%	0.5363	0.5837
11	85.18%	84.84%	0.4448	0.4421	82.57%	80.68%	0.4994	0.5682
12	86.12%	84.59%	0.4205	0.4398	83.21%	81.46%	0.4796	0.5517
13	86.39%	85.43%	0.4094	0.4324	83.61%	81.35%	0.4706	0.5513
14	86.76%	86.57%	0.3925	0.4045	84.04%	81.79%	0.4591	0.5426
15	87.34%	85.86%	0.3787	0.4248	84.13%	82.25%	0.4514	0.5303
Table 2: Full epoch-by-epoch training accuracy, test accuracy, training loss, and test loss for both models
4.4 Training and Inference Time

Despite its much greater computational complexity, the VGG-style model trained in 324.8s compared to 329.5s for MobileNet a negligible difference of less than 2%. This is because both models process similar data volumes per batch and the GPU parallelism of the T4 effectively masked the FLOPs difference at these batch sizes. In inference, the VGG model averaged 5.03ms per batch while MobileNet averaged 6.67ms counter intuitively, the heavier model was faster at inference. This is likely because VGG's larger, regular matrix operations are highly optimised by cuDNN's GEMM routines; whereas MobileNet's depthwise convolutions are less efficiently parallelized on server grade GPUs.
5. Analysis and Discussion
5.1 Accuracy vs. Efficiency Trade-off

The results reveal a nuanced trade-off. With 12 times fewer parameters and 8 times fewer FLOPs, the MobileNet-style model achieves 82.25% test accuracy compared to VGG's 86.57% a 4.32% point gap. Expressed differently, MobileNet achieves approximately 95% of VGG's best accuracy while consuming only 8% of its computational budget. This is a highly favorable efficiency ratio and demonstrates the practical value of depthwise separable convolutions as an architectural strategy.
However, the inference time results challenge the assumption that fewer FLOPs always means faster inference. On a server rade T4 GPU, VGG was marginally faster at inference (5.03ms vs. 6.67ms), highlighting that FLOPs alone are not a reliable proxy for latency. GPUs are optimized for large dense matrix multiplications (GEMM), which standard convolutions exploit efficiently, whereas the grouped nature of depthwise convolutions results in less parallelism. On CPU or mobile processors (where MobileNets were originally designed to run), the FLOPs advantage would translate more directly into speed improvements.
This phenomenon is a known artifact of hardware architecture. NVIDIA GPUs are massively parallel processors heavily optimized for dense matrix multiplications with high arithmetic intensity (the ratio of FLOPs to memory accesses). Standard convolutions (VGG) have high arithmetic intensity and map perfectly to optimized cuDNN kernels. Depthwise convolutions, conversely, are memory-bandwidth bound. They require loading separate channels from memory, applying a small filter, and writing back to memory, resulting in fragmented memory access patterns. Consequently, the GPU spends more time moving data than performing calculations, rendering the theoretical FLOP reduction irrelevant for empirical speed on this specific hardware.
5.2 Learning Dynamics

As shown in Fig 5, the learning curves reveal important differences in convergence behavior. The VGG model made a large jump at epochs 6–7 (test accuracy from 72.01% to 81.94%), likely coinciding with the first learning rate reduction at epoch 5. After this, it continued improving but at a slower rate, showing mild divergence between training and test accuracy in later epochs. The MobileNet model improved more gradually and steadily, maintaining a tighter gap between training and test accuracy evidence of better regularization through its global average pooling design, which eliminates the large parameter-heavy fully connected head that is a known source of overfitting.
6. Research Outlook
The findings of this project highlight that algorithmic efficiency (low FLOPs) does not universally guarantee hardware efficiency (low latency). To deploy deep learning models in highly constrained, real-time edge environments such as UAV-based ground object detection or mobile brain-data science applications future research must bridge the gap between mathematical design and silicon execution.
Proposed Direction: Hardware-Aware Neural Architecture Search (HW-NAS) for Edge UAVs
Future work should focus on HW-NAS frameworks that incorporate specific hardware constraints (e.g., latency, energy consumption, and memory hierarchy of edge TPUs or FPGAs (Field-Programmable Gate Array)) directly into the loss function during the architecture search phase. Instead of relying on proxy metrics like FLOPs, the search space should be guided by empirical latency measurements from the target hardware. Furthermore, integrating Mechanistic Interpretability tools and Probabilistic Graphical Models (PGMs) to understand which specific channels in a depthwise layer contribute most to accurate predictions could allow for targeted, structured pruning. Combining HW-NAS with 8-bit Integer (INT8) quantization would yield robust models capable of high-fidelity; real-time inference specifically optimized for the constraints of airborne or mobile edge systems.
7. Conclusion

This project implemented and compared a VGG-style CNN and a MobileNet-style CNN on the CIFAR-10 dataset. The VGG-style model achieved a best test accuracy of 86.57% with 3.25M parameters and 311.57M FLOPs, while the MobileNet-style model achieved 82.25% with only 0.27M parameters and 39.87M FLOPs representing approximately 95% of VGG's accuracy at roughly 8% of its computational cost. Training times were comparable (324.8s vs. 329.5s), while inference time unexpectedly favored the larger VGG model (5.03ms vs. 6.67ms), revealing that FLOPs alone are insufficient as a proxy for practical deployment speed on GPU hardware.
As shown in Figure 5, both models demonstrated consistent improvement across 15 epochs, with the VGG model converging faster after the first learning rate decay and MobileNet displaying more stable generalization. The efficiency comparison in Figure 4 visually confirms the dramatic difference in model size and computational cost between the two architectures.
These findings confirm that depthwise separable convolutions offer a highly favorable accuracy-efficiency trade-off, but that the translation of theoretical efficiency gains (FLOPs) into practical efficiency (latency) is hardware-dependent. This motivates the proposed research direction of hardware-aware NAS frameworks that directly optimize for device level latency across heterogeneous edge targets. Ultimately, designing efficient architectures requires a paradigm shift from pure mathematical factorization to holistic, hardware-aware co-optimization; ensuring models are tailored to the precise execution constraints of their target deployment environment.

8. References
1.	MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications(Andrew G. Howard and others)
2.	MobileNetV2: Inverted Residuals and Linear Bottlenecks(Mark Sandler and others)
3.	Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In Proceedings of the International Conference on Machine Learning (ICML), 6105–6114.
