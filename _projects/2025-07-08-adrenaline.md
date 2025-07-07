---
layout: page
title: Adrenaline
description: Accelerating LLM Inference in Prefill-Decoding Disaggregation with Attention Offloading
img: assets/img/12.jpg
importance: 1
category: work
related_publications: true
---

{% include repository/repo.liquid repository='ASISys/Adrenaline' %}

## TL;DR

Large Language Model (LLM) inference is a two-stage process: a compute-intensive **prefill stage** and a memory-intensive **decoding stage**. To prevent performance interference between these two distinct phases, current LLM serving systems often adopt a **prefill-decoding (PD) disaggregation** deployment strategy, where these stages are executed on different machines. However, our observations reveal that PD disaggregation leads to significant GPU resource waste. Specifically, compute-intensive prefill instances suffer from low memory utilization, while memory-intensive decoding instances experience low compute utilization.

To address this challenge, we introduce **Adrenaline** {% cite Liang2025InjectingAI %}, an **a**ttention **d**isagg**re**gatio**n** **a**nd off**l**oad**in**g m**e**chanism designed to boost resource utilization and performance in LLM serving systems. Adrenaline's core idea is to separate a portion of the decoding stage's attention computation and offload it to prefill instances. Given the memory-intensive nature of decoding attention, Adrenaline enhances the memory capacity and bandwidth utilization of prefill instances. Simultaneously, it increases the batch size for decoding, thereby improving the compute utilization of decoding instances. Collectively, Adrenaline significantly boosts end-to-end inference throughput.

Adrenaline achieves this through three key techniques: load-aware offloading scheduling, low-latency decoding synchronization, and resource-efficient prefill colocation. Our experimental results demonstrate that, compared to state-of-the-art PD disaggregation systems, Adrenaline improves prefill instance memory capacity utilization by up to **2.3x** and memory bandwidth utilization by up to **2.07x**. For decoding instance, Adrenaline increases compute utilization by up to **1.67x** and throughput by up to **1.68x**.

---

## 1. Background: PD Disaggregation

In LLM inference systems, the execution of each request involves two sequential stages. The **prefill stage** computes all prompt tokens in parallel to generate the KV cache and the first token. The **decoding stage** then iteratively outputs subsequent tokens based on the previously generated KV cache.

The prefill stage, by processing many tokens in parallel, is typically **compute-intensive**, and its latency is measured by the **Time To First Token (TTFT)**. In contrast, the decoding stage becomes **memory-intensive** due to frequently loading the ever-growing KV cache and low arithmetic intensity, with its latency measured by the **Time Per Output Token (TPOT)**.

Since the prefill step usually incurs higher latency than the decoding step, running both stages on the same GPU can lead to significant interference. This interference increases the TTFT for the prefill stage and the TPOT for the decoding stage within the same batch. To mitigate this interference, LLM serving systems commonly employ a PD disaggragation deployment strategy. By assigning the prefill and decoding stages to different GPUs, PD disaggregation eliminates interference between these two stages, allowing each to independently meet its **Service Level Objectives (SLOs)**. Furthermore, separating the prefill and decoding stages into distinct GPU pools enables automatic and flexible resource scaling to accommodate the differing resource demands of each stage and dynamic real-world workloads.

---

## 2. The Bottlenecks of PD Disaggregation

Despite flexibility, we have observed that PD disaggregation leads to severe GPU resource waste in LLM serving systems. Specifically, GPUs running the compute-intensive prefill stage often experience low HBM (High Bandwidth Memory) capacity and bandwidth utilization. Conversely, the memory-intensive decoding stage faces issues with low compute resource utilization.

We evaluated the resource utilization in PD disaggregation with Llama-2 7B on A100 using ShareGPT workload. As shown in Figure 1a and 1c, our results indicate that the HBM bandwidth utilization in the prefill instance is below 25%, and HBM capacity utilization is below 20%. In addition, as depicted in Figure 1b, compute utilization in the decoding instance is below 26%. Given the high cost of GPUs, this low resource utilization directly translates to increased inference costs.

<div class="row justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/2025-07-08-adrenaline/fig1-resource-util.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Figure 1: Compute, memory bandwidth, and memory capacity utilization of prefill and decoding stages.
</div>

To investigate the reasons behind this underutilization, we measured the resource consumption of different kernels within the prefill and decoding stages. As illustrated in Figure 2a, the four main kernels in the prefill stage — including QKV Projection, Attention, O Projection, and feed-forward network (FFN) — are all compute-intensive, leading to insufficient memory bandwidth utilization. Additionally, the batch size for the prefill stage is typically kept small to reduce TTFT, which limits the KV cache size, resulting in low memory capacity utilization. In the decoding stage, however, the attention kernel dominates HBM capacity and bandwidth consumption. The high memory resource consumption of attention primarily stems from the storage and access of the KV cache.

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/2025-07-08-adrenaline/fig2-kernel-util.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Figure 2: Compute and HBM bandwidth utilization of various kernels in prefill and decoding stages.
</div>

---

## 3. Adrenaline: Our Solution

To address the aforementioned issues, we propose **Adrenaline**, an attention disaggregation and offloading mechanism designed to enhance resource utilization and performance in LLM serving systems. The core idea behind Adrenaline is to separate a portion of the decoding stage's attention computation tasks and offload them to prefill instances.

By offloading these memory-intensive attention computation tasks, Adrenaline significantly improves the HBM capacity and bandwidth utilization of prefill instances. Furthermore, this offloading allows for an increased total batch size in decoding instances, thereby boosting their compute resource utilization.

Figure 3 illustrates the difference between traditional PD disaggregation and Adrenaline. In the PD disaggregation scheme shown in Figure 3a, the decoding batch size is limited to $$ M $$ due to HBM bandwidth and capacity constraints on the decoding instance. In Adrenaline, depicted in Figure 3b, we offload $$ N $$ attention computation requests to the prefill instance for execution. This increases the total decoding batch size from $$ M $$ to $$ M+N $$, significantly boosting the overall throughput of the inference system.


<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/2025-07-08-adrenaline/fig3-architecture-comparison.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Figure 3: Comparison of traditional PD disaggregation and Adrenaline.
</div>

---

## 4. Adrenaline System Design

The system architecture of Adrenaline is depicted in Figure 4. Adrenaline consists of three main modules: the proxy, prefill instances, and decoding instances. The proxy module routes prefill or decoding computation tasks to the appropriate prefill and decoding instances. Prefill and decoding instances are used to execute prefill and decoding computation tasks, respectively. However, unlike existing designs, Adrenaline separates and offloads partial of decoding attention computation tasks, assigning them to a remote **Attention Executor** located on the prefill instance. The attention executor is specifically designed to leverage the underutilized GPU memory resources within the prefill instance to execute offloaded decoding attention computation tasks.

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/2025-07-08-adrenaline/fig4-arch-overview.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Figure 4: Adrenaline system architecture.
</div>

To maximize system performance under SLOs, Adrenaline employs the following three key techniques within the LLM inference workflow:

First, in the decoding stage, Adrenaline introduces a **Low-latency Decoding Synchronization** mechanism to minimize the synchronization overhead between the remote attention executor and the local decoding engine.

Second, through **Resource-efficient Prefill Colocation** design, Adrenaline improves GPU memory utilization in prefill instances while ensuring sufficient compute resources for prefill computation. This eliminates performance interference and allows the system to meet the required SLOs.

Third, Adrenaline utilizes an **Load-aware Offloading Scheduling** strategy to adaptively determine if a decoding attention needs to be offloaded. The proxy monitors the GPU resource and system load, determining the allocation of attention tasks.

For more design details, please refer to our paper {% cite Liang2025InjectingAI %}.

---

## 5. Performance Evaluation

We implemented a system prototype of Adrenaline based on vLLM (The source code of Adrenaline is available at [GitHub](https://github.com/ASISys/Adrenaline) for public use). Below, we compare the end-to-end performance and resource utilization of vLLM with Adrenaline.

### 5.1 End-to-End Performance Comparison

<div class="row justify-content-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/2025-07-08-adrenaline/fig5-e2e-results.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Figure 5: End-to-end performance comparison of vLLM and Adrenaline running Llama-2 13B in 1P1D configuration (ShareGPT workload).
</div>

**TTFT:** We measured the TTFT of the Llama 2-13B model at different request rates, as shown in Figure 5a. At lower request rates (< 1.5), Adrenaline's TTFT is close to vLLM's. However, as the request rate increases, vLLM's decoding instance exhausts HBM resources, blocking new decoding requests and significantly increasing the queuing time, which contributes to the request's TTFT. When the request rate increases to 3.5, vLLM's TTFT is 5x higher than that of Adrenaline. By offloading some decoding attention tasks, Adrenaline increases the maximum batch size of the decoding stage, thereby reducing the queuing time of requests and ultimately lowering TTFT.

**TPOT:** As shown in Figure 5b and 5c, Adrenaline's TPOT is generally close to vLLM's at the same request rates.

**Decoding Throughput:** As shown in Figure 5d, at low request rates, the throughput of both schemes is similar. As the request rate increases, vLLM's throughput plateaus due to HBM space and bandwidth limitations in the decoding instance. By utilizing the prefill instance's HBM resources to offload decoding attention tasks, Adrenaline achieves a **1.63x** improvement in decoding throughput.

### 5.2 Resource Utilization Comparison


<div class="row justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/2025-07-08-adrenaline/fig6-resource-util-results.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
      Figure 6: Comparison of memory bandwidth & capacity utilization in prefill stage and compute utilization in decoding stage for vLLM and Adrenaline.
</div>

**HBM Bandwidth Utilization:** As shown in Figure 6a, by offloading attention computation tasks, Adrenaline significantly improves HBM bandwidth utilization in prefill instances. Compared to the original vLLM scheme, Adrenaline achieves a **1.49-2.07x** HBM bandwidth utilization improvement in the prefill instance in the Llama-2 7B model. The results using the Llama-2 13B model also show a similar trend (1.37-1.93x improvement compared to vLLM).

**Compute Utilization:** As shown in Figure 6b, by disaggregating and offloading attention computation tasks, Adrenaline significantly increases the decoding batch size, achieving a **1.67x** improvement in compute utilization compared to vLLM.

**HBM Capacity Utilization:** As shown in Figure 6c, after loading model weights, the HBM capacity utilization of prefill instances in vLLM remains around 20%, with the remaining 80% of HBM capacity unused. Adrenaline's attention executor utilizes the idle HBM of prefill instances to store the KV cache for offloaded attention. Therefore, after the warm-up phase, Adrenaline achieves a **2.3x** HBM utilization improvement in prefill instances compared to the default PD disaggregation scheme.

---

## Acknowledgments

This project is built upon [vLLM](https://github.com/vllm-project/vllm). We really appreciate open-source software, models, and workloads used in Adrenaline.

## Citation

The source code is available in our [Adrenaline project](https://github.com/ASISys/Adrenaline). If you find this project useful for your research, please consider citing our  paper {% cite Liang2025InjectingAI %}:

```bibtex
@misc{liang2025injectingadrenalinellmserving,
      title={Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation}, 
      author={Yunkai Liang and Zhangyu Chen and Pengfei Zuo and Zhi Zhou and Xu Chen and Zhou Yu},
      year={2025},
      eprint={2503.20552},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2503.20552}, 
}
```