---
layout: default
title: "Robot-Barista"
---
<div align="center">
  
# Uncertainty-Aware Opportunistic Hybrid Language Model in Wireless Robotic Systems

</div>

**Abstract**

The hybrid language model (HLM) is an emerg-
ing architecture that efficiently distributes com-
putation between on-device small language mod-
els (SLMs) and remote large language models
(LLMs). In HLM, an SLM drafts tokens and its
paired LLM validates and refines them, thereby
achieving higher token throughput than LLMs
and higher inference accuracy than SLMs. Re-
cently, the uncertainty-aware opportunistic HLM
has been proposed to improve communication and
computation efficiency by skipping LLM verifica-
tion when the SLM’s uncertainty is low. However,
this approach has only been evaluated on sim-
ple text prediction tasks under a statistical chan-
nel model for theoretical analysis. To validate
the practical feasibility of U-HLM, in this pa-
per, we implement U-HLM on a real-world robot
testbed, where an industrial-grade robotic ma-
nipulator (high-precision robot arm with gripper)
runs an SLM and communicates with a remote
LLM over Wi-Fi. In this experimental setup, we
observe that computing uncertainty itself incurs
non-negligible latency. To mitigate this, we pro-
pose a conditional uncertainty calculation omis-
sion method, which skips the uncertainty calcula-
tion when a lightweight logistic regression model
predicts the uncertainty to be sufficiently low. Ex-
perimental results show that, compared to HLM,
the proposed U-HLM improves token through-
put by 24.9% and 41.8% under strong and weak
Wi-Fi coverage conditions, respectively, while
maintaining a 98.11% F1 score

## Experiments
- 실험1 요약
- 실험2 요약

[Paper](링크) · [Code](https://github.com/jeyoung78/Robot-Barista)
