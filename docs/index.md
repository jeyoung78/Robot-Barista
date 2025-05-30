<h1 style="text-align: center;">Uncertainty-Aware Opportunistic Hybrid Language Model <br> in Wireless Robotic Systems</h1>

<p align="center" style="font-size: 0.9em; color: #555; margin-top: 0.2em;">
  Jeyoung Park<sup>1</sup>, Yeonsub Lim<sup>2</sup>, Seungeun Oh<sup>2</sup>,
  Jihong Park<sup>3</sup>, Jinho Choi<sup>4</sup>, Seong-Lyun Kim<sup>2</sup>
</p>

<p align="center" style="font-size: 0.9em; color: #555; margin-top: 0.2em;">
  University of Waterloo<sup>1</sup>, Yonsei University<sup>2</sup>, Singapore University of Technology and Design<sup>3</sup>, University of Adelaide<sup>4</sup>
</p>
<p align="center" style="margin: 1em 0;">
  <a href="https://github.com/jeyoung78/Robot-Barista"
     style="
       display: inline-block;
       padding: 0.6em 1.2em;
       background-color: #24292e;
       color: #fff;
       border-radius: 9999px;
       text-decoration: none;
       font-weight: bold;
     ">
    💻 Code
  </a>
</p>
<h2 style="text-align: center; font-size: 1.5em; margin-top: 2em;">
Abstract
</h2>

<img src="Robot_str.png" 
    style="
    display: block;
    margin-top: 0em;
    margin-bottom: 0em;
    max-width: 100%;
    height: auto;
  "
/>

<div align="justify" style="max-width: 900px; margin: 0 auto;">
  The hybrid language model (HLM) is an emerging architecture that efficiently distributes computation between on-device small language models (SLMs) and remote large language models (LLMs). In HLM, an SLM drafts tokens and its paired LLM validates and refines them, thereby achieving higher token throughput than LLMs and higher inference accuracy than SLMs. Recently, the uncertainty-aware opportunistic HLM has been proposed to improve communication and computation efficiency by skipping LLM verification when the SLM’s uncertainty is low. However, this approach has only been evaluated on simple text prediction tasks under a statistical channel model for theoretical analysis. To validate the practical feasibility of U-HLM, in this paper, we implement U-HLM on a real-world robot testbed, where an industrial-grade robotic manipulator (high-precision robot arm with gripper) runs an SLM and communicates with a remote LLM over Wi-Fi. In this experimental setup, we observe that computing uncertainty itself incurs non-negligible latency. To mitigate this, we propose a conditional uncertainty calculation omission method, which skips the uncertainty calculation when a lightweight logistic regression model predicts the uncertainty to be sufficiently low. Experimental results show that, compared to HLM, the proposed U-HLM improves token throughput by 24.9% and 41.8% under strong and weak Wi-Fi coverage conditions, respectively, while maintaining a 98.11% F1 score. <br><br>
</div>

<h2 style="text-align: center; font-size: 1.5em; margin-top: 2em;">
Testbed Implementation for U-HLM
</h2>

<div align="justify" style="max-width: 900px; margin: 0 auto;">
The Uncertainty-Aware Opportunistic Hybrid Language Model (U-HLM) has been proposed as a practical framework that not only reduces the computational overhead of LLMs by leveraging both an on-device small language model (SLM) and a remote LLM, but also improves token throughput—by enhancing overall communication and computation efficiency.
<br><br>
U-HLM leverages uncertainty—the model’s self-assessed confidence in its outputs—to decide whether uplink communication is necessary, enabling the system to skip transmitting the full vocabulary distribution and avoid remote LLM computation for verification and resampling when uncertainty is low.
<br><br>
These characteristics make U-HLM a feasible way to improve latency and reduce computational load in LLM-integrated robotic systems.
</div>

<img src="Robot_env.png" 
    style="
    display: block;
    margin-top: 0em;
    margin-bottom: 0em;
    max-width: 100%;
    height: auto;
  "
/>

<div align="justify" style="max-width: 900px; margin: 0 auto;">
In this proof-of-concept study to verify U-HLM’s effectiveness on an actual wireless network and as a robotic task planner, we implement a testbed consisting of the following three main components: a laptop (local device), a remote server, a the robot, connected over a wireless network, as shown in Figure. U-HLM deployed on the testbed serves as a task planner, generating sequences of action-object pairs corresponding to given natural language orders to be performed by the robot.
</div>


<p align="center"
   style="font-size: 1.1em; color: #555; margin-top: 1.5em; font-weight: bold;">
  <br> Experimental Setup
</p>

+ <div align="justify" style="max-width: 900px; margin: 0 auto;">
Local SLM : Tiny-Llama 1.1B on a laptop (6-core Intel Core i7-10750H CPU, 8 GB of DDR4 RAM, and an NVIDIA GeForce GTX 1650 Ti GPU connected to IEEE 802.11ac Wi-Fi on a 5 GHz band)
+ <div align="justify" style="max-width: 900px; margin: 0 auto;">
LLM : Llama 27B on a Linux server (8-core Intel Xeon Silver 4215R CPU, 64 GB of DDR4 RAM, and three NVIDIA GeForce RTX 3090 GPUs, connected to Ethernet)
+ <div align="justify" style="max-width: 900px; margin: 0 auto;">
Robot : Doosan A0912s with robot arm (GEP2016IO-00-A gripper), connected same Wi-Fi with a laptop.

<div style="
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  max-width: 800px;
  margin: 0 auto;
">

  <!-- 왼쪽 블록 -->
  <div style="flex: 0 0 48%; text-align: center; margin-bottom: 1em;">
    <img
      src="robot1.jpg"
      style="width: 100%; border-radius: 8px;"
    />
    <p style="margin-top: 0.5em; font-size: 0.95em; line-height: 1.4;">
      Experimental Workspace
    </p>
  </div>

  <!-- 오른쪽 블록 -->
  <div style="flex: 0 0 48%; text-align: center; margin-bottom: 1em;">
    <img
      src="robot2.jpg"
      style="width: 100%; border-radius: 8px;"
    />
    <p style="margin-top: 0.5em; font-size: 0.95em; line-height: 1.4;">
Robot Pouring ingredient To Cup
    </p>
  </div>

</div>



<h3 style="text-align: center; font-size: 1.5em; margin-top: 2em;">
Results
</h3>

<p align="center">
  <iframe
    width="800" height="450"
    src="https://www.youtube.com/embed/Yp4QAQ76CIc"
    frameborder="0"
    allowfullscreen
    style="max-width:100%;"
  ></iframe>
</p>
