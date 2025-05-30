<h1 style="text-align: center;">Uncertainty-Aware Opportunistic Hybrid Language Model <br> in Wireless Robotic Systems</h1>

<p align="center" style="font-size: 0.9em; color: #555; margin-top: 0.2em;">
  Jeyoung Park<sup>1</sup>, Yeonsub Lim<sup>2</sup>, Seungeun Oh<sup>2</sup>,
  Jihong Park<sup>3</sup>, Jinho Choi<sup>4</sup>, Seong-Lyun Kim<sup>2</sup>
</p>

<p align="center" style="font-size: 0.9em; color: #555; margin-top: 0.2em;">
  University of Waterloo<sup>1</sup>, Yonsei University<sup>2</sup>, Singapore University of Technology and Design<sup>3</sup>, University of Adelaide<sup>4</sup>
</p>
<p align="center" style="margin: 1em 0;">
  <a href="https://arxiv.org/abs/2407.02666"
     style="
       display: inline-block;
       padding: 0.6em 1.2em;
       background-color: #24292e;
       color: #fff;
       border-radius: 9999px;
       text-decoration: none;
       font-weight: bold;
       margin-right: 0.5em;
     ">
    📄 Paper
  </a>
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
The hybrid language model (HLM) is an emerging architecture that efficiently distributes computation between on-device small language models (SLMs) and remote large language models(LLMs). In HLM, an SLM drafts tokens and its paired LLM validates and refines them, thereby achieving higher token throughput than LLMs and higher inference accuracy than SLMs. Recently, the uncertainty-aware opportunistic HLM has been proposed to improve communication and computation efficiency by skipping LLM verification when the SLM’s uncertainty is low. However, this approach has only been evaluated on simple text prediction tasks under a statistical channel model for theoretical analysis. To validate the practical feasibility of U-HLM, in this paper, we implement U-HLM on a real-world robot testbed, where an industrial-grade robotic manipulator (high-precision robot arm with gripper) runs an SLM and communicates with a remote LLM over Wi-Fi. In this experimental setup, we observe that computing uncertainty itself incurs non-negligible latency. To mitigate this, we propose a conditional uncertainty calculation omission method, which skips the uncertainty calculation when a lightweight logistic regression model predicts the uncertainty to be sufficiently low. Experimental results show that, compared to HLM, the proposed U-HLM improves token throughput by 24.9% and 41.8% under strong and weak Wi-Fi coverage conditions, respectively, while maintaining a 98.11% F1 score.
</div>

<h2 style="text-align: center; font-size: 1.5em; margin-top: 2em;">
Hybrid Language Model & Robot Testbed
</h2>

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

<p align="center" style="font-size: 1.1em; color: #555; margin-top: 0.2em;">
  Experimental Setup
</p>

<div align="justify" style="max-width: 900px; margin: 0 auto;">
In this proof-of-concept study to verify U-HLM’s effectiveness on an actual wireless network and as a robotic task planner, we implement a testbed consisting of the following three main components: a laptop (local device), a remote server, a the robot, connected over a wireless network, as shown in Figure. U-HLM deployed on the testbed serves as a task planner, generating sequences of action-object pairs corresponding to given natural language orders to be performed by the robot.
</div>

