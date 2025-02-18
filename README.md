# Robot-Barista
Model capable of processing abstract human language command into list of actions then low level robot commands. Motivated and built upon Google Deepmind's SayCan and Robotics Transformer 2. 

# Scoring Language Model

# Robotics Transformer
The approach DeepMind takes is to directly train vision-language models designed for open-vocabulary visual question answering and visual dialogue to output low-level robot actions. Typical vision language models are trained to produce natural language tokens, they can be trained on robot trajectories by tokeninzing the actions into text tokens and creating multimodal sentences that respond to robotic instructions paired with camera observation by producing correponsidng actions. 

The action space consists of 6-DoF positional and rotational displacement of the robot end effector, as well as the level of extension of the robot gripper and a special discrete command for terminating the episode, which should be triggered by the policy to signal successful completion. The continuous dimensions, with the exception of the discrete termination command, are discretized into 256 uniform bins, which enables the robot actions to be represented using ordinals of the discrete bins as 8 integer numbers. 

Even though the Deepmind team used PaLI-X and PaLM-E, the two models are not opensourced and thus cannot be used. Therefore, DeepSeek VL2 tiny was chosen to be used in this work. DeepSeek VL2 consists of three main components: 1. A vision encoder that uses a dynamic tiling strategy to split high-resolution images into manageable patches. The encoder is based on SigLIP-SO400M-384 and processes each tile to generate embeddings. 2. A vision language adaptor that compresses and rearranges these visual tokens using operations like 2 x 2 pixel shuffle and the insertion of special tokens to alighn them with the text tokens. 3. A Mixture-of-Experts (MoE) language model that incorporates Multi-head Latent Attention to efficiently fuse the visual and textual information based on a decoder-only LLaVA style architecture. 
