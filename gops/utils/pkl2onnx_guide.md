# The "pkl2onnx" file aims to convert the trained .pkl network format into the .onnx format, making it convenient for calling from C++ code.

Copyright © 2022 Intelligent Driving Laboratory (iDLab). All rights reserved.

## Installation
pkl2onnx requires:
1. install gops package.
2. have access to the GOPS and IDSim code and the ability to train models.
2. pip install onnxruntime in conda env.
3. pip install onnx in conda env.

## User guide
1. modify the path of the network you want to convert in the "log_policy_dir".
2. modify the specific network, i.e., the network saved after how many iterations, such as '10000' or '1000_opt' in "log_path".
3. select the appropriate function for converting the desired network. 
a. If it's a method from the 'fhadp' class, only the state dimension needs modification; choose functions related to 'fhadp/fhadp2' for network conversion. 
b. If it's a method from the 'DSAC' class, both the state and action dimensions need modification; choose 'dsac' related functions for network conversion. 
The function call format is consistent: (networks, input_dim/obs_dim(obs_dim+act_dim), policy_dir), where they respectively represent the saved network, network input dimension, network save path, and name.
4. load the ONNX and original networks for output comparison (optional). Initialize the network input randomly and ensure consistency between the networks before and after conversion by verifying the outputs. The required function calls can be based on the comments or remarks provided within the 'pkl2onnx' documentation. These might include specific function names or descriptions indicating the steps or methods to follow for network conversion.
5. Common issue: 
a. In torch.onnx.export, the first dimension represents the specific network to be converted, such as policy network or value network. Formats like network.policy or network.value should align with the terminology used in the algorithm, such as 'dsac-t' having network.q1 and network.q2.
b. The dimensions of the network input are not aligned.
c. The loading or saving paths might have issues. Please verify if the relative paths are correct or consider using absolute paths.
