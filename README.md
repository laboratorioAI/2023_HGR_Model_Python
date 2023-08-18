# CodigoTesis
# Hand Gesture Recognition Applied to the Interaction with Video Games
---
## Introduction
This document lists the steps needed to create the CNN-LSTM model and Evaluate the result in: . 

### Description
Hand Gesture Recognition system development used CRISP-ML as a process model for the machine learning application lifecycle. The table shown below contains all the project files related to activities performed in the lifecycle of the HGR system.   

| **Activity**                                                        | **Branch**      | **File**                                |
|---------------------------------------------------------------------|-----------------|-----------------------------------------|
| Data preparation (sEMG spectrogram generation)                      | StaticModel     | spectogramDatasetGeneration.py          |
| Data preparation (quaternion spectrogram generation)                | DynamicModel    | spectogramDatasetGenerationQuat.py      |
| Individual modeling and evaluation (static gesture model)           | StaticModel     | staticModel.py                          |
| Individual modeling and evaluation (dynamic gesture model)          | DynamicModel    | dynamicModel.py                         |
| Individual modeling and evaluation (switch classifier)              | SwitchModel     | switchModel.py                          |
| Evaluation App (generation of JSON )                                | EvaluationApp   | evaluationApp.py                        |
