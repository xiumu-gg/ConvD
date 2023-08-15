# ConvD：Attention Enhanced Dynamic Convolutional Embeddings for Knowledge Graph Completion
This is an implementation of ConvD from the paper "ConvD".

[//]: # (Paper: )

## Requirements

Python is running at version <u>3.9.16</u>. Other Python package versions can be found in **requirements.txt**

It is recommended to create a virtual environment with the above version of Python using **conda**, and install the python packages in requirements.txt using **pip** in the virtual environment.



## Running a model

Parameters are configured in `configs/ConvD.json`, all the hyperparameters in the configuration file come from the paper.

Start training command:
```
$ python main.py -c configs/ConvD.json
```

## Citation
