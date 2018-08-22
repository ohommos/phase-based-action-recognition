This is the code release for the paper titled "Using phase instead of optical flow for action recognition", to be published under the EECV 2018 workshop "What is Optical Flow For?".
Paper text and results are excerpt from my MSc thesis titled "Learning Phase-Based Descriptions for Action Recognition". Link: http://resolver.tudelft.nl/uuid:40a08f3b-5af7-4281-bf0c-9e7e57da6f52

Code ran on both Linux and Windows OS, with the following package versions:
Python 3.5.2
Numpy 1.14.2
Tensorflow GPU 1.7
Ospencv-python 3.3

Usage:
- Dataset.py shows the function of how phase images were calculated
- Model.py shows the implementation of the PO layers:
-- learnable_po_conv2d_layer: trian a PO layer from randomly initiaized weights
-- finetunable_po_conv2d_layer: generate PO layer from trained weights, and finetune them as needed.