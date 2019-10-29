# Conv/Deconv Autoencoder

This repository contains all the codes for the Autoencoder question in HW3 of CE-40959: Deep Learning Course, 
presented in Sharif University of Technology. In this question, a complete set of Jupyter Notebook and python scripts 
is prepared for implementing a Conv/Deconv Autoencoder model applied on images.

# Usage

In the jupyter notebook, we are going to work on [farsi OCR dataset](http://farsiocr.ir/%d9%85%d8%ac%d9%85%d9%88%d8%b9%d9%87-%d8%af%d8%a7%d8%af%d9%87/%d9%85%d8%ac%d9%85%d9%88%d8%b9%d9%87-%d8%a7%d8%b1%d9%82%d8%a7%d9%85-%d8%af%d8%b3%d8%aa%d9%86%d9%88%db%8c%d8%b3-%d9%87%d8%af%db%8c/). As its name implies, it is like famous MNIST dataset but it consists of images of handwritten digits in farsi. 

The problem we define for this dataset is to reconstruct original image after making some random rotations. We want to develop a model which recieves as input a rotated image and outputs its original without rotation. Meanwhile, a latent embedding is learned in the training process which its quality will be examined later.

Alongside the notebook, there some python files with TODO sections filled with proper lines of code. For Each TODO section, 
a comprehensive description of the required code is provided.
