# Graduation Thesis

The repository for a part of our graduation thesis at University of Information Technology (UIT-VNUHCM):

<p align="center"> 
**"BLACK-BOX SPARSE ADVERSARIAL ATTACK ON COMPUTER VISION MODELS\
USING EVOLUTIONARY COMPUTATION."** 
</p>

## General Information

- **Students:**
    - Le Chi Cuong
    - Phan Truong Tri

- **Instructor:** Dr. Luong Ngoc Hoang

- **Keywords:** Adversarial Attack, Genetic Algorithm, Computer Vision

- **Grade:** 9/10

## Abstract

- We employed **Genetic Algorithm (GA)** to generate sparse pertubations to alter the input images. These perturbed images could cause the model to predict incorrect results.
- **Black-box Attack:** we conducted the attacks under the constraint that only the input images and output results were accessable, with no internal information about the models (e.g., the models' weights, gradients,...) being exposed.
- **Sparse attack:** the perturbations only modify a small amount of pixels in the input images.

## Example

Some successful attacks on object detection model (YOLOv8)

![alternative text](gif/demo.gif)


