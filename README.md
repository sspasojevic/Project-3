# Name

**Sara Spasojevic**

# Description

The goal of this project is real-time 2D object recognition. Given a white surface as a
background, and objects placed on top of it, with the camera positioned above, the program
detects objects and does classification to assign labels to them. It does this by performing a
series of steps - thresholding, morphological filtering, and segmenting. First three are implemented 
from scratch, and segmenting uses OpenCV's functions. The regions are then
analyzed to compute a feature vector for each, and based on those, they are classified with
either known labels or a unique unknown label. The program can perform either nearest
neighbor (NN) or k-NN (k-nearest neighbors) classification, and this is specified when
running the program in the terminal. In addition, the program is limited to recognizing only 3 closest
objects to the center of the frame (based on their centroids). The program uses the iPhone’s back
camera (continuity feature) for real-time video feed.

The program has two training (labeling) modes:

• **Normal training mode** - allows users to label all 3 objects in the frame.

• **Automatic training mode** - allows labeling of only the unknown object in the
frame.

# Project Information:

**Class**: CS 5330 - Computer Vision, Spring 2025
**Date**: Feb 20, 2025

# Operating system and IDE

**OS**: Mac 15.2
**IDE**: Visual Studio Code

Code was compiled using CMAKE (.txt file is provided in the submission for reference).

# Time Travel Days

None days used

---

# Links

Video demonstration:

https://drive.google.com/file/d/1Iyhhwd-Bn4Pm0D12C9UOj04UVogZ9gf4/view?usp=sharing


# Resources

No external files needed for running. The program will create its own CSV files 
if labeling is done.


# Programs Created using CMake

- ObjectRecognition

---

# Build Instructions

## Step 1: Install Required Dependencies

Make sure you have **CMAKE** and **OpenCV** installed on the machine.
Also make sure to have a CMakeLists.txt set up, and make sure it references the correct paths to files if folder structure is modified.

## Step 2: Build Directory

Create a build directory in your folder.

## Step 3: Run CMake and build

Run CMake to generate build files using command: cmake ..
Build the programs using the command: make

## Step 4: Run the programs (overview of supported methods and distance metrics)

**create_csv.cpp**
The program takes the following input in terminal:
- <classification_method>: Method to use: "nn" | "knn"

Example: ./CreateCSV knn




