# Name

**Sara Spasojevic**

# Group Members

Individual submission

# Project Information:

**Class**: CS 5330 - Computer Vision, Spring 2025
**Project**: Project 3
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




