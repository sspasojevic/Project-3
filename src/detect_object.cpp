/**
 * Sara Spasojevic
 * CS5330, Spring 2025
 * Project 
 * Feb 20, 2025
 *
 * The program connects to the iPhone back camera (connected over the same iCloud login/connectivity feature) and 
 * detects and recognizes top 3 objects closest to the center of the frame. Objects must be layed on a white background.
 * The program allows for labeling the objects in the training mode, and then automatically classifies them once it has 
 * sufficient data.
 * 
 * The program takes one argument in terminal, telling it which classifying method to use - 'nn' or 'knn'.
 * nn method uses nn.csv, and if not present, you can use the labeling modes to create it.
 * knn method uses knn.csv and if the file is not present, you can use the labeling modes to create it.
 *
 * 
 * Keypresses:
 * 'n' - normal labeling mode - labels all topN objects in the frame, regardless if they are recognized or not
 * 'a' - automatic labeling mode - labels only the objects labeled as unknown in the frame (Extension 3)
 * 'q' - quit the program
 */

// Imports
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "helpers.hpp"


int main(int argc, char* argv[]) {

    // Check if an argument was provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <classification_method> \n";
        return 1;
    }

    // Get the classification type 
    std::string classificationType = argv[1];
    std::string csvName; // Initialize csvName

    // If classification type is nn, use the nn CSV file, if it's knn, use knn CSV file
    if (classificationType == "nn") {
        csvName = "../src/nn.csv";
    } else if (classificationType == "knn") {
        csvName = "../src/knn.csv";
    }

    // Initializations
    cv::Mat capturedFrame, thresholded, dilated, eroded, segmented, featuresFrame; // Frames for preview
    cv::Mat regionMap; // Region map
    std::vector<int> topRegionIds; // Stores top 3 objects closest to the center of the frame / extracted from the region map
    std::vector<double> features; // Stores features for each object, meant to be overritten
    char key; // For keypress

    // Initialize video capture
    cv::VideoCapture capture(1); // 1 for iPhone back camera

    // Load the CSV file and calculate standard deviations for scaling
    std::map<std::string, std::vector<std::vector<double>>> objectsDB = loadObjectsDatabase(csvName);
    std::vector<double> stdDevs = calculateStandardDeviations(objectsDB); // Calculate standard deviations for scaling

    // Check if the camera was opened
    if (!capture.isOpened()) {
        std::cout << "Unable to open the camera.\n";
        return -1;
    }

    // For freezing the frame while labeling
    bool freezeFrame = false;

    cv::namedWindow("Video Display", 1);

    // Video feed
    while (true) {
        if (!freezeFrame) {
            capture >> capturedFrame;
        }

        // Check if the frame is empty
        if (capturedFrame.empty()){
            std::cout << "Frame is empty\n";
            break;
        }

        // Task 1: Threshold the video - implemented from scratch
        customThreshold(capturedFrame, thresholded);
        imshow("Original Video", capturedFrame);
        imshow("Thresholded Video", thresholded);

        // Task 2: Dilate the objects - implemented from scratch (Extension 1)
        customDilate(thresholded, dilated);
        imshow("Dilated Video", dilated);

        // Task 2: Erode the objects - implemented from scratch (Extension 1)
        customErode(dilated, eroded);
        //imshow("Eroded Video", eroded);

        // Task 3: Segment the image - implemented from scratch (Extension 2)
        customSegmentation(eroded, segmented, regionMap, topRegionIds); // populates the topRegionIds - 3 objects closest to the center
        //segmentationCV(eroded, segmented, regionMap, topRegionIds); // implemented for comparison using OpenCV
        imshow("Segmented Video", segmented);

        // How many unknown objects in the frame / to be used inside computeFeatures
        int unknownCounter = 0;
        featuresFrame = capturedFrame.clone(); // clone capturedframe for preview, to overlay object stats on it

        std::map<int, std::vector<double>> featuresMap; // used in normal labeling mode
        std::map<std::string, std::vector<double>> labeledMap; // used in automatic labeling mode, stores the label with features

        // Task 4 and 6: Iterate through the top 3 objects closest to the center, compute features and classify
        for (int regionID : topRegionIds) {

            // Get the features and the label for the object - this function calls the classification method as well
            auto [label, features] = computeFeatures(regionMap, regionID, featuresFrame, objectsDB, stdDevs, unknownCounter, classificationType);

            // Add it to maps for the purpose of labeling modes
            labeledMap[label] = features;
            featuresMap[regionID] = features;
        }

        // Show the object stats and features  
        imshow("Features Frame", featuresFrame);

        // Check for a keypress
        key = cv::waitKey(10);


        if (key == 'q') { // quit
            return 1;
        } else if (key == 'n') { // Task 5: Normal labeling mode - collect training data

            // Freeze the frame
            freezeFrame = true;
            std::cout << "Enter labels for the current regions:\n";

            // Iterate through all top regions and ask for a label
            for (int regionID : topRegionIds) {
                std::cout << "Enter label for region " << regionID << ": ";
                std::string label;
                std::cin >> label;

                // Store the features for the region with its label in the appropriate CSV file
                features = featuresMap[regionID];
                storeFeatures(label, features, csvName);
                std::cout << "Stored feature vector for " << label << " in region " << regionID << ".\n";

                // Reload the CSV file and standard deviations for dynamic labeling
                objectsDB = loadObjectsDatabase(csvName);
                stdDevs = calculateStandardDeviations(objectsDB);
            }

            // Show the frozen frame until 'b' is pressed
            std::cout << "Press 'B' again to resume.\n";
            while (true) {
                key = cv::waitKey(1);
                if (key == 'b') {
                    freezeFrame = false;  // Resume frame processing
                    std::cout << "Resuming regular feed.\n";
                    break;
                }
            }
        } else if (key == 'a') { // Extension 3: Automatic labeling mode

            // Freeze frame
            freezeFrame = true;
            std::cout << "Enter labels for the current regions:\n";

            // Iterate through map and only if the name contains Unknown, ask to label it
            for (const auto& [name, features] : labeledMap) {
                if (name.find("Unknown") != std::string::npos) {
                    std::cout << "Enter label for " << name << ": ";
                    std::string label;
                    std::cin >> label;

                    // Store the features for the region with its label in the appropriate CSV file
                    storeFeatures(label, features, csvName);
                    std::cout << "Stored feature vector for " << name << " as " << label << ".\n";

                    // Reload the CSV file and standard deviations for dynamic labeling
                    objectsDB = loadObjectsDatabase(csvName);
                    stdDevs = calculateStandardDeviations(objectsDB);
                }
            }

            // Show the frozen frame until 'b' is pressed
            std::cout << "Press 'B' again to resume.\n";
            while (true) {
                key = cv::waitKey(1);
                if (key == 'b') {
                    freezeFrame = false;  // Resume frame processing
                    std::cout << "Resuming regular feed.\n";
                    break;
                }
            }
        }
    }
}