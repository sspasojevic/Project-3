/**
 * Sara Spasojevic
 * CS5330, Spring 2025
 * Project 3
 * Feb 20, 2025
 *
 * The file contains header and named functions are implemented in helpers.cpp. They are being used in detect_object.cpp.
 * Docstrings explaining the purpose of each function are located here and will not be repeated in helpers.cpp.
 */

#ifndef HELPERS_HPP
#define HELPERS_HPP

// Imports
#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * Thresholds the input frame by dynamically seting the threshold by analyzing the pixel values. 
 * It uses the ISODATA algorithm (k-means on a 1/16 random sample of the pixels in the image) with K=2, to find the mean
 * of the two dominant colors - the threshold is the average of those means.
 * 
 * (Before thresholding, it converts the image to grayscale and blurs it with 5x5 Gaussian filter).
 *
 * @param src The input image.
 * @param dst The output image, binary.
 */
void customThreshold(cv::Mat &src, cv::Mat &dst);

/**
 * Performs custom eroding, 8-connected. If the pixel is the background, everything under the kernel becomes background, otherwise skips it.
 *
 * @param src The input image.
 * @param dst The output image, binary.
 */
void customErode(cv::Mat &src, cv::Mat &dst);

/**
 * Performs custom dilating, 8-connected. If the pixel is the foreground, everything under the kernel becomes foreground, otherwise skips it.
 *
 * @param src The input image.
 * @param dst The output image, binary.
 */
void customDilate(cv::Mat &src, cv::Mat &dst);

/**
 * Performs the custom segmentation of the input image into distinct regions using custom two-pass algorithm.
 * It filters regions based on size and proximity to the boundaries (with margin) and assigns colors to significant
 * regions. It also updates the topRegionIds by selecting the top three regions closest to the image center based on their centroids.
 *
 * @param binary_image The input binary image.
 * @param dst The output image with colored segmented regions.
 * @param label The labeled matrix where each pixel's value corresponds to a unique region.
 * @param topRegionIds A vector storing the IDs of the top three regions closest to the image center.
 */
void customSegmentation(cv::Mat &binary_image, cv::Mat &dst, cv::Mat &label, std::vector<int> &topRegionIds);

/**
 * Performs the segmentation of the input image into distinct regions using OpenCV's connectedComponentsWithStats.
 * It filters regions based on size and proximity to the boundaries (with margin) and assigns colors to significant
 * regions. It also updates the topRegionIds by selecting the top three regions closest to the image center based on their centroids.
 *
 * @param src The input binary image.
 * @param dst The output image with colored segmented regions.
 * @param label The labeled matrix where each pixel's value corresponds to a unique region.
 * @param topRegionIds A vector storing the IDs of the top three regions closest to the image center.
 */
void segmentationCV(cv::Mat &src, cv::Mat &dst, cv::Mat &label, std::vector<int> &topRegionIds);

/**
 * Computes feature vectors for a given region in a labeled image.
 * It makes use of the appropriate classification method to classify the object based on a database of known objects.
 * It also draws the properties of the object for preview (features and stats).
 *
 * @param label The labeled matrix where each pixel's value corresponds to a unique regionID.
 * @param regionID The ID of the region to compute features for.
 * @param dst The output image used for visualization.
 * @param objectsDB The database of known object feature vectors.
 * @param stdDevs The standard deviations of features used for normalization.
 * @param unknownCounter A counter tracking the number of unknown objects in the frame.
 * @param classificationType The type of classification method to be used.
 * 
 * @return A pair that is the classification label and the computed feature vector.
 */
std::pair<std::string, std::vector<double>> computeFeatures(cv::Mat& label, int regionID, cv::Mat& dst, 
                                                            const std::map<std::string, std::vector<std::vector<double>>>& objectsDB, 
                                                            const std::vector<double>& stdDevs, int& unknownCounter, 
                                                            const std::string& classificationType);

/**
 * Stores a computed feature vector into a CSV file.
 *
 * @param label The label of the object.
 * @param features The feature vector.
 * @param csvName The name of the CSV file where features will be stored.
 */
void storeFeatures(const std::string& label, const std::vector<double>& features, const std::string& csvName);

/**
 * Loads an object database from a CSV file.
 * It supports more entries having the same labels for knn functionality, and stores them as vector of vectors for that label.
 *
 * @param filename The name of the CSV file containing object feature vectors.
 * 
 * @return A map containing object labels as keys and vectors of feature vectors as values. 
 */
std::map<std::string, std::vector<std::vector<double>>> loadObjectsDatabase(const std::string& filename);

/**
 * Calculates the standard deviations of features across the object database for normalization.
 *
 * @param objectsDB The database of known object feature vectors.
 * 
 * @return A vector containing the standard deviation of each feature.
 */
std::vector<double> calculateStandardDeviations(const std::map<std::string, std::vector<std::vector<double>>>& objectsDB);

/**
 * Classifies an object based on its feature vector using scaled Euclidean distance.
 *
 * @param unknownFeatures The feature vector of the unknown object.
 * @param objectsDB The database of known object feature vectors.
 * @param stdDevs The standard deviations used for feature normalization.
 * @param maxDistanceThreshold The maximum allowable distance for classification.
 * 
 * @return The label of the classified object or "Unknown" if no match is found.
 */
std::string classifyObject(const std::vector<double>& unknownFeatures, 
                           const std::map<std::string, std::vector<std::vector<double>>>& objectsDB, 
                           const std::vector<double>& stdDevs, 
                           double maxDistanceThreshold = 2.0); 

/**
 * Classifies an object using the k-Nearest Neighbors (k-NN) algorithm.
 *
 * @param unknownFeatures The feature vector of the unknown object.
 * @param objectsDB The database of known object feature vectors.
 * @param stdDevs The standard deviations used for feature normalization.
 * @param K The number of nearest neighbors to consider.
 * @param maxDistanceThreshold The maximum allowable distance for classification.
 * 
 * @return The label of the classified object or "unknown" if no match is found.
 */
std::string classifyObjectKNN(const std::vector<double>& unknownFeatures, 
                                const std::map<std::string, std::vector<std::vector<double>>>& objectsDB, 
                                const std::vector<double>& stdDevs, 
                                int K, 
                                double maxDistanceThreshold = 20.0);             




#endif