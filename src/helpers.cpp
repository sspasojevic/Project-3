/**
 * Sara Spasojevic
 * CS5330, Spring 2025
 * Project 3
 * Feb 20, 2025
 *
 * The file contains filter implementations of filters from helpers.hpp. They are being used in detect_object.cpp.
 * Docstrings explaining the purpose of each function are located in helpers.hpp.
 */

// Imports
#include "helpers.hpp"
#include "iostream"
#include <numeric>
#include <fstream>

/**
 * @class UnionFind
 * A data structure to manage disjoint sets efficiently.
 * 
 * This class implements the Union-Find data structure with union by rank.
 * Used in two-pass segmentation algorithm.
 */
class UnionFind {
public:

    /**
     * Constructs a Union-Find structure with `n` elements. Each element is initially its own parent,
     * and ranks are initialized to zero.
     * 
     * @param n The number of elements.
     */
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0); // Initially, all elements have rank 0
        for (int i = 0; i < n; ++i) {
            parent[i] = i;  // Each element is its own parent
        }
    }

    /**
     * Finds the root of the set containing x. Links nodes to the root.
     * @param x The element whose representative is to be found.
     * 
     * @return The representative of the set containing `x`.
     */
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    /**
     * Merges the sets containing x and y. Uses union by rank to keep the tree shallow.
     * 
     * @param x An element in the first set.
     * @param y An element in the second set.
     */
    void unionSets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            // Attach the smaller tree to the larger tree
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++; // Increase rank if trees were of equal height
            }
        }
    }

private:
    std::vector<int> parent; // stores the parent of each element.
    std::vector<int> rank; // stores the rank/depth of each set.
};

// Task 1: Thresholding - Dynamic, implemented from scratch
void customThreshold(cv::Mat &src, cv::Mat &dst) {

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Blur the image slightly
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);

    // Make sure that the output is of same size as the destination, but one channel
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // Initialize vector to store random pixels
    std::vector<int> pixels;

    int sampleSize = blurred.total() / 16; // Get 1/16 of the number of pixels in image

    // Get that number of random pixels and remember them in pixels
    for (int i = 0; i < sampleSize; i++) {
        int randRow = rand() % blurred.rows;
        int randCol = rand() % blurred.cols;
        pixels.push_back(blurred.at<uchar>(randRow, randCol));
    }

    // Initialize centroids from minimum value and maximum value (because binary, background VS object)
    int centroidOne = *std::min_element(pixels.begin(), pixels.end());
    int centroidTwo = *std::max_element(pixels.begin(), pixels.end());

    std::vector<int> clusterOne, clusterTwo;

    // Iterate over each pixel in random pixels, and update the centroids (means)
    for (int i = 0; i < 10; i++) {
        clusterOne.clear();
        clusterTwo.clear();

        for (int pixel : pixels) {
            if (abs(pixel - centroidOne) < abs(pixel - centroidTwo))
                clusterOne.push_back(pixel);
            else
                clusterTwo.push_back(pixel);
        }

        centroidOne = clusterOne.empty() ? centroidOne : std::accumulate(clusterOne.begin(), clusterOne.end(), 0) / clusterOne.size();
        centroidTwo = clusterTwo.empty() ? centroidTwo : std::accumulate(clusterTwo.begin(), clusterTwo.end(), 0) / clusterTwo.size();
    }

    // Threshold will be in between two centroids
    int threshold = (centroidOne + centroidTwo) / 2;

    // Iterate through pixels and check against the threshold. If they are below the threshold - white, if above, black
    for (int i = 0; i < blurred.rows; i++) {
        for (int j = 0; j < blurred.cols; j++) {
            dst.at<uchar>(i, j) = (blurred.at<uchar>(i, j) > threshold) ? 0 : 255;
        }
    }
}

// Task 2: Custom erode (Extension 1)
void customErode(cv::Mat &src, cv::Mat &dst) {
    dst = src.clone();

    // Kernel size, and symmetric k size
    int kernelSize = 15;
    int k = kernelSize / 2;

    // Iterate through the image
    for (int i = k; i < src.rows - k; i++) {
        for (int j = k; j < src.cols - k; j++) {
            
            // If the pixel at the center is black (background) -- then erode; if it's not, we skip it
            if (src.at<uchar>(i, j) == 0) {

                // Iterate through the kernel pixels
                for (int m = -k; m <= k; m++) {
                    for (int n = -k; n <= k; n++) {
                        int x = i + m;
                        int y = j + n;
                        dst.at<uchar>(x, y) = 0; // Set the surrounding pixels to black, all under the kernel
                    }
                }
            }
        }
    }
}

// Task 2: Custom dilate (Extension 1)
void customDilate(cv::Mat &src, cv::Mat &dst) {
    dst = src.clone();

    // Kernel size, and symmetric k size
    int kernelSize = 9;
    int k = kernelSize / 2;

    // Iterate through the image
    for (int i = k; i < src.rows - k; i++) {
        for (int j = k; j < src.cols - k; j++) {
            
            // If the pixel at the center is white (foreground) -- then dilate; if it's not, we skip it
            if (src.at<uchar>(i, j) == 255) {

                // Iterate through the kernel pixels
                for (int m = -k; m <= k; m++) {
                    for (int n = -k; n <= k; n++) {
                        int x = i + m;
                        int y = j + n;
                        dst.at<uchar>(x, y) = 255; // Set the surrounding pixels to white, all under kernel
                    }
                }
            }
        }
    }
}

// Custom two-pass segmentation (Extension 2)
void twoPass (const cv::Mat& src, cv::Mat& label) {
        
    int rows = src.rows;
    int cols = src.cols;

    label = cv::Mat::zeros(rows, cols, CV_32S);

    int nextLabel = 1;

    // Initialize the union find
    UnionFind uf(rows * cols);

    // First pass: Assign labels and find unions for each pixel
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (src.at<uchar>(i, j) == 0) continue; // If pixel is the background (0), skip it

            // For storing labels of neighboring foreground pixels
            std::vector<int> neighbors;

            // Check the neighbor to the top, and if not background or out of bounds, add to list of neighbors
            if (i > 0 && label.at<int>(i - 1, j) > 0) {
                neighbors.push_back(label.at<int>(i - 1, j));
            }

            // Check the neighbor to the left, and if not background or out of bounds, add to list of neighbors
            if (j > 0 && label.at<int>(i, j - 1) > 0) {
                neighbors.push_back(label.at<int>(i, j - 1));
            }

            // If there are no foreground neighbors
            if (neighbors.empty()) {
                label.at<int>(i, j) = nextLabel; // assign the value of label
                uf.unionSets(nextLabel, nextLabel); // Union with itself to initialize the set
                nextLabel++; // Increment for the next potential new label
            } else {
                // If there are labeled neighbors, choose the smallest label
                int minLabel = *std::min_element(neighbors.begin(), neighbors.end());
                label.at<int>(i, j) = minLabel; // Assign the minimum label

                // Merge different labels found in the neighbors into the same set
                for (int neighborLabel : neighbors) {
                    if (minLabel != neighborLabel) {
                        uf.unionSets(minLabel, neighborLabel);
                    }
                }
            }
        }
    }

    // Second Pass: Resolve labels using Union-Find
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (label.at<int>(i, j) > 0) {
                // Find the root representative label for the component
                int rootLabel = uf.find(label.at<int>(i, j));
                label.at<int>(i, j) = rootLabel; // Assign the root label
            }
        }
    }
}

// Task 3: Segmentation - using custom two-pass (as extension 2)
void customSegmentation(cv::Mat &src, cv::Mat &dst, cv::Mat &label, std::vector<int> &topRegionIds) {

    // Get image dimensions
    int rows = src.rows;
    int cols = src.cols;

    // Perform two-pass connected-component labeling to generate labels for different regions
    twoPass(src, label);

    std::vector<cv::Point2f> centroids; // Store centroids of valid regions
    int minSize = 1000; // Minimum number of pixels required for a region to be considered

    // Dictionary to store pixels corresponding to each label
    std::map<int, std::vector<cv::Point>> regionPixels;

    // Iterate over each pixel in the image to collect pixel positions for each labeled region
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int lbl = label.at<int>(i, j);
            if (lbl > 0) regionPixels[lbl].push_back(cv::Point(j, i));
        }
    }

    // Predefined set of colors
    std::vector<cv::Vec3b> colorPalette = {
        cv::Vec3b(0, 0, 255),    // Red
        cv::Vec3b(0, 255, 0),    // Green
        cv::Vec3b(255, 0, 0),    // Blue
        cv::Vec3b(0, 255, 255),  // Yellow
        cv::Vec3b(255, 255, 0),  // Cyan
        cv::Vec3b(255, 0, 255),  // Magenta
    };

    std::map<int, cv::Vec3b> labelColors; // Maps labels to their assigned colors
    int newLabel = 1; // New label ID counter for filtered regions

    std::map<int, int> labelMap; // Maps old labels to new ones after filtering

    std::vector<std::pair<int, float>> regionDistances;  // Stores region labels and their distances from the image center

    // Get the image center
    cv::Point2f imageCenter(cols / 2.0, rows / 2.0);
    
    // Iterate through all detected regions
    for (const auto& region : regionPixels) {
        int labelId = region.first; // Original label ID
        const std::vector<cv::Point>& pixels = region.second; // Get pixel positions belonging to this label

        // Ignore small regions, small than the size specified
        if (pixels.size() < minSize) continue;

        // Get the bounding box of the region
        cv::Rect boundingBox = cv::boundingRect(pixels);

        int margin = 10;  // Margin to exclude regions near the image border

        // Check if the bounding box is too close to the boundary and exclude it
        if (boundingBox.x < margin || boundingBox.y < margin || 
            boundingBox.x + boundingBox.width > cols - margin || 
            boundingBox.y + boundingBox.height > rows - margin) {
            continue;
        }

        // Compute the centroid of the region
        cv::Point2f centroid(0, 0);
        for (const cv::Point& p : pixels) {
            centroid.x += p.x;
            centroid.y += p.y;
        }
        centroid.x /= pixels.size();
        centroid.y /= pixels.size();
        centroids.push_back(centroid);

        // Compute the Euclidean distance of the centroid from the image center
        float distance = std::sqrt((centroid.x - imageCenter.x) * (centroid.x - imageCenter.x) + (centroid.y - imageCenter.y) * (centroid.y - imageCenter.y));

        // Store the label and its distance to the image center
        regionDistances.push_back({newLabel, distance});

        labelMap[labelId] = newLabel;  // Map the original label to the new label

        // Assign color from the predefined palette
        labelColors[newLabel] = colorPalette[(newLabel - 1) % colorPalette.size()];
        newLabel++;
    }

    // Sort regions by the distance to the center (ascending order)
    std::sort(regionDistances.begin(), regionDistances.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });

    // Clear the previous top region IDs and select the top 3 closest regions to the center
    topRegionIds.clear();
    for (int i = 0; i < 3 && i < regionDistances.size(); ++i) {
        topRegionIds.push_back(regionDistances[i].first);
    }

    // For troubleshooting
    // std::cout << "Top 3 Region IDs closest to the center:" << std::endl;
    // for (int i = 0; i < topRegionIds.size(); ++i) {
    //     std::cout << "Region " << (i + 1) << ": " << topRegionIds[i] << std::endl;
    // }

    // Update the label matrix with the new labels
    dst = cv::Mat::zeros(rows, cols, CV_8UC3);

    // Iterate over the entire image again to update labels and assign colors to the output image
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int lbl = label.at<int>(i, j);
            if (lbl > 0) { // If pixel belongs to a labeled region
                int newLabel = labelMap[lbl];  // Map old label to new one
                label.at<int>(i, j) = newLabel;  // Update label in the map
                dst.at<cv::Vec3b>(i, j) = labelColors[newLabel];  // Assign the correct color
            }
        }
    }
}

// Segmentation - using OpenCV, for speed comparison purposes
void segmentationCV(cv::Mat &src, cv::Mat &dst, cv::Mat &label, std::vector<int> &topRegionIds) {

    // Get image dimensions
    int rows = src.rows;
    int cols = src.cols;

    // Connected components labeling with statistics
    cv::Mat stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, label, stats, centroids, 8, CV_32S);

    // To store valid labels
    std::vector<int> validLabels;
    validLabels.reserve(nLabels);

    // Get image center
    cv::Point2f imageCenter(cols / 2.0, rows / 2.0);

    // Predefined set of colors, like above
    std::vector<cv::Vec3b> colorPalette = {
        cv::Vec3b(0, 0, 255), cv::Vec3b(0, 255, 0), cv::Vec3b(255, 0, 0),
        cv::Vec3b(0, 255, 255), cv::Vec3b(255, 255, 0), cv::Vec3b(255, 0, 255)
    };

    // Go through the regions
    for (int labelId = 1; labelId < nLabels; labelId++) {
        int size = stats.at<int>(labelId, cv::CC_STAT_AREA);

        // Skip if smaller than needed size
        if (size < 1000) continue;

        // Get the box stats
        int x = stats.at<int>(labelId, cv::CC_STAT_LEFT);
        int y = stats.at<int>(labelId, cv::CC_STAT_TOP);
        int width = stats.at<int>(labelId, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(labelId, cv::CC_STAT_HEIGHT);

        // Check if it passed into the margins
        int margin = 10;
        if (x < margin || y < margin || x + width > cols - margin || y + height > rows - margin) continue;

        validLabels.push_back(labelId);
    }

    // Calculate distances of valid labels' centroids from the image center
    std::vector<std::pair<int, float>> labelDistances;
    for (int labelId : validLabels) {
        cv::Point2f centroid = centroids.at<cv::Point2f>(labelId);
        float distance = cv::norm(centroid - imageCenter);
        labelDistances.push_back({labelId, distance});
    }

    // Sort by distance to the center
    std::sort(labelDistances.begin(), labelDistances.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
        return a.second < b.second; // Ascending
    });

    // Select top 3 closest regions
    topRegionIds.clear();
    for (int i = 0; i < std::min(3, (int)labelDistances.size()); ++i) {
        topRegionIds.push_back(labelDistances[i].first);
    }

    // Create color mapping for valid labels
    std::vector<cv::Vec3b> labelColors(nLabels, cv::Vec3b(0, 0, 0));
    for (size_t i = 0; i < validLabels.size(); ++i) {
        labelColors[validLabels[i]] = colorPalette[i % colorPalette.size()];
    }

    // Assign colors
    dst = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; i++) {
            for (int j = 0; j < cols; j++) {
                int lbl = label.at<int>(i, j);
                if (lbl > 0) {
                    dst.at<cv::Vec3b>(i, j) = labelColors[lbl];
                }
            }
        }
    });
}

// Task 4: Compute features
std::pair<std::string, std::vector<double>> computeFeatures(cv::Mat& label, int regionID, cv::Mat& dst, 
                                                            const std::map<std::string, std::vector<std::vector<double>>>& objectsDB, 
                                                            const std::vector<double>& stdDevs, int& unknownCounter, 
                                                            const std::string& classificationType) {

    // Create a mask for the current region
    cv::Mat regionMask = (label == regionID);

    // Calculate moments
    cv::Moments m = cv::moments(regionMask, true);
    double area = m.m00;

    // Handle case with no valid region
    if (area < 1e-5) {
        std::cerr << "Region " << regionID << " has no pixels in the area." << std::endl;
        return {"Unknown", {0, 0, 0, 0, 0}};
    }

    // Compute centroid (center of mass)
    double centroidX = m.m10 / area;
    double centroidY = m.m01 / area;

    // Central moments for orientation
    double mu20 = m.mu20 / area;
    double mu02 = m.mu02 / area;
    double mu11 = m.mu11 / area;

    // Compute angle of the axis of least central moment
    double alpha = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);

    // Compute unit direction vector of the principal axis.
    double cosAlpha = cos(alpha);
    double sinAlpha = sin(alpha);
    cv::Point2d primaryAxis(cosAlpha, sinAlpha);

    // Extract the contour of the region by iterating through the region map.
    std::vector<cv::Point> contour;
    for (int y = 0; y < label.rows; y++) {
        for (int x = 0; x < label.cols; x++) {
            if (label.at<int>(y, x) == regionID)
                contour.push_back(cv::Point(x, y));
        }
    }

    // Compute the minimum-area bounding rectangle that encloses the region.
    cv::RotatedRect bbox = cv::minAreaRect(contour);

    // Extract the four corner points of the bounding rectangle.
    cv::Point2f rectPoints[4];
    bbox.points(rectPoints);

    // Draw the bounding box onto the destination image using green lines.
    for (int i = 0; i < 4; i++) {
        cv::line(dst, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);  // Green bounding box
    }

    // Aspect ratio
    float aspectRatio = std::max(bbox.size.width, bbox.size.height) / std::min(bbox.size.width, bbox.size.height);

    // Percent filled (ratio of pixels in region to area of bounding box)
    float percentFilled = area / (bbox.size.width * bbox.size.height);

    // Compute Hu moments
    double huMoments[7];
    cv::HuMoments(m, huMoments);

    // Visualize centroid, primary axis, and region ID
    cv::Point2d centroid(centroidX, centroidY);
    double axisLength = 200.0;  // Length of the axis for visualization

    // Get endpoints of the primary axis for visualization.
    cv::Point2d endPoint1(centroidX + axisLength * cosAlpha, centroidY + axisLength * sinAlpha);
    cv::Point2d endPoint2(centroidX - axisLength * cosAlpha, centroidY - axisLength * sinAlpha);

    // Draw the primary axis and centroid
    cv::line(dst, endPoint1, endPoint2, cv::Scalar(255, 0, 0), 2);  // Blue line for the axis
    cv::circle(dst, centroid, 5, cv::Scalar(0, 0, 255), -1);  // Red dot for centroid

    // Text for display
    std::string printText = "Region ID: " + std::to_string(regionID) + ", Aspect ratio: " + std::to_string(aspectRatio);
    cv::putText(dst, printText, centroid + cv::Point2d(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);


    // CLASSIFICATION PART

    std::string labelName; // Label name for classification

    // Classification based on the specified method
    if (classificationType == "nn") {
        // Nearest Neighbor classification
        labelName = classifyObject({aspectRatio, percentFilled, huMoments[0], huMoments[1], huMoments[2]}, objectsDB, stdDevs); // Classify the current object
    } else if (classificationType == "knn") {
        // k-Nearest Neighbors (k=3) classification
        labelName = classifyObjectKNN({aspectRatio, percentFilled, huMoments[0], huMoments[1], huMoments[2]}, objectsDB, stdDevs, 3);
    }

    // If classification result is unknown, assign an unknown label with counter
    if (labelName.find("Unknown") != std::string::npos) {
        labelName = "Unknown" + std::to_string(unknownCounter++);
    }

    // Show classification label
    cv::putText(dst, labelName, centroid + cv::Point2d(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    // Return the computed features in a map with the label name as the key
    return {labelName, {aspectRatio, percentFilled, huMoments[0], huMoments[1], huMoments[2]}};
}

// Task 5: Collecting training data
void storeFeatures(const std::string& label, const std::vector<double>& features, const std::string& csvName) {
    // Open CSV file in append mode
    std::ofstream outFile(csvName, std::ios::app);

    // Check if the file is open
    if (!outFile.is_open()) {
        std::cerr << "Error opening file to store features.\n";
        return;
    }

    // If the file is empty, write the header
    if (outFile.tellp() == 0) {
        outFile << "Label,AspectRatio,PercentFilled, HuMoment1, HuMoment2, HuMoment3\n";
    }

    // Write the label and feature vector to the CSV file
    outFile << label;
    for (double feature : features) {
        outFile << "," << feature;
    }
    outFile << "\n";  // End of line

    outFile.close();
}

// Grabs entries (labels, features) from a database
std::map<std::string, std::vector<std::vector<double>>> loadObjectsDatabase(const std::string& filename) {

    // Map where each label (class name) is associated with multiple feature vectors to support knn for several entries having same label
    std::map<std::string, std::vector<std::vector<double>>> objectsDB;

    // Open the CSV file
    std::ifstream inFile(filename);

    // Check if the file was opened.
    if (!inFile.is_open()) {
        std::cerr << "Error opening file for reading object features.\n";
        return objectsDB; // Empty map
    }

    bool firstLine;
    std::string line;

    // Skip header before processing
    if (std::getline(inFile, line)) {
        firstLine = false;
    }

    // Process each line
    while (std::getline(inFile, line)) {  
        std::stringstream ss(line);
        std::string label;

        // Object label
        std::getline(ss, label, ',');

        std::vector<double> features;
        double feature;

        // Go through features
        while (ss >> feature) {
            features.push_back(feature);
            if (ss.peek() == ',') ss.ignore();
        }

        // Add the new feature vector to the list for that label
        objectsDB[label].push_back(features);
    }

    return objectsDB;
}


std::vector<double> calculateStandardDeviations(const std::map<std::string, std::vector<std::vector<double>>>& objectsDB) {

    // Store calculated standard deviations for each feature
    std::vector<double> stdDevs;
    
    // Check if the input dataset is empty
    if (objectsDB.empty()) {
        std::cerr << "Error: objectsDB is empty. Returning an empty standard deviation vector.\n";
        return {}; // Empty vector if no data
    }

    
    size_t objectCount = 0; // Total number of feature vectors (data points)
    size_t featureCount = objectsDB.begin()->second.front().size();  // Use the first feature vector's size to get the number of features
    
    // Count the total number of feature vectors in the objectsDB
    for (const auto& entry : objectsDB) {
        objectCount += entry.second.size();  // Add number of feature vectors for each label (could be more per label)
    }

    // If there is only one object, return a default standard deviation to avoid overfitting - this was an issue I needed to solve
    if (objectCount == 1) {
        stdDevs.assign(featureCount, 1.0);
        return stdDevs;
    }

    // Initialize the standard deviations and means vectors
    stdDevs.assign(featureCount, 0.0);
    std::vector<double> means(featureCount, 0.0);

    // Get mean for each feature
    for (const auto& entry : objectsDB) {
        for (const auto& features : entry.second) {  // Iterate over each feature vector for the label
            for (size_t i = 0; i < featureCount; ++i) {
                means[i] += features[i]; // Sum the appropriate feature
            }
        }
    }

    // Calculate the mean for each feature by dividing with total number of entries
    for (size_t i = 0; i < featureCount; ++i) {
        means[i] /= objectCount;
    }

    // Get the variance for each feature
    for (const auto& entry : objectsDB) {
        for (const auto& features : entry.second) {  // Iterate over each feature vector for the label
            for (size_t i = 0; i < featureCount; ++i) {
                // Sum of squared deviations from the mean
                stdDevs[i] += (features[i] - means[i]) * (features[i] - means[i]);
            }
        }
    }

    // Get standard deviation - square root of the variance
    for (size_t i = 0; i < featureCount; ++i) {
        stdDevs[i] = std::sqrt(stdDevs[i] / (objectCount - 1));  // Unbiased sample std deviation
    }

    return stdDevs;
}


// Function to compute the scaled Euclidean distance - this is between two vectors
double computeScaledEuclideanDistance(const std::vector<double>& features1, const std::vector<double>& features2, const std::vector<double>& stdDevs) {
    
    // Check if the feature vectors and standard deviation vector have the same size
    if (features1.size() != features2.size() || features1.size() != stdDevs.size()) {
        std::cerr << "Feature vectors and standard deviation vectors must have the same size.\n";
        return -1;
    }

    // Iterate over all the features
    double distance = 0.0;
    for (size_t i = 0; i < features1.size(); ++i) {
        // Calculate the scaled difference between the features
        double diff = (features1[i] - features2[i]) / stdDevs[i]; // Scale by the standard deviation
        distance += diff * diff; // Accumulate the squared difference
    }

    // Euclidean distance
    return std::sqrt(distance);
}

// Classify an object using NN based on its feature vector by comparing it to a database of known objects
std::string classifyObject(const std::vector<double>& unknownFeatures, 
                           const std::map<std::string, std::vector<std::vector<double>>>& objectsDB, 
                           const std::vector<double>& stdDevs, 
                           double maxDistanceThreshold) {

    // Initialize minDistance with the maximum possible value                        
    double minDistance = std::numeric_limits<double>::max();

    // Default classification is "Unknown"
    std::string closestLabel = "Unknown";

    // Iterate through all entries in the database
    for (const auto& entry : objectsDB) {
        // For NN - Get the feature vector for the first object under each label (nn.csv will have one feature vector for each class label)
        const std::vector<double>& features = entry.second.front();
        
        // Compute the scaled Euclidean distance between the unknown object and the current object in the database
        double distance = computeScaledEuclideanDistance(unknownFeatures, features, stdDevs);

        //std::cout << "Distance to " << entry.first << ": " << distance << std::endl; // Troubleshooting to find threshold
        
        // Update the minimum distance and closest label if the current distance is smaller
        if (distance < minDistance) {
            minDistance = distance;
            closestLabel = entry.first;
        }
    }

    // If the minimum distance is too high, classify as "Unknown"
    if (minDistance > maxDistanceThreshold) {
        return "Unknown";
    }

    return closestLabel;
}

// Classify an object using k-NN based on its feature vector by comparing it to a database of known objects
std::string classifyObjectKNN(const std::vector<double>& unknownFeatures, 
                               const std::map<std::string, std::vector<std::vector<double>>>& objectsDB, 
                               const std::vector<double>& stdDevs, 
                               int K, 
                               double maxDistanceThreshold) {

    // Map to store the distances for each class label
    std::map<std::string, std::vector<double>> labelDistances;

    // Iterate through each class
    for (const auto& entry : objectsDB) {
        const std::string& label = entry.first;
        
        // Iterate over all feature vectors for this class to get the distances
        for (const auto& features : entry.second) {

            // Calculate the scaled Euclidean distance
            double distance = computeScaledEuclideanDistance(unknownFeatures, features, stdDevs);

            // Add this distance to the list of distances for this class
            labelDistances[label].push_back(distance);
        }
    }

    // Closes label is unknown by default and minDistanceSum is the maximum possible value
    std::string closestLabel = "Unknown";
    double minDistanceSum = std::numeric_limits<double>::max();

    // Iterate through each class and its distances
    for (auto& labelEntry : labelDistances) {

        // Sort the distances for this class
        std::sort(labelEntry.second.begin(), labelEntry.second.end());

        // // USED FOR TWEAKING THE THRESHOLD
        // std::cout << "Sorted distances for label " << labelEntry.first << ": ";
        // for (auto dist : labelEntry.second) {
        //     std::cout << dist << " ";
        // }
        // std::cout << std::endl;

        // Sum the K smallest distances (or all if there are fewer than K)
        double sumOfDistances = 0.0;
        int count = std::min(K, static_cast<int>(labelEntry.second.size()));
        for (int i = 0; i < count; ++i) {
            sumOfDistances += labelEntry.second[i];
        }

        // For troubleshooting and tweaking the threshold
        // std::cout << "Sum of " << count << " smallest distances for label " << labelEntry.first << ": " << sumOfDistances << std::endl;

        // Label will be the smallest sum of distances
        if (sumOfDistances < minDistanceSum) {
            minDistanceSum = sumOfDistances;
            closestLabel = labelEntry.first;
        }
    }

    // For throubleshooting
    // std::cout << "Closest label: " << closestLabel << ", Minimum distance sum: " << minDistanceSum << std::endl;

    // Check if the minimum sum of distances exceeds the threshold, if yes, label is unknown
    if (minDistanceSum > maxDistanceThreshold) {
        return "Unknown";
    }

    return closestLabel;
}
