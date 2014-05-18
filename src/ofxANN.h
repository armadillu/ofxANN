#pragma once

#include "ANN.h"
#include "ofMain.h"

struct AnnResult{
    ANNidxArray indexs;
    ANNdistArray dists;
};

class ofxANN {
public:
	ofxANN(int maxPoints = 10000, int maxResults = 1000);
    ~ofxANN();

    void setEps(double anEps);
    void loadPoints(vector<ofVec3f*>& vertices);
	void generateTree();
    AnnResult getNeighbors(int k, ofVec3f p);

private:
    int nDim;
    double eps;
    ANNkd_tree* kdTree;
    ANNpointArray dataPoints;
	ANNpoint queryPt;
	ANNidxArray	nnIdx;
	ANNdistArray dists;
	int numP;
};


