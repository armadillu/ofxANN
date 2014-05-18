//
//  ofxANN.cpp
//  emptyExample
//
//  Created by Greg Borenstein on 5/11/14.
//
//

#include "ofxANN.h"


ofxANN::ofxANN(int maxPoints, int maxResults){
    // defaults
    nDim = 3;
    eps = 0;
	kdTree = NULL;
	dataPoints = NULL;
	queryPt = annAllocPt(nDim);
	nnIdx = new ANNidx[maxResults];
	dists = new ANNdist[maxResults];
	dataPoints = annAllocPts(maxPoints, nDim); //TODO this is so ugly and will cause problems
}

void ofxANN::setEps(double anEps){
    eps = anEps;
}

void ofxANN::loadPoints(vector<ofVec3f*>& vertices){

	for(int i = 0; i < vertices.size(); i++){
		dataPoints[i] = &(vertices[i]->x);
	}
	numP = vertices.size();
}

void ofxANN::generateTree(){
	if (kdTree != NULL){
		delete kdTree;
	}
	kdTree = new ANNkd_tree(dataPoints, numP, nDim);
}



AnnResult ofxANN::getNeighbors(int k, ofVec3f p){

	AnnResult res;
	queryPt = &p.x;
	kdTree->annkSearch(queryPt, k, nnIdx, dists, eps);
	res.indexs = nnIdx;
	res.dists = dists;

	return res;
}

ofxANN::~ofxANN(){
    delete kdTree;
    annClose();
}