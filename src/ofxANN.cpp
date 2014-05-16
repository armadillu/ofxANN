//
//  ofxANN.cpp
//  emptyExample
//
//  Created by Greg Borenstein on 5/11/14.
//
//

#include "ofxANN.h"


ofxANN::ofxANN(){
    // defaults
    nDim = 3;
    eps = 0;
	kdTree = NULL;
	dataPoints = NULL;
	queryPt = annAllocPt(nDim);
	nnIdx = new ANNidx[1024];
	dists = new ANNdist[1024];
	dataPoints = annAllocPts(1024, nDim);
}

void ofxANN::setEps(double anEps){
    eps = anEps;
}

void ofxANN::loadPoints(vector<ofVec3f*>& vertices){

	for(int i = 0; i < vertices.size(); i++){
		dataPoints[i] = &(vertices[i]->x);
	}
	if (kdTree != NULL){
		delete kdTree;
	}
	kdTree = new ANNkd_tree(dataPoints, vertices.size(), nDim);
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