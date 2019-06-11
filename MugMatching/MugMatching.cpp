﻿#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int, char**)
{
	Mat frame;
	Mat grayFrame;
	Mat filteredFrame;
	Mat FreezedFrame;
	bool viewFrame = true;
	int openCnt = 0;
	int closeCnt = 0;
	vector<Mat> contours;
	Mat templateA, templateB, templateC;
	int settingMode=0;
	int showCountourID = 0;
	ShapeMatchModes matchmode = ShapeMatchModes::CONTOURS_MATCH_I1;
	double matchThreshold = 0.1;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 1;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID + apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl;
	for (;;)
	{
		if (settingMode) {
			Mat showContour(frame.rows, frame.cols, CV_8UC1, Scalar(0));
			drawContours(showContour, contours, showCountourID, Scalar(255, 0, 0), 1, 8);
			imshow("Live", frame);
			imshow("Setting Template", showContour);
		}
		else {
			// wait for a new frame from camera and store it into 'frame'
			cap.read(frame);
			// check if we succeeded
			if (frame.empty()) {
				cerr << "ERROR! blank frame grabbed\n";
				break;
			}
			// show live and wait for a key with timeout long enough to show images
			cvtColor(frame, grayFrame, ColorConversionCodes::COLOR_BGR2GRAY);
			bilateralFilter(grayFrame, filteredFrame, 5, 30, 30);
			double thresh = threshold(filteredFrame, filteredFrame, 0, 255, ThresholdTypes::THRESH_OTSU);
			Canny(filteredFrame, filteredFrame, thresh * 0.5, thresh);
			morphologyEx(filteredFrame, filteredFrame, MorphTypes::MORPH_CLOSE, Mat());
			findContours(filteredFrame, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_SIMPLE);
			double minA, minB, minC;
			int Aidx, Bidx, Cidx;
			Aidx = Bidx = Cidx = -1;
			minA = minB = minC = DBL_MAX;
			double tmp;
			if (templateA.rows != 0) {
				for (int i = 0; i < contours.size();i++) {
					tmp = matchShapes(templateA, contours[i], matchmode, 0);
					if (tmp < minA) {
						minA = tmp;
						Aidx = i;
					}
				}
			}
			if (templateB.rows != 0) {
				for (int i = 0; i < contours.size(); i++) {
					tmp = matchShapes(templateB, contours[i], matchmode, 0);
					if (tmp < minB) {
						minB = tmp;
						Bidx = i;
					}
				}
			}
			if (templateC.rows != 0) {
				for (int i = 0; i < contours.size(); i++) {
					tmp = matchShapes(templateC, contours[i], matchmode, 0);
					if (tmp < minC) {
						minC = tmp;
						Cidx = i;
					}
				}
			}
			if (!viewFrame)
				frame = filteredFrame;
			if (minA == minB && minB == minC && minA == DBL_MAX) {
				cout << "Template not setted" << endl;
			}
			else {
				if (minA < matchThreshold) {
					cout << "A mug " << minA << endl;
					drawContours(frame, contours, Aidx, Scalar(255, 0, 0), 2, 8);
				}
				if (minB < matchThreshold) {
					cout << "B mug " << minB << endl;
					drawContours(frame, contours, Bidx, Scalar(255, 0, 0), 2, 8);
				}
				if (minC < matchThreshold) {
					cout << "C mug " << minC << endl;
					drawContours(frame, contours, Cidx, Scalar(255, 0, 0), 2, 8);
				}
			}
			imshow("Live", frame);
		}
		switch (waitKey(1))
		{
		case 'S':
		case 's':
			cout << "Enter Template Setting Mode:" << endl;
			showCountourID = 0;
			FreezedFrame = filteredFrame;
			settingMode = 1;
			break;
		case 'A':
		case 'a':
			cout << "Setting Template A:" << endl;
			showCountourID = 0;
			settingMode = 2;
			break;
		case 'B':
		case 'b':
			cout << "Setting Template B:" << endl;
			showCountourID = 0;
			settingMode = 3;
			break;
		case 'C':
		case 'c':
			cout << "Setting Template C:" << endl;
			showCountourID = 0;
			settingMode = 4;
			break;
		case '+':
			showCountourID++;
			showCountourID = min(showCountourID, (int)contours.size() - 1);
			cout << showCountourID << "/" << contours.size() - 1 << endl;
			break;
		case '-':
			showCountourID--;
			showCountourID = max(showCountourID, 0);
			cout << showCountourID << "/" << contours.size() - 1 << endl;
			break;
		case 13:
			if (settingMode == 1) {
				cout << "Exit settingMode" << endl;
			}
			else if(settingMode == 2){
				cout << "Template A setted" << endl;
				templateA = contours[showCountourID];
			}
			else if (settingMode == 3) {
				cout << "Template B setted" << endl;
				templateB = contours[showCountourID];
			}
			else if (settingMode == 4) {
				cout << "Template C setted" << endl;
				templateC = contours[showCountourID];
			}
			destroyWindow("Setting Template");
			settingMode = 0;
			break;
		case '\\':
			cout << "switch view" << endl;
			viewFrame = !viewFrame;
			break;
		case '1':
			matchmode = ShapeMatchModes::CONTOURS_MATCH_I1;
			break;
		case '2':
			matchmode = ShapeMatchModes::CONTOURS_MATCH_I2;
			break;
		case '3':
			matchmode = ShapeMatchModes::CONTOURS_MATCH_I3;
			break;
		case 't':
		case 'T':
			cout << "new matching threshold:";
			cin >> matchThreshold;
			break;

		default:
			break;
		}
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}