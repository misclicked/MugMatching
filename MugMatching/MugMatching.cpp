#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int thresh = 100;


///Trackbar variable
int FilterType = 1; //0 高斯 1 平滑
int FilterSize = 5;
int EdgeDetectType = 0;//0 Canny 1 Sobel
int morthType = 4; //0~7
int ContourMode = 1; //1~4
int retrMode = 0; //0~4
int viewMode = 0; //0: frame 1: filteredframe 2:contourframe
int selectTemplate = 0;//0: None 1: TemplateA 2: TemplateB 3: TemplateC
int showCountourID = 0;
int setTemplateDone = 0;
int matchMode = 1; //1~3;
///end
vector<Mat> contours;
Mat templateA, templateB, templateC;
bool dFlag = false;
void resetContourID(int, void*) {
	showCountourID = 0;
	setTemplateDone = 0;
	dFlag = true;
}
void setTemplate(int, void*) {
	if (selectTemplate == 1) {
		templateA = contours[showCountourID];
	}
	if (selectTemplate == 2) {
		templateB = contours[showCountourID];
	}
	if (selectTemplate == 3) {
		templateC = contours[showCountourID];
	}
	selectTemplate = 0;
	dFlag = true;
}
int main(int, char**)
{
	Mat frame;
	Mat grayFrame;
	Mat filteredFrame;
	Mat FreezedFrame;
	Mat contourFrame;
	Mat pic;
	bool viewFrame = true;
	int openCnt = 0;
	int closeCnt = 0;
	double matchThreshold = 0.1;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID + apiID);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl;
	for (;;)
	{
		if (dFlag) {
			destroyWindow("Live");
			dFlag = false;
		}
		if (selectTemplate) {
			createTrackbar("Select Template", "Live", &selectTemplate, 3, resetContourID);
			createTrackbar("Select Countour", "Live", &showCountourID, contours.size()-1);
			createTrackbar("setTemplateDone", "Live", &setTemplateDone, 1, setTemplate);
			Mat showContour(frame.rows, frame.cols, CV_8UC1, Scalar(0));
			drawContours(showContour, contours, -1, Scalar(255, 0, 0), 1, 8);
			drawContours(showContour, contours, showCountourID, Scalar(255, 0, 0), 3, 8);
			imshow("Live", showContour);
		}
		else {
			cap.read(frame);
			if (frame.empty()) {
				cerr << "ERROR! blank frame grabbed\n";
				break;
			}
			createTrackbar("Select Template", "Bars", &selectTemplate, 3, resetContourID);
			createTrackbar("FilterType", "Bars", &FilterType, 1);
			createTrackbar("FilterSize", "Bars", &FilterSize, 10);
			createTrackbar("EdgeDetectType", "Bars", &EdgeDetectType, 1);
			createTrackbar("morthType", "Bars", &morthType, 7);
			createTrackbar("ContourMode", "Bars", &ContourMode, 4);
			createTrackbar("retrMode", "Bars", &retrMode, 3);
			createTrackbar("Match Mode", "Bars", &matchMode, 3);
			createTrackbar("viewMode", "Bars", &viewMode, 2);
			cvtColor(frame, grayFrame, ColorConversionCodes::COLOR_BGR2GRAY);
			if (FilterType == 0)
				GaussianBlur(grayFrame, filteredFrame, Size(FilterSize, FilterSize), 0, 0);
			else if (FilterType == 1)
				bilateralFilter(grayFrame, filteredFrame, FilterSize, 30, 30);
			if (EdgeDetectType == 0) {
				thresh = threshold(filteredFrame, grayFrame, 0, 255, ThresholdTypes::THRESH_OTSU);
				Canny(filteredFrame, filteredFrame, thresh * 0.9, thresh);
				morphologyEx(filteredFrame, filteredFrame, morthType, Mat());
			}
			else if (EdgeDetectType == 1) {
				Mat grad_x, grad_y;
				Mat abs_grad_x, abs_grad_y;
				Sobel(filteredFrame, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
				convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U
				Sobel(filteredFrame, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
				convertScaleAbs(grad_y, abs_grad_y);
				addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, filteredFrame);
				threshold(filteredFrame, filteredFrame, 80, 255, THRESH_BINARY | THRESH_OTSU);
			}
			findContours(filteredFrame, contours, retrMode, ContourMode);
			double minA, minB, minC;
			int Aidx, Bidx, Cidx;
			Aidx = Bidx = Cidx = -1;
			minA = minB = minC = DBL_MAX;
			double tmp;
			if (templateA.rows != 0) {
				for (int i = 0; i < contours.size();i++) {
					tmp = matchShapes(templateA, contours[i], matchMode, 0);
					if (tmp < minA) {
						minA = tmp;
						Aidx = i;
					}
				}
			}
			if (templateB.rows != 0) {
				for (int i = 0; i < contours.size(); i++) {
					tmp = matchShapes(templateB, contours[i], matchMode, 0);
					if (tmp < minB) {
						minB = tmp;
						Bidx = i;
					}
				}
			}
			if (templateC.rows != 0) {
				for (int i = 0; i < contours.size(); i++) {
					tmp = matchShapes(templateC, contours[i], matchMode, 0);
					if (tmp < minC) {
						minC = tmp;
						Cidx = i;
					}
				}
			}
			if (viewMode==1)
				frame = filteredFrame;
			else if (viewMode == 2) {
				contourFrame = Mat::zeros(frame.size(), CV_8UC1);
				drawContours(contourFrame, contours, -1, Scalar(255, 0, 0), 2, 8);
				frame = contourFrame;
			}
			pic = cv::Mat::zeros(250, 500, CV_8UC3);
			if (minA == minB && minB == minC && minA == DBL_MAX) {
				cout << "Template not setted" << endl;
			}
			else {
				if (minA < minB && minA < minC && minA <= 0.05) {
					putText(pic, "A", Point(250, 125), FONT_HERSHEY_SIMPLEX, 4, Scalar(255), 4, 8, false);
					drawContours(frame, contours, Aidx, Scalar(255, 0, 0), 2, 8);
				}
				else if (minB < minA && minB < minC && minB <= 0.05) {
					putText(pic, "B", Point(250, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255), 1, 8, false);
					drawContours(frame, contours, Bidx, Scalar(255, 0, 0), 2, 8);
				}
				else if (minC < minB && minC < minA && minC <= 0.05) {
					putText(pic, "C", Point(250, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255), 1, 8, false);
					drawContours(frame, contours, Cidx, Scalar(255, 0, 0), 2, 8);
				}
			}
			imshow("Bars", pic);
			imshow("Live", frame);
		}
		switch (waitKey(1))
		{
		default:
			break;
		}
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}