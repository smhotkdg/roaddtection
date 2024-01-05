///////////////////////////////////////////////////////////////////////
// OpenCV tracking example.
#pragma once


// include opencv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <stack>
#include<list>

using namespace cv;

#define cvCvtPixToPlane split
#define cvQueryHistValue_2D(hist,idx0,idx1) cvGetReal2D( (hist)->bins,(idx0),(idx1))

enum {CM_GRAY, CM_HUE, CM_RGB, CM_HSV,CM_LAB,CM_LUV};	// color model
#define PI 3.1415926
enum {MEANSHIFT, CAMSHIFT,ROAD_DETECTION}; // method

struct tracker_opencv_param
{
	int hist_bins;
	int max_itrs;
	int color_model;
	int method;



	cv::Mat src_f;
	cv::Mat dest;	

	tracker_opencv_param()
	{
		hist_bins = 16;
		max_itrs = 10;
		color_model = CM_HSV;
		method = CAMSHIFT;
	}
};

class tracker_opencv
{
public:
	tracker_opencv(void);
	~tracker_opencv(void);

	void init(Mat img, Rect rc);
	bool run(Mat img, Rect& rc,Mat outputImg);
	int xGradient(Mat image, int x, int y);
	int yGradient(Mat image, int x, int y);
	void configure();
	void Labeling(const IplImage *src, IplImage *dst);
	Mat get_bp_image();
	Mat m_model;
	MatND m_model3d;
	Mat m_backproj;
	Rect m_rc;
	tracker_opencv_param m_param;
	Mat hsv;
	Mat intensity;
	Point pt1;
	Point pt2;
	Rect roi;
	Mat BirdView;

};
class LineFinder {
private:
	cv::Mat img; // 원 영상
	std::vector<cv::Vec4i> lines; // 선을 감지하기 위한 마지막 점을 포함한 벡터
	double deltaRho;
	double deltaTheta; // 누산기 해상도 파라미터
	int minVote; // 선을 고려하기 전에 받아야 하는 최소 투표 개수
	double minLength; // 선에 대한 최소 길이
	double maxGap; // 선에 따른 최대 허용 간격
	

public:
	Rect m_rc;
	LineFinder() : deltaRho(1), deltaTheta(PI/180), minVote(10), minLength(0.), maxGap(0.) {}
	// 기본 누적 해상도는 1각도 1화소 
	// 간격이 없고 최소 길이도 없음

	// 해당 세터 메소드들

	// 누적기에 해상도 설정
	void setAccResolution(double dRho, double dTheta) {
		deltaRho= dRho;
		deltaTheta= dTheta;
	}

	// 투표 최소 개수 설정
	void setMinVote(int minv) {
		minVote= minv;
	}

	// 선 길이와 간격 설정
	void setLineLengthAndGap(double length, double gap) {
		minLength= length;
		maxGap= gap;
	}

	// 허프 선 세그먼트 감지를 수행하는 메소드
	// 확률적 허프 변환 적용
	std::vector<cv::Vec4i> findLines(cv::Mat& binary) {
		lines.clear();
		cv::HoughLinesP(binary,lines,deltaRho,deltaTheta,minVote, minLength, maxGap);
		return lines;
	} // cv::Vec4i 벡터를 반환하고, 감지된 각 세그먼트의 시작과 마지막 점 좌표를 포함.

	// 위 메소드에서 감지한 선을 다음 메소드를 사용해서 그림
	// 영상에서 감지된 선을 그리기
	void drawDetectedLines(cv::Mat &image, cv::Scalar color=cv::Scalar(0,0,255)) {

		// 선 그리기
		std::vector<cv::Vec4i>::const_iterator it2= lines.begin();
		
		while (it2!=lines.end()) {
			/*cv::Point pt1((*it2)[0] + m_rc.x,(*it2)[1]+ m_rc.y);
			cv::Point pt2((*it2)[2]+ m_rc.x,(*it2)[3]+ m_rc.y);*/
			
			cv::Point pt1((*it2)[0] ,(*it2)[1]);
			cv::Point pt2((*it2)[2],(*it2)[3]);
			cv::line( image, pt1, pt2, color);
			++it2;
		}
	}
};



