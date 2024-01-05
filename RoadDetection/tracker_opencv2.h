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
	cv::Mat img; // �� ����
	std::vector<cv::Vec4i> lines; // ���� �����ϱ� ���� ������ ���� ������ ����
	double deltaRho;
	double deltaTheta; // ����� �ػ� �Ķ����
	int minVote; // ���� ����ϱ� ���� �޾ƾ� �ϴ� �ּ� ��ǥ ����
	double minLength; // ���� ���� �ּ� ����
	double maxGap; // ���� ���� �ִ� ��� ����
	

public:
	Rect m_rc;
	LineFinder() : deltaRho(1), deltaTheta(PI/180), minVote(10), minLength(0.), maxGap(0.) {}
	// �⺻ ���� �ػ󵵴� 1���� 1ȭ�� 
	// ������ ���� �ּ� ���̵� ����

	// �ش� ���� �޼ҵ��

	// �����⿡ �ػ� ����
	void setAccResolution(double dRho, double dTheta) {
		deltaRho= dRho;
		deltaTheta= dTheta;
	}

	// ��ǥ �ּ� ���� ����
	void setMinVote(int minv) {
		minVote= minv;
	}

	// �� ���̿� ���� ����
	void setLineLengthAndGap(double length, double gap) {
		minLength= length;
		maxGap= gap;
	}

	// ���� �� ���׸�Ʈ ������ �����ϴ� �޼ҵ�
	// Ȯ���� ���� ��ȯ ����
	std::vector<cv::Vec4i> findLines(cv::Mat& binary) {
		lines.clear();
		cv::HoughLinesP(binary,lines,deltaRho,deltaTheta,minVote, minLength, maxGap);
		return lines;
	} // cv::Vec4i ���͸� ��ȯ�ϰ�, ������ �� ���׸�Ʈ�� ���۰� ������ �� ��ǥ�� ����.

	// �� �޼ҵ忡�� ������ ���� ���� �޼ҵ带 ����ؼ� �׸�
	// ���󿡼� ������ ���� �׸���
	void drawDetectedLines(cv::Mat &image, cv::Scalar color=cv::Scalar(0,0,255)) {

		// �� �׸���
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



