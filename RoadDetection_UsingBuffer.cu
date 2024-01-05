#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <time.h>
using namespace std;
using namespace cv;

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

const string trackbarWindowName = "Trackbars";

double work_fps = 0;
double work_begin = 0;

void workBegin() { work_begin = getTickCount(); }

void workEnd()
{
	int64 delta = getTickCount() - work_begin;
	double freq = getTickFrequency();
    work_fps = freq / delta;
}


#define drawCross( img, center, color, d )\
	line(img, Point(center.x - d, center.y - d), Point(center.x + d, center.y + d), color, 2, CV_AA, 0);\
	line(img, Point(center.x + d, center.y - d), Point(center.x - d, center.y + d), color, 2, CV_AA, 0 )\

void gettingS(const cv::Mat& img_out, const cv::Mat& img_in, uchar *gpu_in, uchar *gpu_out, int in_cols, int rows, int out_cols, int out_rows, int in_step, int out_step);
__global__ void gettingS_kernel(unsigned char *img_out, unsigned char * img_in, int in_cols, int rows, int out_cols, int out_rows, int in_step, int out_step);
void raisingFromTh(const cv::Mat& img_out, const cv::Mat& img_in, uchar *gpu_in, uchar *gpu_out, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step);
__global__ void raisingFromTh_kernel(unsigned char *img_out, unsigned char *img_in, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step);
__global__ void otsu_kernel(unsigned char *img_out, unsigned char * img_in, int in_cols,int in_rows, int out_cols, int out_rows, int in_step, int out_step);
void finalFrame(const cv::Mat& img_out, const cv::Mat& img_in, uchar *gpu_in, uchar *gpu_out, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step);
__global__ void finalFrame_kernel(unsigned char *img_out, unsigned char * img_in, int in_cols,int in_rows, int out_cols, int out_rows, int in_step, int out_step);


void gettingS(const cv::Mat& img_out, const cv::Mat& img_in, uchar *gpu_in, uchar *gpu_out, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	cudaMemcpy( gpu_in, img_in.data, sizeof(uchar)*in_cols*in_rows, cudaMemcpyHostToDevice);
	
	//GPU kernel calling
	dim3 nBlocks;
	dim3 nThreads;
	
	nThreads.x = BLOCK_WIDTH; nThreads.y = BLOCK_HEIGHT;
	nBlocks.x = (img_in.cols+nThreads.x-1)/nThreads.x; nBlocks.y = (img_in.cols+nThreads.y-1)/nThreads.y; 

	gettingS_kernel<<< nBlocks, nThreads >>>(gpu_out, gpu_in,in_cols, in_rows, out_cols, out_rows, in_step, out_step);
	
	cudaMemcpy(img_out.data, gpu_out, sizeof(uchar)*out_cols*out_rows, cudaMemcpyDeviceToHost);
}

__global__ void gettingS_kernel(unsigned char *img_out, unsigned char * img_in, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	int y = blockIdx.y*blockDim.y+threadIdx.y; //y
	int x = blockIdx.x*blockDim.x+threadIdx.x; //x

	if(x > out_cols || y > out_rows)
		return;
	img_out[(y*out_cols)+(x)] = img_in[(y*in_cols)+(x)];

	return;
}


void raisingFromTh(const cv::Mat& img_out, const cv::Mat& img_in, uchar *gpu_in, uchar *gpu_out, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	cudaMemcpy( gpu_in, img_in.data, sizeof(uchar)*in_cols*in_rows, cudaMemcpyHostToDevice);

	//GPU kernel calling
	dim3 nBlocks;
	dim3 nThreads;
	
	nThreads.x = BLOCK_WIDTH; nThreads.y = BLOCK_HEIGHT;
	nBlocks.x = (img_in.cols+nThreads.x-1)/nThreads.x; nBlocks.y = (img_in.cols+nThreads.y-1)/nThreads.y; 
		
	raisingFromTh_kernel<<< nBlocks, nThreads >>>(gpu_out, gpu_in, in_cols,in_rows, out_cols, out_rows, in_step, out_step);
	otsu_kernel<<< nBlocks, nThreads >>>(gpu_out, gpu_in, in_cols,in_rows, out_cols, out_rows, in_step, out_step);

	cudaMemcpy(img_out.data, gpu_out, sizeof(uchar)*out_cols*out_rows, cudaMemcpyDeviceToHost);
}

__global__ void raisingFromTh_kernel(unsigned char *img_out, unsigned char *img_in, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	int y = blockIdx.y*blockDim.y+threadIdx.y; //y
	int x = blockIdx.x*blockDim.x+threadIdx.x; //x

	y=in_rows-120-y;	
	if(x > out_cols || y > out_rows || y < 0)
		return;
	if(y-2>0)
	{
		if(abs(img_in[(y*in_cols)+(x)]-img_in[((y-1)*in_cols)+(x)]) >= 90)
		{
			for(int k = y; k > 0; k--)
			{
				img_out[(k*out_cols)+(x)] = 255;
			}
		}
	}

	return;
}

__global__ void otsu_kernel(unsigned char *img_out, unsigned char * img_in, int in_cols,int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	int y = blockIdx.y*blockDim.y+threadIdx.y; //y
	int x = blockIdx.x*blockDim.x+threadIdx.x; //x

	if(x > out_cols || y > out_rows)
		return;

	if(x < out_cols-1 && y < 140)
		img_out[(y*out_cols)+(x)] = 255;		

	return;
}

void finalFrame(const cv::Mat& img_out, const cv::Mat& img_in, uchar *gpu_in, uchar *gpu_out, int in_cols, int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	cudaMemcpy( gpu_in, img_in.data, sizeof(uchar)*in_cols*in_rows, cudaMemcpyHostToDevice);
	cudaMemcpy( gpu_out, img_out.data, sizeof(uchar)*in_cols*in_rows*3, cudaMemcpyHostToDevice);

	//GPU kernel calling
	dim3 nBlocks;
	dim3 nThreads;
	
	nThreads.x = BLOCK_WIDTH; nThreads.y = BLOCK_HEIGHT;
	nBlocks.x = (img_in.cols+nThreads.x-1)/nThreads.x; nBlocks.y = (img_in.cols+nThreads.y-1)/nThreads.y; 
		
	finalFrame_kernel<<< nBlocks, nThreads >>>(gpu_out, gpu_in, in_cols,in_rows, out_cols, out_rows, in_step, out_step);
	

	cudaMemcpy(img_out.data, gpu_out, sizeof(uchar)*out_cols*out_rows*3, cudaMemcpyDeviceToHost);
}

__global__ void finalFrame_kernel(unsigned char *img_out, unsigned char * img_in, int in_cols,int in_rows, int out_cols, int out_rows, int in_step, int out_step)
{
	int y = blockIdx.y*blockDim.y+threadIdx.y; //y
	int x = blockIdx.x*blockDim.x+threadIdx.x; //x



	y=in_rows-60-y;	
	if(x > out_cols-1 || y > out_rows || y < 0)
		return;

	if(img_in[(y*in_cols)+(x)] != 255)
		img_out[(y*out_cols*3)+(x*3)+2] = 180;

	return;
}

int main()
{
	
	vector<cv::Mat> imageVec;
	vector<cv::Mat>::iterator imageVecIter;

	Mat frame, thresh_frame,hsv, temp_img;
	vector<Mat> channels;
	VideoCapture capture;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	cv::Mat input_img; // input

	uchar *cuda_src;
	uchar *cuda_s;
	uchar *cuda_thres;
	uchar *cuda_sisThres;	
	uchar *cuda_frame_src;
	uchar *cuda_frame;



	//input_img = imread("Z:/전방영상/전방 카메라 장착 VGA/2/캡쳐/shadow.jpg");
	//capture.open("Z:/후방차량영상/vga/vga 2/REC_0003.avi");
	
	capture.open("D:/a3.avi");
	//E:\Temptest\전방영상\전방 카메라장착 VGA
	
	if(!capture.isOpened())
		cerr << "Problem opening video source" << endl;

	KalmanFilter KF(4, 2, 0);
	Mat_<float> state(4, 1);
	Mat_<float> processNoise(4, 1, CV_32F);
	Mat_<float> measurement(2,1);   measurement.setTo(Scalar(0));

	KF.statePre.at<float>(0) = 0; 
	KF.statePre.at<float>(1) = 0;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;

	KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
	KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));
	 //time_t start, end;
	  double fps;
	
	  // frame counter
  int counter = 0;
  double a1_sum=0,a2_sum=0,a3_sum=0;
  double fps_sum = 0;
  double fps_avg = 0;
  double sec;
  
  clock_t t1,t2,t3;
  int r1=0,r2=0,r3=0;
 
	while((char)waitKey(1) != 'q' && capture.grab())
	{		
		capture.retrieve(temp_img);
		imageVec.push_back(temp_img.clone());					
	}	
	imageVecIter = imageVec.begin();
	
	while((char)waitKey(1) != 'q' && imageVecIter != imageVec.end())
	{
		workBegin();
		cv::Mat dst;
		
		frame = *imageVecIter;
		//capture.retrieve(frame);
		//frame = input_img;
		//imshow("input",frame);
		
		
	//	 time(&end);
	   ++counter;        
//      sec = difftime (end, start);      
       
 //     fps = getTickFrequency() / sec; // fps계산 변경됨 사용안함 (workEnd()에서 최종계산)
	  //fps = counter / sec;
 
      // will print out Inf until sec is greater than 0
	  printf("FPS = %.2f\n", work_fps);
	  fps_sum += work_fps;
      //printf("FPS = %.2f\n", fps);

		cvtColor(frame,thresh_frame,COLOR_BGR2GRAY);

		Canny(thresh_frame,thresh_frame,100,200,3);//Detect Edges.
		
		cvtColor(frame, hsv, CV_BGR2HSV);
		float hrange[] = {0,180};
		float vrange[] = {0,255};
		const float* ranges[] = {hrange, vrange, vrange};	// hue, saturation, brightness
		int channels[] = {0, 1, 2};
		//calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);
		cv::Mat v(frame.rows,frame.cols,CV_8UC1);
		int rows = hsv.rows;
		int cols = hsv.cols;
		
		// s추출
		t1 = clock();
		vector<Mat> hsv_channels;
		cv::split(hsv, hsv_channels);
		//v = hsv_channels[1]; //사실 s  fps 2정도 개선	
		cudaMalloc((void**)&cuda_src, sizeof(uchar) * hsv_channels[1].cols * hsv_channels[1].rows);
		cudaMalloc((void**)&cuda_s, sizeof(uchar) * hsv_channels[1].cols * hsv_channels[1].rows);
		cudaMemset(cuda_s,0,sizeof(uchar) * hsv_channels[1].cols * hsv_channels[1].rows);
		
		gettingS(v, hsv_channels[1],cuda_src, cuda_s, hsv_channels[1].cols, hsv_channels[1].rows, v.cols, v.rows, hsv_channels[1].step1(),v.step1());
		r1++;
		double result = (double)(clock()-t1)/CLOCKS_PER_SEC;
		a1_sum += result;
		printf("algo1 = %.2f  ", result);
		
		//		imshow("v",v);

		//------------- 2번째과정 ---------------
		t2 = clock();
		cv::Mat ImageThresh; 
		double thresh1 = cv::threshold(v, ImageThresh, 50, 120.0, cv::THRESH_BINARY+cv::THRESH_OTSU); 
		//medianBlur(thresh_frame, ImageThresh, 5);
			
//		imshow("thresh",ImageThresh);
		Mat Imgshow(ImageThresh.rows,ImageThresh.cols,CV_8UC1);
		Canny(Imgshow,Imgshow,100,200,3);//Detect Edges.	
		
		

		cudaMalloc((void**)&cuda_thres, sizeof(uchar) * ImageThresh.cols * ImageThresh.rows);
		cudaMalloc((void**)&cuda_sisThres, sizeof(uchar) * Imgshow.cols * Imgshow.rows);
		cudaMemset(cuda_sisThres,0,sizeof(uchar)*Imgshow.cols*Imgshow.rows);

		raisingFromTh(Imgshow, ImageThresh, cuda_thres, cuda_sisThres, ImageThresh.cols, ImageThresh.rows, Imgshow.cols, Imgshow.rows, ImageThresh.step1(), Imgshow.step1());

	
		//imshow("sis threshold2",Imgshow);
		r2++;
		result = (double)(clock()-t2)/CLOCKS_PER_SEC;
		a2_sum += result;
		printf("algo2 = %.2f  ", result);

		//-------------- 3번째과정 -----------------

		t3 = clock();
		//Canny(Imgshow,Imgshow,100,200,3);//Detect Edges.
		//imshow("Asdfasdf",Imgshow);

		thresh_frame = Imgshow;


		//medianBlur(thresh_frame, thresh_frame, 5);

		findContours(thresh_frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		Mat drawing = Mat::zeros(thresh_frame.size(), CV_8UC1);

		//안씀
		for(size_t i = 0; i < contours.size(); i++)
		{
			//          cout << contourArea(contours[i]) << endl;
			if(contourArea(contours[i]) > 500)
				drawContours(drawing, contours, i, Scalar::all(255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());
		}
		thresh_frame = drawing;     

		// Get the moments
		vector<Moments> mu(contours.size() );
		for( size_t i = 0; i < contours.size(); i++ )
		{ mu[i] = moments( contours[i], false ); }

		//  Get the mass centers:
		vector<Point2f> mc( contours.size() );
		for( size_t i = 0; i < contours.size(); i++ )
		{ mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

		Mat prediction = KF.predict();
		Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

		//안씀
		for(size_t i = 0; i < mc.size(); i++)
		{
			drawCross(frame, mc[i], Scalar(255, 0, 0), 5);//Scalar is color for predicted cross mark on identified object.
			measurement(0) = mc[i].x;
			measurement(1) = mc[i].y;
		}

		Point measPt(measurement(0),measurement(1));
		Mat estimated = KF.correct(measurement);
		Point statePt(estimated.at<float>(0),estimated.at<float>(1));

		drawCross(frame, statePt, Scalar(255, 255, 255), 5);//Scalar is color for another correct cross mark on identified object.

		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );
		for( size_t i = 0; i < contours.size(); i++ )
		{ approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		}

		for( size_t i = 0; i < contours.size(); i++ )
		{
			//rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );//Scalar is for Bounding Box
		}

		//추출영역 색상변경
	

		cudaMalloc((void**)&cuda_frame_src, sizeof(uchar) * thresh_frame.cols * thresh_frame.rows);
		cudaMalloc((void**)&cuda_frame, sizeof(uchar) * thresh_frame.cols * thresh_frame.rows*3);
		cudaMemset(cuda_frame, 0,sizeof(uchar) * thresh_frame.cols * thresh_frame.rows *3);
		Mat final_outframe(frame);		

		finalFrame(final_outframe,thresh_frame, cuda_frame_src, cuda_frame, thresh_frame.cols, thresh_frame.rows, final_outframe.cols, final_outframe.rows, thresh_frame.step1(),final_outframe.step1());
		

		r3++;
		result = (double)(clock()-t2)/CLOCKS_PER_SEC;
		a3_sum += result;
		printf("algo3 = %.2f  \n", result);

		//imshow("Video", final_outframe);  
		imageVecIter++;
		
		workEnd();
		cudaFree(cuda_src);
		cudaFree(cuda_s);
		cudaFree(cuda_thres);
		cudaFree(cuda_sisThres);
		cudaFree(cuda_frame_src);
		cudaFree(cuda_frame);
	}
	fps_avg=fps_sum/counter;
	printf("fps_average = %.2f, number of frames = %d   \n algo1_ave = %.6f . algo2_ave = %.6f  . algo2_avg = %.6f  \n",fps_avg,counter,a1_sum/r1,a2_sum/r2,a3_sum/r3);
	return 0;
}
