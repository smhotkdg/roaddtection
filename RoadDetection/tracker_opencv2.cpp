///////////////////////////////////////////////////////////////////////
// OpenCV tracking example.

//#include "stdafx.h"
#include <windows.h>
#include <iostream>
#include <cstdlib>

#include <math.h>
#include "tracker_opencv2.h"
#include "RegionGrowing.h"
int kernel_size=21;
int pos_sigma= 5;
int pos_lm = 50;
int pos_th = 0;
int pos_psi = 90;
cv::Mat src_f;
cv::Mat dest;
cv::Mat v;
int count_t = 0;
double a[33] = {0,};
const int count_arry = 30;

int tempcount = 30;
IplImage *image = 0 ;   

void SetCurve()
{
	a[0] = 1.0;
	a[1] = 1.0;
	a[2] = 2.0;
	a[3] = 6.0;
	a[4] = 24.0;
	a[5] = 120.0;
	a[6] = 720.0;
	a[7] = 5040.0;
	a[8] = 40320.0;
	a[9] = 362880.0;
	a[10] = 3628800.0;
	a[11] = 39916800.0;
	a[12] = 479001600.0;
	a[13] = 6227020800.0;
	a[14] = 87178291200.0;
	a[15] = 1307674368000.0;
	a[16] = 20922789888000.0;
	a[17] = 355687428096000.0;
	a[18] = 6402373705728000.0;
	a[19] = 121645100408832000.0;
	a[20] = 2432902008176640000.0;
	a[21] = 51090942171709440000.0;
	a[22] = 1124000727777607680000.0;
	a[23] = 25852016738884976640000.0;
	a[24] = 620448401733239439360000.0;
	a[25] = 15511210043330985984000000.0;
	a[26] = 403291461126605635584000000.0;
	a[27] = 10888869450418352160768000000.0;
	a[28] = 304888344611713860501504000000.0;
	a[29] = 8841761993739701954543616000000.0;
	a[30] = 265252859812191058636308480000000.0;
	a[31] = 8222838654177922817725562880000000.0;
	a[32] = 263130836933693530167218012160000000.0;
}

double factorial(int n)
{
	return a[n]; /* returns the value n! as a SUMORealing point number */
}

double Ni(int n, int i)
{
	double ni=0;
	double a1 = factorial(n);
	double a2 = factorial(i);
	double a3 = factorial(n - i);
	ni =  a1/ (a2 * a3);
	return ni;
}
// Calculate Bernstein basis
double Bernstein(int n, int i, double t)
{
	double basis;
	double ti; /* t^i */
	double tni; /* (1 - t)^i */

	/* Prevent problems with pow */

	if (t == 0.0 && i == 0) 
		ti = 1.0; 
	else 
		ti = pow(t, i);

	if (n == i && t == 1.0) 
		tni = 1.0; 
	else 
		tni = pow((1 - t), (n - i));

	//Bernstein basis
	basis = Ni(n, i) * ti * tni; 
	return basis;
}

void Bezier2D(double b[], int cpts, double p[])
{	
	int npts = tempcount;
	int icount, jcount;
	double step, t;

	// Calculate points on curve

	icount = 0;
	t = 0;
	step = (double)1.0 / (cpts - 1);

	for (int i1 = 0; i1 != cpts; i1++)
	{ 
		if ((1.0 - t) < 5e-6) 
			t = 1.0;

		jcount = 0;
		p[icount] = 0.0;
		p[icount + 1] = 0.0;
		for (int i = 0; i != npts; i++)
		{
			double basis = Bernstein(npts - 1, i, t);
			p[icount] += basis * b[jcount];
			p[icount + 1] += basis * b[jcount + 1];
			jcount = jcount +2;
		}

		icount += 2;
		t += step;
	}
}

void BezierDraw(IplImage* image, double points[])  
{  
	CvPoint pt_pre=cvPoint(points[0],points[1]);  
	CvPoint pt_now;  
	int countx = 1;
	int county = 1;
	int precision = 500;  
	for (int i=1;i<=precision;i++)   
	{  
		countx = i*2 -1;
		county = i*2;
		float u = (float)i/precision;  


		pt_now.x = points[countx-1];  
		pt_now.y = points[county-1];

		if(i>0) cvLine(image,pt_now,pt_pre,CV_RGB(255,0,0),3,CV_AA, 0 );  
		pt_pre = pt_now;  
	}  
	//DrawControlLine2(points);  
}  
Mat correctGamma( Mat& img, double gamma ) {
	double inverse_gamma = 1.0 / gamma;

	Mat lut_matrix(1, 256, CV_8UC1 );
	uchar * ptr = lut_matrix.ptr();
	for( int i = 0; i < 256; i++ )
		ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

	Mat result;
	LUT( img, lut_matrix, result );

	return result;
}
int tracker_opencv::xGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y-1, x-1) +
		2*image.at<uchar>(y, x-1) +
		image.at<uchar>(y+1, x-1) -
		image.at<uchar>(y-1, x+1) -
		2*image.at<uchar>(y, x+1) -
		image.at<uchar>(y+1, x+1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int tracker_opencv::yGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y-1, x-1) +
		2*image.at<uchar>(y-1, x) +
		image.at<uchar>(y-1, x+1) -
		image.at<uchar>(y+1, x-1) -
		2*image.at<uchar>(y+1, x) -
		image.at<uchar>(y+1, x+1);
}


cv::Mat mkKernel(int ks, double sig, double th, double lm, double ps)
{
	int hks = (ks-1)/2;
	double theta = th*CV_PI/180;
	double psi = ps*CV_PI/180;
	double del = 2.0/(ks-1);
	double lmbd = lm;
	double sigma = sig/ks;
	double x_theta;
	double y_theta;
	cv::Mat kernel(ks,ks, CV_32F);
	for (int y=-hks; y<=hks; y++)
	{
		for (int x=-hks; x<=hks; x++)
		{
			x_theta = x*del*cos(theta)+y*del*sin(theta);
			y_theta = -x*del*sin(theta)+y*del*cos(theta);
			kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
		}
	}
	return kernel;
}

cv::Mat Process(int , void *)
{
	double sig = pos_sigma;
	double lm = 0.5+pos_lm/100.0;
	double th = pos_th;
	double ps = pos_psi;
	cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
	cv::filter2D(src_f, dest, CV_32F, kernel);
	cv::imshow("Process window", dest);
	cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
	cv::resize(kernel, Lkernel, Lkernel.size());
	Lkernel /= 2.;
	Lkernel += 0.5;
	cv::imshow("Kernel", Lkernel);
	cv::Mat mag;
	cv::pow(dest, 2.0, mag);
	cv::imshow("Mag", mag);
	return mag;
}
void Process2(int , void *)
{
	double sig = pos_sigma;
	double lm = 0.5+pos_lm/100.0;
	double th = pos_th;
	double ps = pos_psi;
	cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
	cv::filter2D(src_f, dest, CV_32F, kernel);
	//cv::imshow("Process window", dest);
	cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
	cv::resize(kernel, Lkernel, Lkernel.size());
	Lkernel /= 2.;
	Lkernel += 0.5;
	//cv::imshow("Kernel", Lkernel);
	cv::Mat mag;
	cv::pow(dest, 2.0, mag);
	cv::imshow("Mag", mag);
}


using namespace std;

tracker_opencv::tracker_opencv(void)
{
}

tracker_opencv::~tracker_opencv(void)
{
}

void tracker_opencv::init(Mat img, Rect rc)
{

	Mat mask = Mat::zeros(rc.height, rc.width, CV_8U);
	ellipse(mask, Point(rc.width/2, rc.height/2), Size(rc.width/2, rc.height/2), 0, 0, 360, 255, CV_FILLED);

	if(img.channels()<=1)
	{
		float vrange[] = {0,256};
		const float* phranges = vrange;
		Mat roi(img, rc);
		calcHist(&roi, 1, 0, mask, m_model, 1, &m_param.hist_bins, &phranges);
	}
	else if(m_param.color_model==CM_GRAY)
	{
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);

		float vrange[] = {0,256};
		const float* phranges = vrange;
		Mat roi(gray, rc);
		calcHist(&roi, 1, 0, mask, m_model, 1, &m_param.hist_bins, &phranges);
	}
	else if(m_param.color_model==CM_HUE)
	{
		Mat hue;
		cvtColor(img, hsv, CV_BGR2YCrCb);

		float hrange[] = {0,180};
		const float* phranges = hrange;
		int channels[] = {0};
		Mat roi(hue, rc);
		calcHist(&roi, 1, channels, mask, m_model, 1, &m_param.hist_bins, &phranges);
	}
	else if(m_param.color_model==CM_RGB)
	{
		float vrange[] = {0,255};
		const float* ranges[] = {vrange, vrange, vrange};	// B,G,R
		int channels[] = {0, 1, 2};
		int hist_sizes[] = {m_param.hist_bins, m_param.hist_bins, m_param.hist_bins};
		Mat roi(img, rc);
		calcHist(&roi, 1, channels, mask, m_model3d, 3, hist_sizes, ranges);
	}
	else if(m_param.color_model==CM_HSV)
	{
		Mat hsv;
		cvtColor(img, hsv, CV_BGR2HSV);

		float hrange[] = {0,180};
		float vrange[] = {0,255};
		const float* ranges[] = {hrange, vrange, vrange};	// hue, saturation, brightness

		int channels[] = {0, 1, 2};
		int hist_sizes[] = {m_param.hist_bins, m_param.hist_bins, m_param.hist_bins};
		Mat roi(hsv, rc);
		calcHist(&roi, 1, channels, mask, m_model3d, 3, hist_sizes, ranges);
	}

	m_rc = rc;
}

//도로 검출 시작부분
bool tracker_opencv::run(Mat img, Rect& rc,Mat outPutimg)
{
	//컬러모델선택
	// elliptic mask
	BirdView = outPutimg;
	Mat mask = Mat::zeros(rc.height, rc.width, CV_8U);
	ellipse(mask, Point(rc.width/2, rc.height/2), Size(rc.width/2, rc.height/2), 0, 0, 360, 255, CV_FILLED);

	// histogram backprojection
	if(img.channels()<=1)
	{
		float vrange[] = {0,256};
		const float* phranges = vrange;
		calcBackProject(&img, 1, 0, m_model, m_backproj, &phranges);
		imshow("image3", m_backproj);

	}
	else if(m_param.color_model==CM_GRAY)
	{
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);

		float vrange[] = {0,256};
		const float* phranges = vrange;
		calcBackProject(&gray, 1, 0, m_model, m_backproj, &phranges);
		imshow("image3", gray);

	}
	else if(m_param.color_model==CM_HUE)
	{
		Mat hue;
		cvtColor(img, hue, CV_BGR2HSV);

		float hrange[] = {0,180};
		const float* phranges = hrange;
		int channels[] = {0};
		calcBackProject(&hsv, 1, channels, m_model, m_backproj, &phranges);
		imshow("image3", hue);
	}
	else if(m_param.color_model==CM_RGB)
	{
		float vrange[] = {0,255};
		const float* ranges[] = {vrange, vrange, vrange};	// B,G,R
		int channels[] = {0, 1, 2};
		int hist_sizes[] = {m_param.hist_bins, m_param.hist_bins, m_param.hist_bins};
		calcBackProject(&img, 1, channels, m_model3d, m_backproj, ranges);
		imshow("image3", img);

	}
	else if(m_param.color_model==CM_HSV)
	{

		cvtColor(img, hsv, CV_BGR2HSV);

		float hrange[] = {0,180};
		float vrange[] = {0,255};
		const float* ranges[] = {hrange, vrange, vrange};	// hue, saturation, brightness
		int channels[] = {0, 1, 2};
		calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);


		cv::Mat v(img.rows,img.cols,CV_8UC1);
		int rows = hsv.rows;
		int cols = hsv.cols;

		for(int i =0; i< rows; i++){
			for(int j =0; j < cols; j++){			
				uchar bgrPixel = hsv.at<uchar>(i,j);
				v.at<uchar>(i,j) = hsv.at<Vec3b>(i,j)[1];

			}
		}		

		intensity = v.clone();

	}
	///////***************************
	else if(m_param.color_model==CM_LAB)
	{
		Mat lab;
		cvtColor(img, lab, CV_BGR2Lab);

		float hrange[] = {0,180};
		float vrange[] = {0,255};
		const float* ranges[] = {hrange, vrange, vrange};	// hue, saturation, brightness
		int channels[] = {0, 1, 2};
		calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);

		imshow("image3", lab);
	}

	else if(m_param.color_model==CM_LUV)
	{

		Mat luv;
		cvtColor(img, luv, CV_BGR2Luv);

		float hrange[] = {0,180};
		const float* phranges = hrange;
		int channels[] = {0};
		calcBackProject(&luv, 1, channels, m_model, m_backproj, &phranges);
		imshow("image3", luv);
	}


	// 도로 검출
	if(m_param.method == ROAD_DETECTION)
	{

		//sis threshold
		cv::Mat Image1Thresh; 
		double thresh1 = cv::threshold(intensity, Image1Thresh, 0.0, 120.0, cv::THRESH_BINARY+cv::THRESH_OTSU); 
		imshow("sis threshold2",Image1Thresh);

		Mat src, dst;
		int gx, gy, sum;
		Mat gray;
		cv::cvtColor(img,gray,CV_BGR2GRAY);


		Mat sobel;
		Mat sobelX;
		Mat sobelY;
		Sobel(gray, sobelX, CV_8U, 1, 0);
		Sobel(gray, sobelY, CV_8U, 0, 1);
		sobel = abs(sobelX) + abs(sobelY);



		Mat sobel_draw2(Image1Thresh.rows,Image1Thresh.cols,CV_8UC1);


		Mat sobel_draw(Image1Thresh.rows,Image1Thresh.cols,CV_8UC1);


		for(int i =0; i < Image1Thresh.cols; i++){
			for(int j = Image1Thresh.rows-100 ; j > 0 ; j--){
				if(j-2 > 0){
					if(abs(Image1Thresh.at<unsigned char>(j,i) -Image1Thresh.at<unsigned char>(j-1,i)) >=90 && 
						abs(Image1Thresh.at<unsigned char>(j,i) -Image1Thresh.at<unsigned char>(j-1,i)) >=90 
						) 
					{						
						for(int k = j; k > 0; k--){
							sobel_draw.at<unsigned char>(k,i) =255; 										
						}
					}
				}
			}
		}
		int tempnum = sobel_draw.cols/ tempcount;
		int setnumber = sobel_draw.cols /tempnum;
		int count = 0;
		int arrcount =1;
		double points[count_arry*2];  
		int county =1;
		int countx =1;
		for(int i =0; i < sobel_draw.cols; i++){
			for(int j = sobel_draw.rows-150 ; j > 0 ; j--){
				uchar bgrPixel = sobel_draw.at<uchar>(j,i);
				if(i % tempnum ==0){
					if(bgrPixel == 255)
					{
						if( i != 0 && j !=0){
							circle( img, Point( i, j ), 2,  Scalar(250), 0, 255, 0 );
							//if(count <= tempcount*2){
								count ++;
								county = count*2;
								countx = 2*count -1;
								points[countx-1] = i;
								points[county-1] = j;
							//}						
						}
						break;
					}
				}
			}
		}
		cv::Mat tempImage = img.clone();
		IplImage* image2=cvCloneImage(&(IplImage)tempImage);

		double p[1000] = {0.0};
		SetCurve();


		if(count !=30 && count !=0){
			tempcount = count;
			double *pointresult = new double[count*2-1];			
			int recounty =1;
			int recountx =1;
			for(int i=1; i<= count;i++){
				
				recounty = i*2;
				recountx = 2*i -1;
				pointresult [recountx-1] = points[recountx-1];
				pointresult [recounty-1] = points[recounty-1];
			}

			Bezier2D(pointresult,500,p);
			
		}
		else
		{
			Bezier2D(points,500,p);
			
		}
		BezierDraw(image2,p);


		cv::Mat compare_img = sobel_draw;
		cv::Mat result_img = img.clone();
		int rows = sobel_draw.rows;
		int cols = sobel_draw.cols;
		cv::Mat resultimg = img.clone();
		for(int i =150; i< rows; i++){
			for(int j =0; j < cols; j++){			
				uchar bgrPixel = sobel_draw.at<uchar>(i,j);

				if(bgrPixel != 255){
					compare_img.at<uchar>(i,j) = 0;						
				}
			}
		}
		for(int i=150; i< rows; i++){
			for(int j =0; j < cols; j++){			
				uchar bgrPixel = compare_img.at<uchar>(i,j);

				if(bgrPixel != 255){
					resultimg.at<Vec3b>(i,j)[2] = 150;						
				}
			}
		}


		Mat aaa(image2,false);
		//imshow("sobel",sobel);
		imshow("road detection",sobel_draw);
		//imshow("road detection2",sobel_draw2);
		imshow("resultimg",resultimg);
		imshow("Line Detection",aaa);
	}

	if(m_param.method == MEANSHIFT)
	{
		if(m_rc.width ==0){
			return 0;
		}
		//int itrs = meanShift(m_backproj, m_rc, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, m_param.max_itrs, 1 ));
		////rectangle(img, m_rc, Scalar(0,0,255), 3, CV_AA);
		//imshow("image2", m_backproj);


		//cv::Mat compare_img = m_backproj;
		//cv::Mat result_img = img.clone();
		//int rows = m_backproj.rows;
		//int cols = m_backproj.cols;

		//for(int i =0; i< rows; i++){
		//	for(int j =0; j < cols; j++){			
		//		uchar bgrPixel = m_backproj.at<uchar>(i,j);

		//		if(bgrPixel != 255){
		//			compare_img.at<uchar>(i,j) = 0;
		//		}
		//		else{
		//			compare_img.at<uchar>(i,j) = 255;	
		//			result_img.at<Vec3b>(i,j)[2] = 255;
		//		}
		//	}
		//}			

		//imshow("compare_image", compare_img);
		//imshow("result", result_img);
		//Mat src,sample;


		///// Separate the image in 3 places ( B, G and R )
		//cvtColor(sample,src,CV_BGR2HSV);
		//vector<Mat> bgr_planes;
		//split( src, bgr_planes );

		///// Establish the number of bins
		//int histSize = 256;

		///// Set the ranges ( for B,G,R) )
		//float range[] = { 0, 180 } ;
		//const float* histRange = { range };

		//bool uniform = true; bool accumulate = false;

		//Mat h_hist, s_hist, v_hist;

		///// Compute the histograms:
		//calcHist( &bgr_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );
		//calcHist( &bgr_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate );
		//calcHist( &bgr_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate );

		//// Draw the histograms for B, G and R
		//int hist_w = 512; int hist_h = 400;
		//int bin_w = cvRound( (double) hist_w/histSize );

		//Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

		///// Normalize the result to [ 0, histImage.rows ]
		//normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		//normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		//normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

		///// Draw for each channel
		//for( int i = 1; i < histSize; i++ )
		//{
		//	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ) ,
		//		Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
		//		Scalar( 255, 0, 0), 2, 8, 0  );
		//	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(s_hist.at<float>(i-1)) ) ,
		//		Point( bin_w*(i), hist_h - cvRound(s_hist.at<float>(i)) ),
		//		Scalar( 0, 255, 0), 2, 8, 0  );
		//	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(v_hist.at<float>(i-1)) ) ,
		//		Point( bin_w*(i), hist_h - cvRound(v_hist.at<float>(i)) ),
		//		Scalar( 0, 0, 255), 2, 8, 0  );
		//}

		///// Display
		//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
		//imshow("calcHist Demo", histImage );

		//compareHist


	}
	else if(m_param.method == CAMSHIFT)
	{
		if(m_rc.width>0 && m_rc.height>0)
		{
			/*RotatedRect trackBox = CamShift(m_backproj, m_rc, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, m_param.max_itrs, 1));
			ellipse( img, trackBox, Scalar(0,0,255), 3, CV_AA );
			imshow("image2", m_backproj);*/
		}	

		if(m_rc.width<=1 || m_rc.height<=1)
		{
			int cols = m_backproj.cols, rows = m_backproj.rows, r = (MIN(cols, rows) + 5)/6;
			m_rc = Rect(m_rc.x-r, m_rc.y-r, m_rc.width+2*r, m_rc.height+2*r) & Rect(0, 0, cols, rows);
		}
	}

	rc = m_rc;

	return true;
}

void tracker_opencv::configure()
{
	char sel = -1;
	//cout << "  1. camshift\n"
	//	<< "  2. meanshift\n"
	//	<< "  3. Sobel\n";
	//cout << "select tracking method[1-3]: ";
	//cin >> sel;
	//cout << endl;
	sel =3;

	if(sel=='1')
		m_param.method = CAMSHIFT;
	else if(sel=='2')
		m_param.method = MEANSHIFT;
	else if(sel = '3')
		m_param.method = ROAD_DETECTION;

	//cout << "  1. HSV\n"
	//	<< "  2. RGB\n"
	//	<< "  3. hue\n"
	//	<< "  4. gray\n"
	//	<< "  5. LAB\n "
	//	<< " 6. LUV \n ";
	//cout << "select color model[1-6]: ";
	//cin >> sel;
	//cout << endl;
	sel =1;

	if(sel=='1')
		m_param.color_model = CM_HSV;
	else if(sel=='2')
		m_param.color_model = CM_RGB;
	else if(sel=='3')
		m_param.color_model = CM_HUE;
	else if(sel=='4')
		m_param.color_model = CM_GRAY;
	else if(sel=='5')
		m_param.color_model = CM_LAB;
	else if(sel=='6')
		m_param.color_model = CM_LUV;

}

Mat tracker_opencv::get_bp_image()
{
	normalize(m_backproj, m_backproj, 0, 255, CV_MINMAX);
	return m_backproj;
}

void    tracker_opencv::Labeling    (const IplImage *src, IplImage *dst)
{

	// Only 1-Channel
	if( src->nChannels != 1 )
		return;

	// image size load
	int height    = src->height;
	int width    = src->width;

	// input image, result image를 image size만큼 동적할당
	unsigned char*    inputImage    = new unsigned char    [height * width];
	int*            resultImage    = new int            [height * width];    

	// before labeling prcess, initializing 
	for( int y = 0; y < height; y++ ){
		for( int x = 0; x < width; x++ ){
			// image copy
			inputImage[width * y + x] = src->imageData[width * y + x];

			// initialize result image
			resultImage[width * y + x] = 0;
		}
	}

	//// 8-neighbor labeling
	// Labeling 과정에서 stack overflow 방지를 위한 stl <stack>사용 
	stack<Point> st;
	int labelNumber = 0;
	for( int y = 1; y < height - 1; y++ ){
		for( int x = 1; x < width - 1; x++ ){
			// source image가 255일 경우 + Labeling 수행되지 않은 픽셀에서만 labeling process 시작
			if( inputImage[width * y + x] != 255 || resultImage[width * y + x] != 0 ) continue;

			labelNumber++;

			// 새로운 label seed를 stack에 push
			st.push(Point(x, y));

			// 해당 label seed가 labeling될 때(stack이 빌 때) 까지 수행
			while( !st.empty() ){
				// stack top의 label point를 받고 pop
				int ky = st.top().y;
				int kx = st.top().x;
				st.pop();

				// label seed의 label number를 result image에 저장
				resultImage[width * ky + kx] = labelNumber;

				// search 8-neighbor
				for( int ny = ky - 1; ny <= ky + 1; ny++ ){
					// y축 범위를 벗어나는 점 제외
					if( ny < 0 || ny >= height ) continue;
					for( int nx = kx - 1; nx <= kx + 1; nx++ ){
						// x축 범위를 벗어나는 점 제외
						if( nx < 0 || nx >= width ) continue;

						// source image가 값이 있고 labeling이 안된 좌표를 stack에 push
						if( inputImage[width * ny + nx] != 255 || resultImage[width * ny + nx] != 0 ) continue;
						st.push(Point(nx, ny));

						// 탐색한 픽셀이니 labeling
						resultImage[width * ny + nx] = labelNumber;
					}
				}
			}        
		}
	}

	// dst image에 복사
	for( int y = 0; y < height; y ++ ){
		for( int x = 0; x < width; x++ ){
			dst->imageData[width * y + x] = resultImage[width * y + x];
		}
	}

	// 메모리 해제
	delete[] inputImage;
	delete[] resultImage;
}



