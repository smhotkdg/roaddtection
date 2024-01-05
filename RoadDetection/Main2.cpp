///////////////////////////////////////////////////////////////////////
// OpenCV tracking example.

//#include "stdafx.h"
#include <iostream>
#include <windows.h>

#include "tracker_opencv2.h"
#include <tchar.h>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <windows.h>

#include "IPM.h"

using namespace std;


struct CallbackParam
{
	Mat frame;
	Point pt1,pt2;
	Rect roi;
	bool drag;
	bool updated;
};

tracker_opencv tracker;
VideoCapture *vc = NULL;

void onMouse( int event, int x, int y, int flags, void* param )
{
	CallbackParam *p = (CallbackParam *)param;
		
	if( event == CV_EVENT_LBUTTONDOWN )
	{
		p->pt1.x = x;
		p->pt1.y = y;
		p->pt2 = p->pt1;
		p->drag = true;
	}
	if( event == CV_EVENT_LBUTTONUP )
	{
		int w = x - p->pt1.x;
		int h = y - p->pt1.y;

		p->roi.x = p->pt1.x;
		p->roi.y = p->pt1.y;
		p->roi.width = w;
		p->roi.height = h;
		p->drag = false;

		if(w>=10 && h>=10)
		{
			p->updated = true;
		}
	}
	if( p->drag && event == CV_EVENT_MOUSEMOVE )
	{
		if(p->pt2.x != x || p->pt2.y != y)
		{
			Mat img = p->frame.clone();
			p->pt2.x = x;
			p->pt2.y = y;
			rectangle(img, p->pt1, p->pt2, Scalar(0,255,0), 1);
		
			imshow("image", img);	
		}
	}
}



void image_proc(Mat frame)
{
	tracker.configure();

	imshow("image", frame);


	CallbackParam param;
	param.frame = frame;
	param.drag = false;
	param.updated = false;
	setMouseCallback("image", onMouse, &param);


	bool tracking = false;
	while(1)
	{
		// image acquisition & target init
		if(param.drag)
		{
		    if( waitKey(10) == 27 ) break;		// ESC key
			continue;
		}
			
			Rect rc = param.roi;
 			tracker.init(frame, rc);
			param.updated = false;
			tracking = true;		
		

		if(frame.empty()) break;

		// image processing
		if(tracking)
		{
			Rect rc;	
			//bool ok = tracker.run(frame, rc,0);
		}

		// image display
		//imshow("image", frame);



		// user input
		char ch = waitKey(10);		
		ch = 32;
		if( ch == 27 ) break;	// ESC Key (exit)
		else if(ch == 32 )	// SPACE Key (pause)
		{
			while((ch = waitKey(10)) != 32 && ch != 27);
			if(ch == 27) break;
		}
	}
}

void proc_video(VideoCapture *vc)
{
	
	tracker.configure();

	Mat frame;
	*vc >> frame;
	imshow("image", frame);


	CallbackParam param;
	param.frame = frame;
	param.drag = false;
	param.updated = false;
    setMouseCallback("image", onMouse, &param);

	Mat outputImg;
	int width = 0, height = 0, fps = 0, fourcc = 0;	
	width = static_cast<int>(vc->get(CV_CAP_PROP_FRAME_WIDTH));
	height = static_cast<int>(vc->get(CV_CAP_PROP_FRAME_HEIGHT));
	fps = static_cast<int>(vc->get(CV_CAP_PROP_FPS));
	fourcc = static_cast<int>(vc->get(CV_CAP_PROP_FOURCC));

	vector<Point2f> origPoints;
	origPoints.push_back( Point2f(0, height-70) );
	origPoints.push_back( Point2f(width, height-70) );
	origPoints.push_back( Point2f(width, height/3) );
	origPoints.push_back( Point2f(0, height/3) );

	//origPoints.push_back( Point2f(width/2+30, 140) );
	//origPoints.push_back( Point2f(width/2-50, 140) );

	// The 4-points correspondences in the destination image
	vector<Point2f> dstPoints;
	dstPoints.push_back( Point2f(120, height) );
	dstPoints.push_back( Point2f(240, height) );
	dstPoints.push_back( Point2f(width, 0) );
	dstPoints.push_back( Point2f(0, 0) );	

	IPM ipm( Size(width, height), Size(width, height), origPoints, dstPoints );


	bool tracking = false;
	while(1)
	{
		// image acquisition & target init
		if(param.drag)
		{
		    if( waitKey(10) == 27 ) break;		// ESC key
			continue;
		}
			
			Rect rc = param.roi;
 			tracker.init(frame, rc);
			param.updated = false;
			tracking = true;			
		
		*vc >> frame;
		if(frame.empty()) break;

		// image processing
		if(tracking)
		{
			Rect rc;
			//ipm.applyHomography(frame, outputImg );		
			//ipm.drawPoints(origPoints, frame );
			bool ok = tracker.run(frame, rc,outputImg);	
		}

		// image display
		imshow("image", frame);
		//imshow("Output", outputImg);



		// user input
		char ch = waitKey(10);		
		if( ch == 27 ) break;	// ESC Key (exit)
		else if(ch == 32 )	// SPACE Key (pause)
		{
			while((ch = waitKey(10)) != 32 && ch != 27);
			if(ch == 27) break;
		}
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	char data_src;
	cout << "1. camera input (640 x 480)\n"
		 << "2. camera input (320 x 240)\n"
		 << "3. video file input\n"
		 << endl
		 << "select video source[1-3]: ";
	cin >> data_src;

	
	if(data_src=='1')
	{
		//camera (vga)
		vc = new VideoCapture(0);
		if (!vc->isOpened())
		{
			cout << "can't open camera" << endl;
			return 0;
		}
		vc->set(CV_CAP_PROP_FRAME_WIDTH, 640);
		vc->set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	}
	else if(data_src=='2')
	{
		//camera (qvga)
		vc = new VideoCapture(0);
		if (!vc->isOpened())
		{
			cout << "can't open camera" << endl;
			return 0;
		}
		vc->set(CV_CAP_PROP_FRAME_WIDTH, 320);
		vc->set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	}
	else if(data_src=='3')
	{
		//video (avi)
		OPENFILENAME ofn;
		TCHAR szFile[MAX_PATH] = {0,};
		ZeroMemory(&ofn, sizeof(OPENFILENAME));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner = NULL;
		ofn.lpstrFile = szFile;
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = _T("Avi Files(*.avi)\0*.avi\0All Files (*.*)\0*.*\0");
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = NULL;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
		if(::GetOpenFileName(&ofn)==false) return 0;

		vc = new VideoCapture(ofn.lpstrFile);
		if (!vc->isOpened())
		{
			cout << "can't open video file" << endl;
			return 0;
		}
	}
	else if(data_src=='4')
	{	//select image source
	
	Mat i = imread("c:\\1.jpg",1);
	if (i.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", i ); 
	
	image_proc(i);
	waitKey(0);
	}
	if(vc) proc_video(vc);
	if(vc) delete vc;

	//destroyAllWindows();

	return 0;
}
