#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#include<list>
using namespace std;

struct Pixel
{
public:
	Pixel (int height, int width)
	{
		h=height;
		w=width;
	}
	int w,h;
};

struct Cluster
{
public:
	Cluster () 
	{
		cnt=0;
	}
	Cluster (Cluster &other)
	{
		cnt=0;
		setV (other.px, other.py, other.c0, other.c1, other.c2);
	}
	Cluster (long int px, long int py, long int c0, long int c1, long int c2)
	{
		cnt=0;
		setV (px, py, c0, c1, c2);
	}
	void setV (long int px, long int py, long int c0, long int c1, long int c2)
	{
		this->px=px;
		this->py=py;
		this->c0=c0;
		this->c1=c1;
		this->c2=c2;
	}
	void addV (long int px, long int py, long int c0, long int c1, long int c2)
	{
		this->px+=px;
		this->py+=py;
		this->c0+=c0;
		this->c1+=c1;
		this->c2+=c2;
		cnt++;
	}
	void print ()
	{
		std::printf("cluster position: (%d,%d)  \t  color (%d,%d,%d)  \t count: %d \n",px,py,c0,c1,c2, cnt);
	}
	bool operator!= (Cluster &other)
	{
		if (px==other.px && 
			py==other.py && 
			c0==other.c0 && 
			c1==other.c1 && 
			c2==other.c2 )
			return false;
		else
			return true;
	}

	long int px, py, c0, c1, c2, cnt;
};

void createDisplay (uchar* disp, uchar* data_map, int height, int width, int step_disp, int step_map, Cluster *center)
{
	int x,y;
	for(y=0;y<height;y++) for(x=0;x<width;x++) 
	{
		disp[y*step_disp+3*x]  =0;
		disp[y*step_disp+3*x+1]=0;
		disp[y*step_disp+3*x+2]=0;

		const int TH_RED=-1; // 200
		const int TH_BLUE=260; // 160
		int c=data_map[y*step_map+x];
		if (center[c].c2>TH_RED &&
			center[c].c0<TH_BLUE)
		{
			// 0, 64, 128, 192, 255
			const int f=3;
			const int val=256 / (f-1);

			int ch0=val * (c % f); // 5
			c /= f;
			int ch1=val * (c % f);
			c /= f;
			int ch2=val * (c % f);
			if (ch0<0) ch0=0; if (ch0>255) ch0=255; 
			if (ch1<0) ch1=0; if (ch1>255) ch1=255; 
			if (ch2<0) ch2=0; if (ch2>255) ch2=255; 

			disp[y*step_disp+3*x]  =ch0;
			disp[y*step_disp+3*x+1]=ch1;
			disp[y*step_disp+3*x+2]=ch2;
		}
	}
	return;
}

void k_means (uchar *data, int height, int width, int step, int channels)
{
	// initialize clusters
	const int MAX_C=12;
	Cluster center[MAX_C];
	Cluster old_center[MAX_C];
	int x,y, c;
	for (c=0; c<MAX_C; c++)
	{
		x = (int)((double)rand() / (RAND_MAX + 1) * width);
		y = (int)((double)rand() / (RAND_MAX + 1) * height);
		center[c].setV(x,y, data[y*step+x*channels], data[y*step+x*channels+1], data[y*step+x*channels+2]);
	}
	for (c=0; c<MAX_C; c++)
		old_center[c]=center[c];
	for (c=0; c<MAX_C; c++)
		center[c].print();
	std::printf("-----\n");

	IplImage* region_img=cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1); 
	uchar* data_map = (uchar *)region_img->imageData;
	int step_map = region_img->widthStep;
	for(y=0;y<height;y++) for(x=0;x<width;x++) 
		data_map[y*step_map+x]=0;

	IplImage* disp_img=cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3); 
	uchar* disp = (uchar *)disp_img->imageData;
	int step_disp = disp_img->widthStep;

	cvNamedWindow("disp", CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("disp", 400, 100);

	// show results
	createDisplay (disp, data_map, height, width, step_disp, step_map, center);
	cvShowImage("disp", disp_img );
	cvWaitKey(1);

	bool changed=true;
	while (changed)
	{
		// allocate pixel to next cluster
		for(y=0;y<height;y++) for(x=0;x<width;x++) 
		{
			long int optDist=1000000;
			for (c=0; c<MAX_C; c++)
			{
				const int TH_DIST=50;
				long int d= 
					MAX (abs(center[c].px-x)-TH_DIST, 0) + 
					MAX (abs(center[c].py-y)-TH_DIST, 0) + 
					abs(center[c].c0-data[y*step+x*channels]) + 
					abs(center[c].c1-data[y*step+x*channels+1]) + 
					abs(center[c].c2-data[y*step+x*channels+2]);
				/*
				long int d= 
				abs(center[c].c0-data[y*step+x*channels]) + 
				abs(center[c].c1-data[y*step+x*channels+1]) + 
				abs(center[c].c2-data[y*step+x*channels+2]);
				*/
				if (optDist>d)
				{
					optDist=d;
					data_map[y*step_map+x]=c;
				}
			}
		}
		// show results
		createDisplay (disp, data_map, height, width, step_disp, step_map, center);
		cvShowImage("disp", disp_img );
		cvWaitKey(1);

		// update cluster centers 
		for (c=0; c<MAX_C; c++)
		{
			center[c].setV (0,0,0,0,0);
			center[c].cnt=0;
		}
		for(y=0;y<height;y++) for(x=0;x<width;x++) 
		{
			c=data_map[y*step_map+x];
			center[c].addV (x,y, data[y*step+x*channels], data[y*step+x*channels+1], data[y*step+x*channels+2]);
		}
		for (c=0; c<MAX_C; c++)
		{
			center[c].px /=center[c].cnt;
			center[c].py /=center[c].cnt;
			center[c].c0 /=center[c].cnt;
			center[c].c1 /=center[c].cnt;
			center[c].c2 /=center[c].cnt;
		}
		changed=false;
		for (c=0; c<MAX_C; c++)
			if (old_center[c] != center[c]) changed=true;

		for (c=0; c<MAX_C; c++)
			center[c].print();
		std::printf("-----\n");
		for (c=0; c<MAX_C; c++)
			old_center[c]=center[c];
	}

	// mark clusters
	const int TH_RED  =200;
	const int TH_BLUE =120;
	const int TH_GREEN=140;
	for (c=0; c<MAX_C; c++)
	{
		if (center[c].c2>TH_RED &&
			center[c].c1<TH_GREEN &&
			center[c].c0<TH_BLUE)
		{
			center[c].cnt=1000;
		} 
		else 
		{
			center[c].cnt=0;
		}
	}

	// mark results
	for(y=0;y<height;y++) for(x=0;x<width;x++) 
	{
		c=data_map[y*step_map+x];
		if (center[c].cnt>0)
		{
			data[y*step+x*channels+2]=0;   // red
			data[y*step+x*channels+1]=255; // green
			data[y*step+x*channels+0]=0;   // blue
		}
		else
		{
			/*
			data[y*step+x*channels+2]=0;
			data[y*step+x*channels+1]=0;
			data[y*step+x*channels  ]=0;
			*/
		}
	}

	return;
}

void threshold (uchar *data, int height, int width, int step, int channels)
{
	const int TH1=180;
	const int TH2=160;
	const int TH3=160;

	int y,x;
	int cnt=0;
	for(y=0;y<height;y++) for(x=0;x<width;x++) 
		if (data[y*step+x*channels+2] >TH1 && 
			data[y*step+x*channels+1] <TH2 && 
			data[y*step+x*channels  ] <TH3)
		{
			cnt++;
			data[y*step+x*channels+2]=0;   // red
			data[y*step+x*channels+1]=255; // green
			data[y*step+x*channels+0]=0;   // blue
		}
		std::printf("number of seed pixels: %d \n",cnt);
		return;
}


void region (uchar *data, int height, int width, int step, int channels)
{
	IplImage* region_img=cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1); 
	uchar* data_map = (uchar *)region_img->imageData;
	int step_map = region_img->widthStep;

	// initialize map and set seed points
	int y,x;
	int cnt=0;
	list<Pixel> pixel;
	for(y=0;y<height;y++) for(x=0;x<width;x++) 
		//R G B
		if (data[y*step+x*channels+2] >85 && data[y*step+x*channels+2] < 120 && data[y*step+x*channels+1] >100 && data[y*step+x*channels+1] <135 && data[y*step+x*channels] > 90 && data[y*step+x*channels  ] <125)    
		{
			cnt++;
			data_map[y*step_map+x]=255;
			pixel.push_back(Pixel(y,x));
		}
		else
			data_map[y*step_map+x]=0;
	//std::printf("number of seed pixels: %d \n",cnt);

	// region growing
	cnt=0;
	while (pixel.size()>0)
	{
		Pixel p=pixel.front();
		pixel.pop_front();
		cnt++;

		const int TH1=180;
		const int TH2=160;
		const int TH3=160;

		x=p.w-1; y=p.h; // left
		if (data[y*step+x*channels+2] >TH1 && 
			data[y*step+x*channels+1] <TH2 && 
			data[y*step+x*channels  ] <TH3 && 
			data_map[y*step_map+x]==0) 
		{
			data_map[y*step_map+x]=255;
			pixel.push_back(Pixel(y,x));
		}

		x=p.w+1; y=p.h; // right
		if (data[y*step+x*channels+2] >TH1 && 
			data[y*step+x*channels+1] <TH2 && 
			data[y*step+x*channels  ] <TH3 && 
			data_map[y*step_map+x]==0) 
		{
			data_map[y*step_map+x]=255;
			pixel.push_back(Pixel(y,x));
		}

		x=p.w; y=p.h-1; // top
		if(y > 0){
			if (data[y*step+x*channels+2] >TH1 && 
				data[y*step+x*channels+1] <TH2 && 
				data[y*step+x*channels  ] <TH3 && 
				data_map[y*step_map+x]==0) 
			{
				data_map[y*step_map+x]=255;
				pixel.push_back(Pixel(y,x));
			}

			x=p.w; y=p.h+1; // bottom
			if (data[y*step+x*channels+2] >TH1 && 
				data[y*step+x*channels+1] <TH2 && 
				data[y*step+x*channels  ] <TH3 && 
				data_map[y*step_map+x]==0) 
			{
				data_map[y*step_map+x]=255;
				pixel.push_back(Pixel(y,x));
			}
		}

	}
	//std::printf("number of seed pixels: %d \n",cnt);

	cnt=0;
	for(y=0;y<height;y++) for(x=0;x<width;x++) 
		if (data_map[y*step_map+x]==255)
		{
			cnt++;
			data[y*step+x*channels+2]=0;   // red
			data[y*step+x*channels+1]=255; // green
			data[y*step+x*channels+0]=0;   // blue
		}
		else
		{
			/*
			data[y*step+x*channels+2]=0;
			data[y*step+x*channels+1]=0;
			data[y*step+x*channels  ]=0;
			*/
		}
		//std::printf("number of seed pixels: %d \n",cnt);
		return;
}