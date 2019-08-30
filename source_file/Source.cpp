///////////////////////////////////
//	source.cpp : Defines the entry point for the console application.
//	Problem statement : find the distance between two points.  Compute the distance between O2 & C.
//	By : Dhruv Vyas
//	Date : 30/08/2019
//	Status: Working
//	Result : Distance between two points C to O2 is  4.2 cm
//////////////////////////////////

#include "stdafx.h"

#include <vector>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>>


using namespace std;
using namespace cv;

//Display
#define IMAGE_DISP (1)

class detection
{
	double asp_ratio;
	int width_resize, height_resize;
	double param1, param2, minDist;
	int x_ratio, y_ratio;
	vector<Vec3f> circles;
	vector<Point> points;
	Mat inp_img, resize_img, blur_img, disp_img;
	int dist_result_original, dist_result_resize;
	int coin_mm = 25;


public:
	detection(Mat img)
	{
		inp_img = img.clone();

		param1 = 110, param2 = 70, minDist = 24;
		asp_ratio = (double)img.rows / img.cols;
		width_resize = 512;
		height_resize = int(asp_ratio*width_resize);

		x_ratio = img.rows / height_resize;
		y_ratio = img.cols / width_resize;


	}

	void resize_input_img()
	{
		//cout << "width_resize : " << width_resize << "   height_resize: " << height_resize << endl;
		///Resize image - 
		resize(inp_img, resize_img, Size(width_resize, height_resize), 0, 0, 1);
		disp_img = resize_img.clone();
#if IMAGE_DISP
		//imshow("resize_img", resize_img);
		//waitKey(1);
#endif
	}

	void blur_input_img()
	{
		//Apply gaussianBlur to remove noise in image
		GaussianBlur(resize_img, blur_img, Size(5, 5), 2, 2);
#if IMAGE_DISP
		//imshow("gaussianBlur_img", blur_img);
		//waitKey(1);
#endif
	}



	void circle_detection()
	{
		//Apply houghcircle to find the all circles in image
		HoughCircles(blur_img, circles, CV_HOUGH_GRADIENT, 1, minDist, param1, param2, 0, 0);	

		// iteration on all circles
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// draw the circle center
			circle(disp_img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// draw the circle outline
			//circle(disp_img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
			putText(disp_img, to_string(i), center, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255), 2);
			//store center points
			points.push_back(center);
		}

		line(disp_img, points.at(0), points.at(3), Scalar(255, 0, 0), 1, 8, 0);
		line(disp_img, points.at(2), points.at(4), Scalar(255, 0, 0), 1, 8, 0);



	}

	///find distance between O2 to C
	void find_distance()
	{
		Mat temp_image_di = disp_img.clone();
		int radius = cvRound(circles[0][2]);	//82
		float pixel_per_metrix =  1.25/ radius;	//Map pixel with actual lenght

		//int pixel_per_metrix = abs(points.at(0).x - points.at(0).x+radius);
		//cout << "radius: " << radius << "  pixel_per_metrix: " << pixel_per_metrix << endl;

		//find slop and const of line 1
		float m_slop = float(float(points.at(0).y - points.at(3).y) / float(points.at(0).x - points.at(3).x));
		float c_const = points.at(0).y - (points.at(0).x * m_slop);
		//find slop and const of line 2
		float m_slop1 = float(float(points.at(2).y - points.at(4).y) / float(points.at(2).x - points.at(4).x));
		float c_const1 = points.at(2).y - (points.at(2).x * m_slop1);

		//cout << "m_slop : " << m_slop << "  c_const: " << c_const << endl;
		//cout << "m_slop1 : " << m_slop1 << "  c_const1: " << c_const1 << endl;
		
		//Find intersection points of line 1 and line 2
		int int_sec_x = ((c_const1 - c_const) / (m_slop - m_slop1));
		int int_sec_y = (m_slop*int_sec_x + c_const);

		int mid_point_x = abs(int((points.at(1).x - int_sec_x) / 2));
		int mid_point_y = abs(int((points.at(1).y - int_sec_y) / 2));

		dist_result_resize = points.at(1).x - int_sec_x;
		//Maping pixels to original Image
		dist_result_original = (points.at(1).x * x_ratio) - (int_sec_x * x_ratio);
		float final_result = dist_result_resize* pixel_per_metrix *2.54;

		// display 
		line(disp_img, Point(int_sec_x, int_sec_y), points.at(1), Scalar(255, 0, 0), 1, 8, 0);
		putText(disp_img, to_string(final_result)+"cm", Point(int_sec_x + mid_point_x, int_sec_y - mid_point_y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255), 2);

		cout << "Total pixels are : " << dist_result_original << endl;
		cout << "****** Apporx distance between point C to O2 is : " << (dist_result_resize* pixel_per_metrix *2.54) << "cm.  ***** " << endl;
		
#if IMAGE_DISP
		imshow("disp_img", disp_img);
		//imwrite("result_image.jpg", disp_img);
		waitKey(0);
#endif



	}

};


int main()
{

	// code start

	/// Input image read -
	Mat inp_img = imread("input_image.jpg", IMREAD_GRAYSCALE);

	/// Initialize the variables
	detection detect_obj1(inp_img);
	///Apply resize - 	
	detect_obj1.resize_input_img();
	///Apply blur - gaussuian blur to remove the noise from image
	detect_obj1.blur_input_img();
	///Apply hough circle to fine the circle shape form image
	detect_obj1.circle_detection();
	/// Find a distance bwtween two points and map to original image shape
	detect_obj1.find_distance();

	cout << "task completed .. Press any key for exit.. "  << endl;

	getchar();
	//end of code

	return 0;
}