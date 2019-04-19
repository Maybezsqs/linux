#ifndef WINDMILL_H
#define WINDMILL_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <General/General.h>

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace rm;
struct windParam
{
    //pre-treatment
    int armor_threshold;
    int leaf_threshold;

    //Filter armor leaf
    float armor_ratio;
    float leaf_ratio;
    float armor_MaxArea;
    float leaf_MaxArea;

    //other params
    float sight_offset_normalized_base;
    float area_normalized_base;
    int enemy_color;
    int max_track_num = 3000;

    /*
    *	@Brief: 为各项参数赋默认值
    */
    windParam()
    {
        //pre-treatment
//		brightness_threshold = 210;
//		color_threshold = 40;
//		light_color_detect_extend_ratio = 1.1;

        // Filter lights
//		light_min_area = 10;
//		light_max_angle = 45.0;
//		light_min_size = 5.0;
//		light_contour_min_solidity = 0.5;
//        	light_max_ratio = 1.0;

        // Filter pairs
//        	light_max_angle_diff_ = 7.0; //20
//        	light_max_height_diff_ratio_ = 0.2; //0.5
//		light_max_y_diff_ratio_ = 2.0; //100
//		light_min_x_diff_ratio_ = 0.5; //100

        // Filter armor
        armor_ratio=0;
        leaf_ratio=0;
        armor_MaxArea=0;
        leaf_MaxArea=0;

        //other params
        sight_offset_normalized_base = 200;
        area_normalized_base = 1000;
        enemy_color = rm::BLUE;
    }
};


class windmill
{
public:
    windmill();
    double getDistance(const Point &A,const Point &B);//Distance between two point

    vector<float> stander(Mat &im);
    Mat getSvmInput(Mat &input);

    void getArmorLeafImg(Mat &srcImage);// create the imgs to use
    vector<Point2f> findTargetLeaf();//return the leaf four points
    vector<Point2f> findArmor();//return the center point
    void loadSVM(string mod);
    void loadImg(const cv::Mat & srcImg);
    /*
    *	@Brief: set the enemy's color
    *	@Others: API for client
    */
    void setEnemyColor(int enemy_color)
    {
        _enemy_color = enemy_color;
        _self_color = enemy_color == BLUE ? RED : BLUE;
    }
private:
    Point centerPoint;
    Mat srcImage;
    Mat img2leaf;
    Mat img2armor;
    bool findTarget=false;
    Ptr<SVM> svm;
    int _enemy_color;
    int _self_color;
};

#endif // WINDMILL_H
