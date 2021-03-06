
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>


using namespace std;
using namespace cv;


#define T_ANGLE_THRE 15
#define T_SIZE_THRE 0.15
#define Lost_times 20
#define track_long 100
int color_change=0;

class RectTrack//
{
public:
    RotatedRect Rect;
    vector<Point> Track;
    int count =Lost_times;
};




void drawBox(RotatedRect box, Mat img);
Mat select(Mat im,int hh,int hl,int sh,int sl,int vh,int vl);
Mat select_Red(Mat im);
Mat select_Blue(Mat im);
vector<RotatedRect> ArmorDetect(vector<RotatedRect> vEllipse);
vector<RotatedRect> getContours(Mat gray);
void drawLine( vector<RectTrack> TRACK, Mat img);
Scalar_<double> colormap(double n);
int find(RotatedRect rect,vector<RectTrack> track);
void gettrack(vector<RotatedRect> vRlt,vector<RectTrack>& TRACK);


void on_Trackbar_color_change(int, void*)
{
}


#define VIDEO


int main(int argc, char** argv)
 {

    namedWindow("gray");
    namedWindow("image");
    createTrackbar("Blue or Red","image",&color_change,1,on_Trackbar_color_change);


    Mat im,gray;//define image and his gray picture
    TickMeter tim;//to take time
    vector<RectTrack> TRACK;// vectory of a class   to memory track and rotate rect
    vector<RotatedRect> vEllipse; //定以旋转矩形的向量，用于存储发现的目标区域
    vector<RotatedRect> vRlt;// the likely armor plate    Typy rotate rect
    stringstream fps;// to show fps in image
    int n=0;// to count times



#ifdef VIDEO
    VideoCapture video(1);//open
//    video.open("/home/zsq/sq_file/opencv/Line_sign/data/23.mp4");

    if(video.isOpened()==0)
    {
        cout<<"failed open the camera"<<endl;
        return -1;
    }
    video.set(CAP_PROP_AUTO_EXPOSURE,0.25);
    video.set(CV_CAP_PROP_EXPOSURE,(float)16/10000);
    video.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));//设置为MJPG格式
#else

    im=imread("/home/zsq/sq_file/opencv/Line_sign/data/2.jpg");
    if(im.empty()==1)
    {
        cout<<"failed open the image";
        return 1;
    }
    resize(im,im,Size(im.cols/4,im.rows/4));
#endif


    fps<<"fps"<<endl;

    while(1)
        {

        if(n==0){tim.reset();tim.start();}
        n++;


#ifdef VIDEO
        video>>im;
        blur(im,im,Size(5,5),Point(-1,-1));
#endif

        //gray=select(im,184,168,255,150,255,117);//red  0~~158  f16     blue 116 70  f16
        if(color_change==0) gray=select_Blue(im);
        else gray=select_Red(im);


        Mat e5(10,10,CV_8U,Scalar(1));
        morphologyEx(gray,gray,MORPH_CLOSE,e5);
//      morphologyEx(gray,gray,MORPH_OPEN,e5);
//      Canny(gray,gray,100,200,5);


        vEllipse=getContours(gray);//返回选卷矩形

        vRlt=ArmorDetect(vEllipse);//选择可能装甲版矩形

        for (unsigned int i = 0; i < vRlt.size(); i++) drawBox(vRlt[i], im);//在当前图像中标出装甲的位置


        gettrack(vRlt,TRACK);

#define ss
#ifdef ss

        vector<RectTrack>::iterator it=TRACK.begin();
        while(it!=TRACK.end())
        {
            (*it).count-=1;
            cout<<(*it).count<<endl;
            if((*it).count<0)
            {
                //delete *it;
                it=TRACK.erase(it);
                cout<<"delete a tract"<<endl;
            }

            else it++;

        }
#endif

        drawLine(TRACK,im);


        if(n==50)//每为
        {
            tim.stop();
            fps.str("");
            fps<<"fps "<<n*1000000/tim.getTimeMicro()<<endl;
            n=0;
        }

        cout<<"asdaddad"<<fps.str();
        putText(im,fps.str(),Point(20,30),FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,5);




        imshow("gray",gray);
        imshow("image",im);

        if(waitKey(1)==27) break;




        }


        return  0;
}








int find(RotatedRect rect,vector<RectTrack> track)
{

    int result = -1;
    for(int i=0;i<track.size();i++)
    {
        if(pow((track[i].Rect.center.x-rect.center.x),2)+pow((track[i].Rect.center.y-rect.center.y),2)<1000
                &&(abs(rect.size.width-track[i].Rect.size.width))/track[i].Rect.size.width<0.1
                &&(abs(rect.size.height-track[i].Rect.size.height))/track[i].Rect.size.width<0.1)
        {
            result=i;
            break;
        }
    }
    return result;
}


Scalar_<double> colormap(double n)
{
    int b=255*rand();
    return Scalar(b,100+155*n,255-155*n);
}


void gettrack(vector<RotatedRect> vRlt,vector<RectTrack>& TRACK)
{

    int F;
    RectTrack TRACKss;
    if(0==vRlt.empty())
    {

        for (unsigned int i = 0; i < vRlt.size(); i++)
        {
            F=find(vRlt[i],TRACK);
            //cout<<F<<endl;


            if(F<0)//find
            {
                TRACKss.Rect=vRlt[i];
                TRACKss.Track.push_back(vRlt[i].center);
                //TRACKss.count=20;
                TRACKss.Track.clear();
                TRACK.push_back(TRACKss);

            }
            else
            {
                //cout<<"track"<<endl;

                TRACK[F].Rect=vRlt[i];

                TRACK[F].count=Lost_times;
                if(TRACK[F].Track.size()<track_long) TRACK[F].Track.push_back(vRlt[i].center);
                else
                {
                    TRACK[F].Track.erase(TRACK[F].Track.begin());
                    TRACK[F].Track.push_back(vRlt[i].center);

                    //cout<<"track"<<TRACK[F].Track.size()<<endl;
                }
            }


        }
    }




}




void drawBox(RotatedRect box, Mat img)
{
    Point2f pt[4];
    int i;
    stringstream str;


    for (i = 0; i<4; i++)
    {
        pt[i].x = 0;
        pt[i].y = 0;
    }
    box.points(pt); //计算二维盒子顶点




    str<<(int)(3185/box.size.width)<<"cm";


    //std::cout<<"with"<<(float)box.size.height/box.size.width<<"\n\n"<<endl;

    line(img, pt[0], pt[1], CV_RGB(0, 255, 0), 3, 8, 0);//short
    line(img, pt[1], pt[2], CV_RGB(200, 0, 200), 3, 8, 0);
    line(img, pt[2], pt[3], CV_RGB(0, 255, 0), 3, 8, 0);//short
    line(img, pt[3], pt[0], CV_RGB(200, 0, 200), 3, 8, 0);

    circle(img, box.center, box.size.width/4, CV_RGB(0, 200, 200), 3, 8, 0);
    putText(img,str.str(),box.center,FONT_HERSHEY_PLAIN,2,Scalar(100,0,200),2,5);
}


void drawLine( vector<RectTrack> TRACK, Mat img)
{

    for (unsigned int i = 0; i < TRACK.size(); i++)
    {
        if(TRACK[i].count > Lost_times-2)
        {
            for (unsigned int j = 1; j < TRACK[i].Track.size(); j++)
            {
            line(img, TRACK[i].Track[j-1], TRACK[i].Track[j], colormap((float)i/TRACK.size()), 3, 8, 0);//short
            }
        }
        else
        {

            cout<<"track "<<i<<"                                                                            missing"<<TRACK[i].count<<endl;
        }

    }
}

Mat select(Mat im,int hh,int hl,int sh,int sl,int vh,int vl)
{


    cvtColor(im,im,CV_BGR2HSV);
    Mat gray(im.rows,im.cols,CV_8UC1);

    for(int i=0;i<im.cols;i++)
    {
        for(int j=0;j<im.rows;j++)
        {
            if(im.at<Vec3b>(j,i)[0]<=hh
                    &&im.at<Vec3b>(j,i)[0]>=hl
                    &&im.at<Vec3b>(j,i)[1]<=sh
                    &&im.at<Vec3b>(j,i)[1]>=sl
                    &&im.at<Vec3b>(j,i)[2]<=vh
                    &&im.at<Vec3b>(j,i)[2]>=vl)

                    gray.at<uchar>(j,i)=255;
            else
                    gray.at<uchar>(j,i)=0;
        }
    }
    cvtColor(im,im,CV_HSV2BGR);
    return gray;
}
//gray=select(im,184,168,255,150,255,117);

Mat select_Blue(Mat im)
{
    return select(im,116,70,255,150,255,117);
}
Mat select_Red(Mat im)
{
    cvtColor(im,im,CV_BGR2HSV);
    Mat gray(im.rows,im.cols,CV_8UC1);

    for(int i=0;i<im.cols;i++)
    {
        for(int j=0;j<im.rows;j++)
        {
            if(
                    ((im.at<Vec3b>(j,i)[0]<=184 && im.at<Vec3b>(j,i)[0]>=168) || (im.at<Vec3b>(j,i)[0]>=0 && im.at<Vec3b>(j,i)[0]<=10))
                    &&im.at<Vec3b>(j,i)[1]<=255
                    &&im.at<Vec3b>(j,i)[1]>=150
                    &&im.at<Vec3b>(j,i)[2]<=255
                    &&im.at<Vec3b>(j,i)[2]>=117
               )

                    gray.at<uchar>(j,i)=255;
            else
                    gray.at<uchar>(j,i)=0;
        }
    }
    cvtColor(im,im,CV_HSV2BGR);
    return gray;
}


vector<RotatedRect> ArmorDetect(vector<RotatedRect> vEllipse)//height gao yi dian
{
    vector<RotatedRect> vRlt;
    RotatedRect Armor; //定义装甲区域的旋转矩形
    int nL, nW;
    double dAngle;
    vRlt.clear();
    if (vEllipse.size() < 2) //如果检测到的旋转矩形个数小于2，则直接返回
    {
        std::cout<<"less                two"<<rand()<<endl;
        return vRlt;
    }

    for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++) //求任意两个旋转矩形的夹角
    {
        for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
        {


            dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
            while (dAngle > 90)
                dAngle=180-dAngle;

            if (
                    (dAngle < T_ANGLE_THRE)

                    &&(abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) / (vEllipse[nI].size.height + vEllipse[nJ].size.height)) < T_SIZE_THRE


               )
//                    &&
//                    (abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) / (vEllipse[nI].size.width + vEllipse[nJ].size.width)) < T_SIZE_THRE) //判断这两个旋转矩形是否是一个装甲的两个LED等条
            {



                Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2; //装甲中心的x坐标
                Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2; //装甲中心的y坐标




                if (abs(vEllipse[nI].angle - vEllipse[nJ].angle)<90)  Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;   //装甲所在旋转矩形的旋转角度
                else {Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle-180) / 2;}

                if(Armor.angle>0) Armor.angle-=90;
                else {Armor.angle+=90;}

                    //Armor.angle += 90;
                nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2; //装甲的高度
                nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y)); //装甲的宽度等于两侧LED所在旋转矩形中心坐标的距离
                if (nL < nW)
                {
                    Armor.size.height = nW;
                    Armor.size.width = nL;

                    if((float)nW/nL<3.5)
                    vRlt.push_back(Armor); //将找出的装甲的旋转矩形保存到vector

                    //cout<<"annnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnllllllllllllllllllllllllllll"<<nL<<"         sd"<<(abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) / (vEllipse[nI].size.height + vEllipse[nJ].size.height))<<endl;


                }
            }
        }
    }
    return vRlt;
}



vector<RotatedRect> getContours(Mat gray)
{
    vector<vector<Point> > contour;
    RotatedRect s;   //定义旋转矩形
    vector<RotatedRect> result; //定以旋转矩形的向量，用于存储发现的目标区域


    findContours(gray, contour, RETR_CCOMP , CHAIN_APPROX_SIMPLE);


    for (int i=0; i<contour.size(); i++)
    {
        if(contour[i].size()> 5)
        {
            s = fitEllipse(contour[i]);
            result.push_back(s); //将发现的目标保存
        }
    }


    return result;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>


using namespace std;
using namespace cv;


#define T_ANGLE_THRE 15
#define T_SIZE_THRE 0.15
#define Lost_times 20
#define track_long 100


class RectTrack//
{
public:
    RotatedRect Rect;
    vector<Point> Track;
    int count =Lost_times;
};




void drawBox(RotatedRect box, Mat img);
Mat select(Mat im,int hh,int hl,int sh,int sl,int vh,int vl);
vector<RotatedRect> ArmorDetect(vector<RotatedRect> vEllipse);
vector<RotatedRect> getContours(Mat gray);
void drawLine( vector<RectTrack> TRACK, Mat img);
Scalar_<double> colormap(double n);
int find(RotatedRect rect,vector<RectTrack> track);
void gettrack(vector<RotatedRect> vRlt,vector<RectTrack>& TRACK);




#define VIDEO


int main(int argc, char** argv)
 {

    namedWindow("gray");
    namedWindow("image");


    Mat im,gray;//define image and his gray picture
    TickMeter tim;//to take time
    vector<RectTrack> TRACK;// vectory of a class   to memory track and rotate rect
    vector<RotatedRect> vEllipse; //定以旋转矩形的向量，用于存储发现的目标区域
    vector<RotatedRect> vRlt;// the likely armor plate    Typy rotate rect
    stringstream fps;// to show fps in image
    int n=0;// to count times



#ifdef VIDEO
    VideoCapture video(1);//open
//    video.open("/home/zsq/sq_file/opencv/Line_sign/data/23.mp4");

    if(video.isOpened()==0)
    {
        cout<<"failed open the camera"<<endl;
        return -1;
    }
    video.set(CAP_PROP_AUTO_EXPOSURE,0.25);
    video.set(CV_CAP_PROP_EXPOSURE,(float)30/10000);
    video.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));//设置为MJPG格式
#else

    im=imread("/home/zsq/sq_file/opencv/Line_sign/data/2.jpg");
    if(im.empty()==1)
    {
        cout<<"failed open the image";
        return 1;
    }
    resize(im,im,Size(im.cols/4,im.rows/4));
#endif


    fps<<"fps"<<endl;

    while(1)
        {

        if(n==0){tim.reset();tim.start();}
        n++;


#ifdef VIDEO
        video>>im;
#endif

        gray=select(im,116,70,255,150,255,117);//70 116


        Mat e5(10      ,10,CV_8U,Scalar(1));
        morphologyEx(gray,gray,MORPH_CLOSE,e5);
//      morphologyEx(gray,gray,MORPH_OPEN,e5);
//      Canny(gray,gray,100,200,5);


        vEllipse=getContours(gray);//返回选卷矩形

        vRlt=ArmorDetect(vEllipse);//选择可能装甲版矩形

        for (unsigned int i = 0; i < vRlt.size(); i++) drawBox(vRlt[i], im);//在当前图像中标出装甲的位置


        gettrack(vRlt,TRACK);

#define ss
#ifdef ss

        vector<RectTrack>::iterator it=TRACK.begin();
        while(it!=TRACK.end())
        {
            (*it).count-=1;
            cout<<(*it).count<<endl;
            if((*it).count<0)
            {
                //delete *it;
                it=TRACK.erase(it);
                cout<<"delete a tract"<<endl;
            }

            else it++;

        }
#endif

        drawLine(TRACK,im);


        if(n==50)//每为
        {
            tim.stop();
            fps.str("");
            fps<<"fps "<<n*1000000/tim.getTimeMicro()<<endl;
            n=0;
        }

        cout<<"asdaddad"<<fps.str();
        putText(im,fps.str(),Point(20,30),FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,5);




        imshow("gray",gray);
        imshow("image",im);

        if(waitKey(1)==27) break;




        }


        return  0;
}








int find(RotatedRect rect,vector<RectTrack> track)
{

    int result = -1;
    for(int i=0;i<track.size();i++)
    {
        if(pow((track[i].Rect.center.x-rect.center.x),2)+pow((track[i].Rect.center.y-rect.center.y),2)<1000
                &&(abs(rect.size.width-track[i].Rect.size.width))/track[i].Rect.size.width<0.1
                &&(abs(rect.size.height-track[i].Rect.size.height))/track[i].Rect.size.width<0.1)
        {
            result=i;
            break;
        }
    }
    return result;
}


Scalar_<double> colormap(double n)
{
    int b=255*rand();
    return Scalar(b,100+155*n,255-155*n);
}


void gettrack(vector<RotatedRect> vRlt,vector<RectTrack>& TRACK)
{

    int F;
    RectTrack TRACKss;
    if(0==vRlt.empty())
    {

        for (unsigned int i = 0; i < vRlt.size(); i++)
        {
            F=find(vRlt[i],TRACK);
            //cout<<F<<endl;


            if(F<0)//find
            {
                TRACKss.Rect=vRlt[i];
                TRACKss.Track.push_back(vRlt[i].center);
                //TRACKss.count=20;
                TRACKss.Track.clear();
                TRACK.push_back(TRACKss);

            }
            else
            {
                //cout<<"track"<<endl;

                TRACK[F].Rect=vRlt[i];

                TRACK[F].count=Lost_times;
                if(TRACK[F].Track.size()<track_long) TRACK[F].Track.push_back(vRlt[i].center);
                else
                {
                    TRACK[F].Track.erase(TRACK[F].Track.begin());
                    TRACK[F].Track.push_back(vRlt[i].center);

                    //cout<<"track"<<TRACK[F].Track.size()<<endl;
                }
            }


        }
    }




}




void drawBox(RotatedRect box, Mat img)
{
    Point2f pt[4];
    int i;
    for (i = 0; i<4; i++)
    {
        pt[i].x = 0;
        pt[i].y = 0;
    }
    box.points(pt); //计算二维盒子顶点


    stringstream str;

    //str<<box.angle<<"%"<<endl;
    //putText(img,str.str(),pt[0],FONT_HERSHEY_PLAIN,1,Scalar(0,255,0),2,5);

    //std::cout<<"with"<<(float)box.size.height/box.size.width<<"\n\n"<<endl;

    line(img, pt[0], pt[1], CV_RGB(0, 255, 0), 3, 8, 0);//short
    line(img, pt[1], pt[2], CV_RGB(200, 0, 200), 3, 8, 0);
    line(img, pt[2], pt[3], CV_RGB(0, 255, 0), 3, 8, 0);//short
    line(img, pt[3], pt[0], CV_RGB(200, 0, 200), 3, 8, 0);

    circle(img, box.center, box.size.width/4, CV_RGB(0, 200, 200), 3, 8, 0);
}


void drawLine( vector<RectTrack> TRACK, Mat img)
{

    for (unsigned int i = 0; i < TRACK.size(); i++)
    {
        if(TRACK[i].count > Lost_times-2)
        {
            for (unsigned int j = 1; j < TRACK[i].Track.size(); j++)
            {
            line(img, TRACK[i].Track[j-1], TRACK[i].Track[j], colormap((float)i/TRACK.size()), 3, 8, 0);//short
            }
        }
        else
        {

            cout<<"track "<<i<<"                                                                            missing"<<TRACK[i].count<<endl;
        }

    }
}


Mat select(Mat im,int hh,int hl,int sh,int sl,int vh,int vl)
{
    cvtColor(im,im,CV_BGR2HSV);
    Mat gray(im.rows,im.cols,CV_8UC1);

    for(int i=0;i<im.cols;i++)
    {
        for(int j=0;j<im.rows;j++)
        {
            if(im.at<Vec3b>(j,i)[0]<=hh
                    &&im.at<Vec3b>(j,i)[0]>=hl
                    &&im.at<Vec3b>(j,i)[1]<=sh
                    &&im.at<Vec3b>(j,i)[1]>=sl
                    &&im.at<Vec3b>(j,i)[2]<=vh
                    &&im.at<Vec3b>(j,i)[2]>=vl)

                    gray.at<uchar>(j,i)=255;
            else
                    gray.at<uchar>(j,i)=0;
        }
    }
    cvtColor(im,im,CV_HSV2BGR);
    return gray;
}



vector<RotatedRect> ArmorDetect(vector<RotatedRect> vEllipse)//height gao yi dian
{
    vector<RotatedRect> vRlt;
    RotatedRect Armor; //定义装甲区域的旋转矩形
    int nL, nW;
    double dAngle;
    vRlt.clear();
    if (vEllipse.size() < 2) //如果检测到的旋转矩形个数小于2，则直接返回
    {
        std::cout<<"less                two"<<rand()<<endl;
        return vRlt;
    }

    for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++) //求任意两个旋转矩形的夹角
    {
        for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
        {


            dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
            while (dAngle > 90)
                dAngle=180-dAngle;

            if (
                    (dAngle < T_ANGLE_THRE)

                    &&(abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) / (vEllipse[nI].size.height + vEllipse[nJ].size.height)) < T_SIZE_THRE


               )
//                    &&
//                    (abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) / (vEllipse[nI].size.width + vEllipse[nJ].size.width)) < T_SIZE_THRE) //判断这两个旋转矩形是否是一个装甲的两个LED等条
            {



                Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2; //装甲中心的x坐标
                Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2; //装甲中心的y坐标




                if (abs(vEllipse[nI].angle - vEllipse[nJ].angle)<90)  Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;   //装甲所在旋转矩形的旋转角度
                else {Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle-180) / 2;}

                if(Armor.angle>0) Armor.angle-=90;
                else {Armor.angle+=90;}

                    //Armor.angle += 90;
                nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2; //装甲的高度
                nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y)); //装甲的宽度等于两侧LED所在旋转矩形中心坐标的距离
                if (nL < nW)
                {
                    Armor.size.height = nW;
                    Armor.size.width = nL;

                    if((float)nW/nL<3.5)
                    vRlt.push_back(Armor); //将找出的装甲的旋转矩形保存到vector

//                    cout<<"asadadasdasdasdasdadadad                   "<<(float)nW/nL<<"         sd"<<(abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) / (vEllipse[nI].size.height + vEllipse[nJ].size.height))<<endl;

                }
            }
        }
    }
    return vRlt;
}



vector<RotatedRect> getContours(Mat gray)
{
    vector<vector<Point> > contour;
    RotatedRect s;   //定义旋转矩形
    vector<RotatedRect> result; //定以旋转矩形的向量，用于存储发现的目标区域


    findContours(gray, contour, RETR_CCOMP , CHAIN_APPROX_SIMPLE);


    for (int i=0; i<contour.size(); i++)
    {
        if(contour[i].size()> 5)
        {
            s = fitEllipse(contour[i]);
            result.push_back(s); //将发现的目标保存
        }
    }


    return result;
}


*/

