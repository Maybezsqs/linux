#include<unistd.h>
#include"iostream"
#include<opencv2/opencv.hpp>
#include<thread>
#include<list>

#include"ImgProdCons.h"
#include"./Armor/ArmorDetector.h"
#include"./Pose/AngleSolver.hpp"

using namespace std;
using namespace cv;

#define DEBUG
//#define GET_VIDEO

namespace rm
{

void ImgProdCons::init()
{
    //prevent usb port from being blocked
    init_signals();


    //Initialize camera
    _videoCapturePtr->open(0,2); // 0 works fine for small panel , if you meet problem opening the camera, try change this
    _videoCapturePtr->setVideoFormat(640, 480, true);
//        _videoCapturePtr->setVideoFormat(1280, 720, true);
    _videoCapturePtr->setExposureTime(60/*100*/);
    //    _videoCapturePtr->setExposureTime(200/*100*/);
    _videoCapturePtr->setFPS(120);
    //    _videoCapturePtr->setFPS(60);
    _videoCapturePtr->startStream();
    _videoCapturePtr->info();

    //Initilize serial
//    _serialPtr->openPort();
    _serialPtr->setDebug(false);
    int self_color=BLUE;
//    while(_serialPtr->setup(self_color) != Serial::OJBK)
//    {
//        sleep(1);
//    }
    cout << "I am " << (self_color == rm::BLUE ? "blue" : "red") << "." << endl;


    //Initialize angle solver
    AngleSolverParam angleParam;
    angleParam.readFile(8);
    _solverPtr->init(angleParam);
    _solverPtr->setResolution(_videoCapturePtr->getResolution());


    //Initialize armor detector
    ArmorParam armorParam;
    _armorDetectorPtr->init(armorParam);
    _armorDetectorPtr->setEnemyColor(self_color == rm::BLUE ? rm::RED : rm::BLUE);
}

//void updateFeeling(list<double>& feeling, const double duration)
//{
//    for(auto it = feeling.begin(); it != feeling.end(); it++)
//    {
//        if(*it < 10)
//        {
//            feeling.erase(it);
//        }
//        else
//        {
//            *it *= exp(0.1*duration);
//        }
//    }
//}

void ImgProdCons::sense()
{
    //	chrono::time_point<chrono::steady_clock> lastTime;
    //	list<double> worrys0, worrys1, pains;
    //	double totalWorry0, totalWorry1, totalPain;
    //	const double TAU = 0.1;
    //	const double M = 10;
    //	int remainHp = 3000;

    /* Loop for sensing */
    for(;;)
    {
        FeedBackData feedBackData;

        /* TODO: Handel exceptions when socket suddenly being plugged out. */
        if(_serialPtr->tryFeedBack(feedBackData, chrono::milliseconds(20)) == Serial::OJBK)
        {
            //			const auto nowTime = chrono::high_resolution_clock::now();
            //			const auto duration = (static_cast<chrono::duration<double, std::milli>>(nowTime - lastTime)).count();

            //			/*	Update historic worrys and pain */
            //			updateFeeling(worrys0, duration);
            //			updateFeeling(worrys1, duration);
            //			updateFeeling(pains, duration);

            //TODO: add other states
            _task = feedBackData.task_mode;

            //			auto pain = remainHp - feedBackData.remain_hp;
            //			remainHp = feedBackData.remain_hp;
            //			if(feedBackData.shot_armor == 0)
            //			{
            //				worrys0.push_back(pain);
            //				pains.push_back(pain);
            //			}
            //			else if(feedBackData.shot_armor == 1)
            //			{
            //				worrys1.push_back(pain);
            //				pains.push_back(pain);
            //			}

            //			totalWorry0 = accumulate()


            //			lastTime = nowTime;
            this_thread::sleep_for(chrono::milliseconds(80));
        }
        else
        {
            this_thread::sleep_for(chrono::milliseconds(50));
        }
    }
}

using namespace cv;
using namespace cv::ml;
using namespace std;
double getDistance(Point A,Point B)
{
    double dis;
    dis=pow((A.x-B.x),2)+pow((A.y-B.y),2);
    return sqrt(dis);
}
Point centerPoint;
vector<float> stander(Mat im)
{

    if(im.empty()==1)
    {
        cout<<"filed open"<<endl;
    }
    resize(im,im,Size(48,48));

    vector<float> result;

    HOGDescriptor hog(cvSize(48,48),cvSize(16,16),cvSize(8,8),cvSize(8,8),9,1,-1,
                      HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);           //初始化HOG描述符
    hog.compute(im,result);

    return result;
}

Mat get(Mat input)
{
    vector<float> vec=stander(input);
    if(vec.size()!=900) cout<<"wrong1 not 900"<<endl;
    Mat output(1,900,CV_32FC1);

    Mat_<float> p=output;
    int jj=0;
    for(vector<float>::iterator iter=vec.begin();iter!=vec.end();iter++,jj++)
    {
        p(0,jj)=*(iter);
    }
    return output;
}


void ImgProdCons::consume()
{
    /*
     * Variables for recording camera
     */
    VideoWriter writer;
    bool isRecording = false;
    time_t t;
    time(&t);
    const string fileName = "/home/nvidia/Robomaster/Robomaster2018/" + to_string(t) + ".avi";
    writer.open(fileName, CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280, 720));


    /* Variables for serial communication*/
    ControlData controlData;

    /* Variables for angle solve module */
    int angleFlag;
    Vec2f targetAngle;

    /* Variables for armor detector modeule */
    int armorFlag;
    int armorType;
    std::vector<cv::Point2f> armorVertex;

    /* The main loop for armor detection */
    auto t1 = chrono::high_resolution_clock::now();
    Frame frame;
    int cnt=0;
    for(;;)
    {
        if(/*_serialPtr->getErrorCode() == Serial::SYSTEM_ERROR || */!_videoCapturePtr->isOpened())
        {
            this_thread::sleep_for(chrono::seconds(3));
        }
        // comment this part when bebugging without STM upper computer
        //        if(_task != Serial::AUTO_SHOOT)
        //        {
        //            cout << "waiting for command." << endl;
        //            continue;
        //        }
        if(!_buffer.getLatest(frame)) continue;
        _armorDetectorPtr->loadImg(frame.img);
        // test
        Mat test1=frame.img.clone();
        imshow("test1",test1);
        waitKey(20);

        //    cap.open("/home/ubuntu/视频/未命名文件夹/机器人视角 红色 背景暗.avi");
            ///load model
            Ptr<SVM> svm=SVM::create();
            svm=SVM::load("/home/ubuntu/文档/SVM/data/SVM4_9.xml");
            while(true)
            {
                Mat srcImage;
                srcImage=test1;
                vector<Mat> imgChannels;
                split(srcImage,imgChannels);
                Mat midImage=imgChannels.at(2)-imgChannels.at(0);
                Mat midImage2=midImage.clone();
                threshold(midImage,midImage,100,255,CV_THRESH_BINARY_INV);
        //        imshow("test1",midImage);
                threshold(midImage2,midImage2,100,255,CV_THRESH_BINARY);
                const int structElementSize=1;
                Mat element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
                dilate(midImage2,midImage2,element);
        //        imshow("test2",midImage2);
                vector<vector<Point>> contours2;
                vector<Vec4i> hierarchy2;
                findContours(midImage2,contours2,hierarchy2,RETR_CCOMP,CHAIN_APPROX_SIMPLE);

                Mat srcImage2=srcImage.clone();

                centerPoint=Point(srcImage.cols/2,srcImage.rows/2);
                RotatedRect rect_tmp2;
                bool findTarget=0;
                vector<Point> v;
                for(int i=0;i>=0;i=hierarchy2[i][0])
                {
                    rect_tmp2=minAreaRect(contours2[i]);
                    Point2f P[4];
                    rect_tmp2.points(P);
                    Point2f srcRect[3];
                    Point2f dstRect[3];
                    float width=rect_tmp2.size.width;
                    float height=rect_tmp2.size.height;
                    if(height>width)
                        swap(height,width);
                    float area=width*height;
                    cout<<"dis #"<<getDistance(P[0],P[1])<<" dis 1#"<<width<< "area "<<area<<endl;
                    if(getDistance(P[0],P[1])>width)
                    {
                        srcRect[0]=P[0];
                        srcRect[1]=P[1];
                        srcRect[2]=P[2];
                    }
                    else
                    {
                        srcRect[0]=P[1];
                        srcRect[1]=P[2];
                        srcRect[2]=P[3];
                    }

        //            Point pnt=P[0];
        //            circle(srcImage2,pnt,4,Scalar(0,0,255));
        //            pnt=P[1];
        //            circle(srcImage2,pnt,4,Scalar(0,255,0));
        //            pnt=P[2];
        //            circle(srcImage2,pnt,4,Scalar(255,0,0));
        //            pnt=P[3];
        //            circle(srcImage2,pnt,4,Scalar(255,255,255));

                    dstRect[0]=Point2f(0,0);
                    dstRect[1]=Point2f(width,0);
                    dstRect[2]=Point2f(width,height);
                    Mat warp_mat=getAffineTransform(srcRect,dstRect);
                    Mat warp_dst_map;
                    warpAffine(midImage2,warp_dst_map,warp_mat,warp_dst_map.size());
                    if(area>5000){

//                        string s="leaf"+to_string(cnnt)+".jpg";
//                        cnnt++;
                        //test

                        Mat testim;
                        testim = warp_dst_map(Rect(0,0,width,height));
//                        if(testim.empty()==1)
//                        {
//                            cout<<"filed open"<<endl;
//                            return -1;
//                        }

                        Mat test=get(testim);

        //                cout<<svm->predict(test)<<endl;
                        if(svm->predict(test)==1)
                        {
                            v.push_back(P[0]);
                           v.push_back(P[1]);
                           v.push_back(P[2]);
                           v.push_back(P[3]);
                            findTarget=true;
                            Point pnt=P[0];
        //                    circle(srcImage2,pnt,10,Scalar(0,0,255),5);
                            Point seedPoint=Point(0,0);
                            floodFill(midImage,seedPoint,Scalar(0));
                            vector<vector<Point>> contours;
                            vector<Vec4i> hierarchy;
                            findContours(midImage,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
                            const float maxHWRatio=0.7153846;
                            const float maxArea=2000;
                            for(int i=0;i>=0;i=hierarchy[i][0])
                            {
                                RotatedRect rect_tmp=minAreaRect(contours[i]);
                                Point2f P[4];
                                rect_tmp.points(P);


                                float width=rect_tmp.size.width;
                                float height=rect_tmp.size.height;
                                if(height>width)
                                    swap(height,width);
                                float area=width*height;

                                cout<<"width " << width << " height "<<height<<" Hwratio "<<height/width<<" area "<<area<<endl;
                                if(height/width<maxHWRatio&&area<maxArea&& findTarget==true&&pointPolygonTest(v,rect_tmp.center,10)>0)
                                {

                                    Point centerP=rect_tmp.center;
                                    circle(srcImage2,centerP,5,Scalar(0,0,255),4);
        //                            Scalar color(0,255,0);
                                    for(int j=0;j<4;++j)
                                    {
                                        line(srcImage2,P[j],P[(j+1)%4],Scalar(255,0,0),2);
                                    }
                                }
                            }
                        }
                    }

                    cout<<"width2 " << width << " height2 "<<height<<" Hwratio2 "<<height/width<<" area2 "<<area<<endl;
        //            Scalar color(0,255,0);
        //            for(int j=0;j<4;++j)
        //            {
        //                line(srcImage2,P[j],P[(j+1)%4],Scalar(0,255,0),2);
        //            }

                }
                imshow("test22",srcImage2);
                imshow("test",srcImage);
                waitKey(30);

            }


//        armorFlag = _armorDetectorPtr->detect();
//        if(armorFlag == ArmorDetector::ARMOR_LOCAL || armorFlag == ArmorDetector::ARMOR_GLOBAL)
//        {
//            armorVertex = _armorDetectorPtr->getArmorVertex();
//            armorType = _armorDetectorPtr->getArmorType();

//            _solverPtr->setTarget(armorVertex, armorType);
//            for(int j=0;j<4;++j)
//            {
//                line(test1,armorVertex[j],armorVertex[(j+1)%4],Scalar(255,0,0),2);
//            }
//            imshow("test",test1);
//            angleFlag = _solverPtr->solve();
//            if(angleFlag != rm::AngleSolver::ANGLE_ERROR)
//            {
////                targetAngle = _solverPtr->getCompensateAngle();
//                targetAngle = _solverPtr->getAngle();
//                controlData.frame_seq   = frame.seq;
//                controlData.shoot_mode  = Serial::BURST_FIRE | Serial::HIGH_SPEED;
//                controlData.pitch_dev   = targetAngle[1];
//                controlData.yaw_dev     = targetAngle[0];
//                //                controlData.speed_on_rail = 0;

//                controlData.gimbal_mode = Serial::SERVO_MODE;
//                if(_serialPtr->tryControl(controlData, chrono::milliseconds(3)) != Serial::OJBK)
//                {
//                    cout<<"not sent"<<endl;
//                }
////                cout << "Deviation: " << targetAngle << endl;
//            }
//        }
//        else
//        {
//            cnt++;
////            if(cnt>10){
//                controlData.pitch_dev   = 0;
//                controlData.yaw_dev     = 0;
////                cnt=0;
////            }
//            if(_serialPtr->tryControl(controlData, chrono::milliseconds(3)) != Serial::OJBK)
//            {
//                cout<<"not sent"<<endl;
//            }

//            controlData.gimbal_mode = Serial::PATROL_AROUND;
//        }
//        cout <<"\033[2J";
////        Mat TT=_solverPtr->getT();
////        cout << "get T : "<<TT.at<double>(0, 0)<<"  "<<TT.at<double>(1, 0)<<"  "<<TT.at<double>(2, 0)<<endl;
//        cout << "Deviation: " << targetAngle << endl;
//        _solverPtr->showPoints2dOfArmor();
//        _solverPtr->showTvec();
//        _solverPtr->showEDistance();
//        _solverPtr->showcenter_of_armor();
//        _solverPtr->showAngle();
//        _solverPtr->showAlgorithm();
#ifdef DEBUG
        auto t2 = chrono::high_resolution_clock::now();
        cout << "Total period: " << (static_cast<chrono::duration<double, std::milli>>(t2 - t1)).count() << " ms" << endl;
        cout << endl;
        t1 = t2;
#endif

#ifdef GET_VIDEO
        if(isRecording)
        {
            writer << frame.img;
        }
        if(!writer.isOpened())
        {
            cout << "Capture failed." << endl;
            continue;
        }
        isRecording = true;
        cout << "Start capture. " + fileName +" created." << endl;
        if(waitKey(1) == 'q')
        {
            return;
        }
#endif

    }
}
}
