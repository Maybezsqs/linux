#include "windmill.h"

windmill::windmill()
{

}

double windmill::getDistance(const Point &A,const Point &B)
{
    double dis;
    dis=pow((A.x-B.x),2)+pow((A.y-B.y),2);
    return sqrt(dis);
}

vector<float> windmill::stander(Mat &im)
{
    if(im.empty()==1)
        cout<<"filed open"<<endl;
    resize(im,im,Size(48,48));
    vector<float> result;
    HOGDescriptor hog(cvSize(48,48),cvSize(16,16),cvSize(8,8),cvSize(8,8),9,1,-1,
                      HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);           //初始化HOG描述符
    hog.compute(im,result);
    return result;
}

Mat windmill::getSvmInput(Mat &input)
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

vector<Point2f> windmill::findArmor()
{
    // floodfill to find the armor
    Point seedPoint=Point(0,0);
    floodFill(img2armor,seedPoint,Scalar(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img2armor,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
    // filter the contours
    const float maxHWRatio=0.7153846;
    const float maxArea=2000;
    vector<Point2f> ret;
    if(hierarchy.size())
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
        // is an armor
        if(height/width<maxHWRatio&&area<maxArea&& findTarget==true/*&&pointPolygonTest(v,rect_tmp.center,10)>0*/)
        {
            // show the center
            Point centerP=rect_tmp.center;
            circle(srcImage,centerP,5,Scalar(0,0,255),4);
            // draw the target armor
            for(int j=0;j<4;++j)
                line(srcImage,P[j],P[(j+1)%4],Scalar(255,0,0),2);
            ret.push_back(centerP);
        }
    }
    return ret;
}
void windmill::getArmorLeafImg(Mat &srcImage)
{
    vector<Mat> imgChannels;
    split(srcImage,imgChannels);
    img2leaf=imgChannels.at(2)-imgChannels.at(0);//red-blue
    img2armor=img2leaf.clone();
    threshold(img2leaf,img2leaf,100,255,CV_THRESH_BINARY_INV);//floodfill to find armor
    threshold(img2armor,img2armor,100,255,CV_THRESH_BINARY);//find leaf
    const int structElementSize=1;//to dilate
    Mat element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
    dilate(img2leaf,img2leaf,element);
}
void windmill::loadSVM(string mod)
{
    ///load model
    svm=SVM::create();
//    svm=SVM::load("/home/ubuntu/文档/SVM/data/SVM4_9.xml");
    svm=SVM::load(mod);
}


vector<Point2f> windmill::findTargetLeaf()
{
    vector<vector<Point>> contours2;
    vector<Vec4i> hierarchy2;
    findContours(img2leaf,contours2,hierarchy2,RETR_CCOMP,CHAIN_APPROX_SIMPLE);

    centerPoint=Point(srcImage.cols/2,srcImage.rows/2);//use as a fixed point to locate the location of four points of a box

    RotatedRect rect_tmp2; // store the leaf boxs
    vector<Point2f> ret;
    if(hierarchy2.size())
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
            swap(height,width);// swap if inverted

        float area=width*height;// leaf area
        cout<<"dis #"<<getDistance(P[0],P[1])<<" dis 1#"<<width<< "area "<<area<<endl;
        if(getDistance(P[0],P[1])>width)// locate the four points to AffineTransform
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
        // show the points with  circuls
//            Point pnt=P[0];
//            circle(srcImage2,pnt,4,Scalar(0,0,255));
//            pnt=P[1];
//            circle(srcImage2,pnt,4,Scalar(0,255,0));
//            pnt=P[2];
//            circle(srcImage2,pnt,4,Scalar(255,0,0));
//            pnt=P[3];
//            circle(srcImage2,pnt,4,Scalar(255,255,255));

        // dist to affineTransform
        dstRect[0]=Point2f(0,0);
        dstRect[1]=Point2f(width,0);
        dstRect[2]=Point2f(width,height);
        Mat warp_mat=getAffineTransform(srcRect,dstRect);
        Mat warp_dst_map;

        warpAffine(img2leaf,warp_dst_map,warp_mat,warp_dst_map.size());
        if(area>5000){
            //save the leafs to train svm
//            string s="leaf"+to_string(cnnt)+".jpg";
//            cnnt++;

            // get the image of leaf
            Mat testim;
            testim = warp_dst_map(Rect(0,0,width,height));
//            if(testim.empty()==1)
//            {
//                cout<<"filed open"<<endl;
//                return -1;
//            }
            //get the vector to svm
            Mat test=getSvmInput(testim);
//                cout<<svm->predict(test)<<endl;
            // is target
            if(svm->predict(test)==1)
            {
                for(int k=0;k<4;++k)
                    ret.push_back(P[k]);
                findTarget=true;
                return ret;
            }
        }
    }
    findTarget=false;
    return ret;
}
//void windmill::init(const ArmorParam &armorParam)
//{
//    _param = armorParam;
//}

void windmill::loadImg(const cv::Mat & srcImg)
{
    srcImage = srcImg;

//#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
//	_debugImg = srcImg.clone();
//    Point pnt;
//    pnt.x=_debugImg.cols/2;
//    pnt.y=_debugImg.rows/2;
//    circle(_debugImg,pnt,4,Scalar(0,0,255));
//#endif // DEBUG_DETECTION || SHOW_RESULT

//	Rect imgBound = Rect(cv::Point(0, 0), _srcImg.size());

//    if(_flag == ARMOR_LOCAL && _trackCnt != _param.max_track_num)
//    {
//        cv::Rect bRect = boundingRect(_targetArmor.vertex) + _roi.tl();
//        bRect = cvex::scaleRect(bRect, Vec2f(3,2));	//以中心为锚点放大2倍
//        _roi = bRect & imgBound;
//        _roiImg = _srcImg(_roi).clone();
//    }
//    else
//    {
//		_roi = imgBound;
//		_roiImg = _srcImg.clone();
//		_trackCnt = 0;
//    }

//#ifdef DEBUG_DETECTION
//	rectangle(_debugImg, _roi, cvex::YELLOW);
//#endif // DEBUG_DETECTION
}












//int main(int argc, char *argv[])
//{
//    VideoCapture cap(0);
//    int cnnt=0;

////    cap.open("/home/ubuntu/视频/未命名文件夹/机器人视角 红色 背景暗.avi");
//    ///load model
//    Ptr<SVM> svm=SVM::create();
//    svm=SVM::load("/home/ubuntu/文档/SVM/data/SVM4_9.xml");
//    while(true)
//    {
//        Mat srcImage;
//        cap >> srcImage;
//        vector<Mat> imgChannels;
//        split(srcImage,imgChannels);
//        Mat midImage=imgChannels.at(2)-imgChannels.at(0);//red-blue
//        Mat midImage2=midImage.clone();
//        threshold(midImage,midImage,100,255,CV_THRESH_BINARY_INV);//floodfill to find armor
////        imshow("test1",midImage);
//        threshold(midImage2,midImage2,100,255,CV_THRESH_BINARY);//find leaf

//        const int structElementSize=1;//to dilate
//        Mat element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
//        dilate(midImage2,midImage2,element);
////        imshow("test2",midImage2);

//        vector<vector<Point>> contours2;
//        vector<Vec4i> hierarchy2;
//        findContours(midImage2,contours2,hierarchy2,RETR_CCOMP,CHAIN_APPROX_SIMPLE);

//        Mat srcImage2=srcImage.clone();

//        centerPoint=Point(srcImage.cols/2,srcImage.rows/2);//use as a fixed point to locate the location of four points of a box

//        RotatedRect rect_tmp2; // store the leaf boxs
//        bool findTarget=0;
//        vector<Point> v;
//        for(int i=0;i>=0;i=hierarchy2[i][0])
//        {
//            rect_tmp2=minAreaRect(contours2[i]);

//            Point2f P[4];
//            rect_tmp2.points(P);
//            Point2f srcRect[3];
//            Point2f dstRect[3];
//            float width=rect_tmp2.size.width;
//            float height=rect_tmp2.size.height;

//            if(height>width)
//                swap(height,width);// swap if inverted

//            float area=width*height;// leaf area
//            cout<<"dis #"<<getDistance(P[0],P[1])<<" dis 1#"<<width<< "area "<<area<<endl;
//            if(getDistance(P[0],P[1])>width)// locate the four points to AffineTransform
//            {
//                srcRect[0]=P[0];
//                srcRect[1]=P[1];
//                srcRect[2]=P[2];
//            }
//            else
//            {
//                srcRect[0]=P[1];
//                srcRect[1]=P[2];
//                srcRect[2]=P[3];
//            }
//            // show the points with  circuls
////            Point pnt=P[0];
////            circle(srcImage2,pnt,4,Scalar(0,0,255));
////            pnt=P[1];
////            circle(srcImage2,pnt,4,Scalar(0,255,0));
////            pnt=P[2];
////            circle(srcImage2,pnt,4,Scalar(255,0,0));
////            pnt=P[3];
////            circle(srcImage2,pnt,4,Scalar(255,255,255));

//            // dist to affineTransform
//            dstRect[0]=Point2f(0,0);
//            dstRect[1]=Point2f(width,0);
//            dstRect[2]=Point2f(width,height);
//            Mat warp_mat=getAffineTransform(srcRect,dstRect);
//            Mat warp_dst_map;

//            warpAffine(midImage2,warp_dst_map,warp_mat,warp_dst_map.size());
//            if(area>5000){
//                string s="leaf"+to_string(cnnt)+".jpg";
//                cnnt++;
//                //test
//                // get the image of leaf
//                Mat testim;
//                testim = warp_dst_map(Rect(0,0,width,height));
//                if(testim.empty()==1)
//                {
//                    cout<<"filed open"<<endl;
//                    return -1;
//                }
//                Mat test=getSvmInput(testim);
////                cout<<svm->predict(test)<<endl;
//                // is target
//                if(svm->predict(test)==1)
//                {
//                    v.push_back(P[0]);
//                    v.push_back(P[1]);
//                    v.push_back(P[2]);
//                    v.push_back(P[3]);

//                    findTarget=true;
////                    Point pnt=P[0];
////                    circle(srcImage2,pnt,10,Scalar(0,0,255),5);

//                    // floodfill to find the armor
//                    Point seedPoint=Point(0,0);
//                    floodFill(midImage,seedPoint,Scalar(0));
//                    vector<vector<Point>> contours;
//                    vector<Vec4i> hierarchy;
//                    findContours(midImage,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
//                    // filter the contours
//                    const float maxHWRatio=0.7153846;
//                    const float maxArea=2000;
//                    for(int i=0;i>=0;i=hierarchy[i][0])
//                    {
//                        RotatedRect rect_tmp=minAreaRect(contours[i]);
//                        Point2f P[4];
//                        rect_tmp.points(P);

//                        float width=rect_tmp.size.width;
//                        float height=rect_tmp.size.height;
//                        if(height>width)
//                            swap(height,width);
//                        float area=width*height;

//                        cout<<"width " << width << " height "<<height<<" Hwratio "<<height/width<<" area "<<area<<endl;
//                        if(height/width<maxHWRatio&&area<maxArea&& findTarget==true&&pointPolygonTest(v,rect_tmp.center,10)>0)
//                        {
//                            // show the center
//                            Point centerP=rect_tmp.center;
//                            circle(srcImage2,centerP,5,Scalar(0,0,255),4);
////                            Scalar color(0,255,0);

//                            // draw the target armor
//                            for(int j=0;j<4;++j)
//                            {
//                                line(srcImage2,P[j],P[(j+1)%4],Scalar(255,0,0),2);
//                            }
//                        }
//                    }
//                }
//            }
//            cout<<"width2 " << width << " height2 "<<height<<" Hwratio2 "<<height/width<<" area2 "<<area<<endl;
////            Scalar color(0,255,0);
////            for(int j=0;j<4;++j)
////            {
////                line(srcImage2,P[j],P[(j+1)%4],Scalar(0,255,0),2);
////            }
//        }
//        imshow("test22",srcImage2);
//        waitKey(0);
//    }
//    return 0;
//}
