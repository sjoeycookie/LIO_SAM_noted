#include "utility.h"
#include "lio_sam/cloud_info.h"

//!自定义点云类型
struct VelodynePointXYZIRT {          //创建点云结构体   
    PCL_ADD_POINT4D ; //位置
    PCL_ADD_INTENSITY;      //添加强度      
    uint16_t ring;                //扫描线
    float time;                      //时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW             //Eigen编译_Eigen向量化_内存对齐（确保定义新类型点云内存与SSE对齐）
} EIGEN_ALIGN16;   //强制SSE填充以正确对齐内存(16字节对齐)

//注册点类型宏
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,                                       
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

//!用不到
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;                // 添加pcl里xyz+padding(新增的类型)
    //新增的类型名称
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation(使用velodyne点云格式作为常见的表示)
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer              //子类：ImageProjection  公共继承,父类：ParamServer  
{
private:

    //*变量声明
    std::mutex imuLock;   //互斥锁
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;            //订阅对象
    ros::Publisher  pubLaserCloud;           //发布对象
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;          //双端队列imuQueue

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];               //开辟堆区指针
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;                                            //创建仿射对象

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;                                 //创建点云指针
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;                        //使用opencv中的mat类，创建变量rangeMat

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;                      //创建容器


public:
    //构造函数
    ImageProjection():deskewFlag(0)           //初始化赋值，deskewFlag = 0;  
    {   
        //!订阅imu+Odom+激光雷达点云，发布去畸变的点云+激光雷达点云信息
        //订阅imu,  imuTopic: "imu_correct"
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅imu里程计: 来自IMUPreintegration.cpp中的类IMUPreintegration发布的里程计话题（增量式）
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅激光点云，pointCloudTopic: "points_raw"
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        //发布去畸变的点云
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        //发布激光点云信息 （这里建议看一下自定义的lio_sam::cloud_info的msg文件 里面包含了较多信息）
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        //分配内存
        allocateMemory();
        //重置部分参数
        resetParameters();
        //*setVerbosityLevel用于设置控制台输出的信息
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);      //(pcl::console::L_ALWAYS)不会输出任何信息;L_DEBUG会输出DEBUG信息;L_ERROR会输出ERROR信息
    }

    void allocateMemory()
    {
        //用智能指针的reset方法在构造函数里面进行初始化
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        //根据params.yaml中给出的N_SCAN Horizon_SCAN参数值分配内存
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);  // 这里重新设置大小，就是行列的范围
        //cloudinfo是msg文件下自定义的cloud_info消息，对其中的变量进行赋值操作
        cloudInfo.startRingIndex.assign(N_SCAN, 0);      //(int size, int value):size-要分配的大小,value-要分配给向量名称的值
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);
        //重置部分参数
        resetParameters();
    }

    void resetParameters()
    {
        //清零操作
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        //初始全部用FLT_MAX 填充，因此后文函数projectPointCloud中有一句if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));    // Mat参数：行16，列1800 ，数据类型CV_32F,颜色填充,FLT_MAX是最大的float数值(float头文件中的两个宏)

        //变量赋值
        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)           //const int queueLength = 2000;
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);      //vector<int> columnIdnCountVec;  
    }

    ~ImageProjection(){}         //析构函数

    /**
        订阅原始imu数据
        1、imu原始测量数据转换到中间系，加速度、角速度、RPY
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)  //(回调函数)
    {
        //imuConverter在头文件utility.h中，作用是把imu数据转换到中间坐标系
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        //锁保护器:防止互斥锁抛出异常，能够正确的解锁互斥对象
        std::lock_guard<std::mutex> lock1(imuLock);              
        //把Imu数据放入队列
        imuQueue.push_back(thisImu);
        
        //!确认惯导输出是否正常
        //debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }
    
    //订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿(回调函数)
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {   
        //锁保护器
        std::lock_guard<std::mutex> lock2(odoLock);
        //把imu里程计放入队列
        odomQueue.push_back(*odometryMsg);
    }

    //激光点云回调函数
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        //添加激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性(是否含有ring或者time信息)
        if (!cachePointCloud(laserCloudMsg))               //为true代码继续运行
            return;
        // 当前帧起止时刻对应的imu数据、imu里程计数据处理
        if (!deskewInfo())                                                          //为true代码继续运行
            return;
        // 当前帧激光点云运动畸变校正
        // 1、检查激光点距离、扫描线是否合规
        // 2、激光运动畸变校正，保存激光点    
        projectPointCloud();
        // 提取有效激光点，存extractedCloud
        cloudExtraction();
        // 发布当前帧校正后点云，有效点
        publishClouds();
        // 重置参数，接收每帧lidar数据都要重置这些参数
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {   
        // cache point cloud
        // 点云数据保存进队列
        cloudQueue.push_back(*laserCloudMsg);
        // 确保队列里大于两帧数据
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        // 取出激光点云队列中最早的一帧
        currentCloudMsg = std::move(cloudQueue.front());         //std::move是将对象的状态或者所有权从一个对象转移到另一个对象，只是转移，没有内存的搬迁或者内存拷贝   链接：https://blog.csdn.net/nihao_2014/article/details/122384040
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)              //雷达为velodyne或者livox时
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);          // 转成pcl的点云格式 形参: (in,out)
        }
        //!代码用不到
        else if (sensor == SensorType::OUSTER)                 //雷达为ouster
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);         // 转成pcl的点云格式
            laserCloudIn->points.resize(tmpOusterCloudIn->size());                     //设置内存大小
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)            //取出tmpOusterCloudIn指向点中的信息放入laserCloudIn指向的点中
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        //获取时间戳
        cloudHeader = currentCloudMsg.header;
        //这一点的时间被记录下来，存入timeScanCur中，函数deskewPoint中会被加上laserCloudIn->points[i].time
        timeScanCur = cloudHeader.stamp.toSec();
        //可以看出lasercloudin中存储的time是一帧中距离起始点的相对时间,timeScanEnd是该帧点云的结尾时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        // is_dense: 若点云中的数据都是有限的(不包含inf/Nan这样的值)，则为true,否则为false
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        //todo 镭神不知道fields是否有ring,可能需要改动
        // 查看驱动里是否有每个点属于哪一根扫描scan这个信息
        static int ringFlag = 0;       
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")    // 检查ring这个field是否存在(veloodyne和ouster都有这个信息)
                {
                    ringFlag = 1;
                    break;
                }
            }
            // 如果没有这个信息就需要像loam或者lego loam那样手动计算scan id，现在velodyne的驱动里都会携带这些信息的
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // 同样，检查是否有时间戳信息
        if (deskewFlag == 0)                  //在构造函数中进行了，初始化赋值，int deskewFlag = 0;
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            //若没有时间戳信息，弹出警告
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    // 获取运动补偿所需的信息
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);               //锁保护器
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        //* 确保imu的数据覆盖这一帧的点云
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // 准备imu补偿的信息
        /*
          当前帧对应imu数据处理
            1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
            2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
            注：imu数据都已经转换到lidar系下了
            imu去畸变参数计算
        */
        imuDeskewInfo();

        /*
        当前帧对应imu里程计处理
            1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
            2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
            注：imu数据都已经转换到lidar系下了
            里程计去畸变参数计算
        */
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        
        cloudInfo.imuAvailable = false;
        //imuQueue队列不为空时
        while (!imuQueue.empty())
        {
            //从imu队列中删除当前激光帧0.01s前面时刻的imu数据
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01) 
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())      //imuQueue为空就退出
            return;

        imuPointerCur = 0;                            //int imuPointerCur;

        //遍历当前激光帧起止时刻（前后扩展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];      //取出imuQueue队列中的元素
            double currentImuTime = thisImuMsg.header.stamp.toSec();    //获取时间戳

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                 // 把imu的姿态转成欧拉角（提取imu姿态角RPY，作为当前lidar帧初始姿态角）
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);      //imuRPY2rosRPY在utility.h文件中定义

            // 超过当前激光帧结束时刻0.01s，结束（这一帧遍历完了就break）
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            // 取出IMU的角速度
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            //imu帧间时差
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            // 当前时刻旋转角 = 前一时刻旋转角 + 角速度 * 时差
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;  // 可以使用imu数据进行运动补偿
    }
    
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())          //里程计队列不为空
        {
            // 从imu里程计队列中删除当前激光帧起始时刻0.01s前面的imu数据
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())                   //里程计队列为空，退出函数
            return;
        // 要求必须有当前激光帧时刻之前的里程计数据
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)        //里程计起始时间戳大于当前帧激光起始时间戳，退出函数
            return;

        // get start odometry at the beinning of the scan(地图优化程序中发布的)
        nav_msgs::Odometry startOdomMsg;
        // 找到对应的最早的点云时间的odom数据  
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)      //ROS_TIME是utility.h中的函数，作用获取时间戳,&startOdomMsg是引用
                continue;
            else
                break;
        }

        // 提取imu里程计姿态角
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);      //将ros消息格式中的姿态转成tf的格式

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);        // 然后将四元数转成欧拉角

        // Initial guess used in mapOptimization
        // 记录点云起始时刻的对应的odom姿态（用当前激光帧起始时刻的imu里程计，初始化lidar位姿，后面用于mapOptmization）
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;          // odom提供了这一帧点云的初始位姿

        // get end odometry at the end of the scan
        odomDeskewFlag = false;                  // bool odomDeskewFlag;

        // 如果当前激光帧结束时刻之后没有imu里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;
        // 找到点云最晚时间对应的odom数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 如果起止时刻对应imu里程计的方差不等，返回（不等代表odom退化了，置信度就不高了）
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))         //round()函数作用：对数据进行四舍五入
            return;
        // 起始位姿转成Affine3f这个数据结构
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        //结束位姿转成Affine3f这个数据结构
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        
        // 计算起始位姿和结束位姿之间的增量 pose(当前激光帧起止时刻对应的imu里程计的相对变换)
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 将这个增量转成xyz和欧拉角的形式
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;  // 表示可以用odom来做运动补偿
    }

    //在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        //imuDeskewInfo中，对imuPointerCur进行计数(计数到超过当前激光帧结束时刻0.01s)
        while (imuPointerFront < imuPointerCur)                  //imuPointerCur是imu计算的旋转buffer的总共大小，这里用的就是一种朴素的确保不越界的方法
        {
            //imuTime在imuDeskewInfo（deskewInfo中调用，deskewInfo在cloudHandler中调用）被赋值，从imuQueue中取值
            if (pointTime < imuTime[imuPointerFront])      //pointTime为当前时刻,要找到imu积分列表里第一个大于当前时间的索引
                break;
            ++imuPointerFront;
        }

        //如果时间戳不在两个imu的旋转之间，就直接赋为离当前时刻最近的旋转增量了（即：点不在当前帧起止时刻对应的IMU数据队列中)
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];          //imuRotX等为之前积分出的内容.(imuDeskewInfo中)
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 否则 做一个线性插值，得到相对旋转（前后时刻插值计算当前时刻的旋转增量）
            int imuPointerBack = imuPointerFront - 1;//此时front的时间是大于当前pointTime时间，back=front-1刚好小于当前pointTime时间，前后时刻插值计算
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.
        //!如果运动比较快，可以尝试把下述代码注释去掉
        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        //这个来源于上文的时间戳通道和imu可用判断，没有或是不可用则返回点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        //点的时间等于scan时间加relTime（后文的laserCloudIn->points[i].time）
            /*
                lasercloudin中存储的time是一帧中距离起始点的相对时间
                在cloudHandler的cachePointCloud函数中，timeScanCur = cloudHeader.stamp.toSec();，即当前帧点云的初始时刻
                二者相加即可得到当前点的准确时刻
            */
        double pointTime = timeScanCur + relTime;

        //根据时间戳插值获取imu计算的旋转量与位置量（注意imu计算的相对于起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);            // 计算当前点相对起始点的相对旋转  

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);         // 这里没有计算平移补偿

        if (firstPointFlag == true)         // 这里的firstPointFlag来源于resetParameters函数，而resetParameters函数每次ros调用cloudHandler都会启动
        {
            // 计算第一个点的位姿并且求逆
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;    ///改成false以后，同一帧激光数据的下一次就不执行了
        }

        // 计算当前点和第一个点的相对位姿
        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        
        PointType newPoint;
        // 就是R × p + t，把点补偿到第一个点对应时刻的位姿
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }


    // 将点云投影到一个矩阵上。并且保存每个点的信息
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();     //当前帧的点云数据大小,用于下面一个个点投影
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            // 取出对应的某个点
            thisPoint.x = laserCloudIn->points[i].x;        //laserCloudIn就是原始的点云话题中，当前帧的点云数据
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 计算这个点距离lidar中心的距离
            float range = pointDistance(thisPoint);
            // 距离太小或者太远都认为是异常点
            if (range < lidarMinRange || range > lidarMaxRange)   //lidarMinRange: 1.0,lidarMaxRange: 1000.0
                continue;

            // 取出对应的在第几根scan上
            //! 如果没有这个信息就需要像loam或者lego loam那样手动计算scan id(可能需要改动，因为镭神雷达中不知道是否有)
            int rowIdn = laserCloudIn->points[i].ring;        
            // scan id必须合理
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 如果需要降采样，就根据scan id适当跳过
            if (rowIdn % downsampleRate != 0)      //downsampleRate= 1        
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)   
            {
                //水平角分辨率
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;  //horizonAngle 为[-180,180]
                static float ang_res_x = 360.0/float(Horizon_SCAN);   // Horizon_SCAN=1800,每格0.2度
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;     //即把horizonAngle从[-180,180]映射到[450,2250]
                // 大于等于1800，则减去1800，相当于把1800～2250映射到0～450 (将它的范围转换到了[0,1800) )
                   /*角度变化
                       (-180)   -----   (-90)  ------  0  ------ 90 -------180
                       变为:  90 ----180(-180) ---- (-90)  ----- (0)    ----- 90
                 */
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;    
            }
            //!用不到
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }
            // 对水平id进行检查
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)   //范围必须为[0,1800）
                continue;
            // 如果这个位置已经有填充了就跳过
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)       //初始rangMat矩阵全部用FLT_MAX填充    （Mat类中的at作用：用于获取图像矩阵某点的值或者改变某点的值）
                continue;
            // 对点做运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 将这个点的距离数据保存进这个rangeMat矩阵中
            rangeMat.at<float>(rowIdn, columnIdn) = range;   //range:点距离lidar中心的距离
            // 算出这个点的索引
            int index = columnIdn + rowIdn * Horizon_SCAN;
            // 保存这个点的坐标(存校正之后的激光点)
            fullCloud->points[index] = thisPoint;
        }
    }
    // 提取出有效的点的信息
    void cloudExtraction()
    {
        // 有效激光点数量
        int count = 0; 
        // extract segmented cloud for lidar odometry
        // 遍历每一根scan
        for (int i = 0; i < N_SCAN; ++i)
        {
            //提取特征的时候，每一行的前5个和最后5个不考虑
            //?感觉应该从第六个点算起(即：count+5)
            cloudInfo.startRingIndex[i] = count - 1 + 5;         //这个scan可以计算曲率的起始点（计算曲率需要左右各五个点）

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)     // 这是一个有用的点
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;    // 这个点对应着哪一根垂直线
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);  // 他的距离信息(点距离lidar中心的距离)
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);   //他的3d坐标信息
                    // size of extracted cloud
                    ++count;  // count只在有效点才会累加
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;      // 这个scan可以计算曲率的终点
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // 发布提取出来的有效的点
        /*
           publishCloud在utility.h头文件中,需要传入发布句柄pubExtractedCloud，提取出的有效点云，该帧时间戳,发布的数据来自那个坐标系(雷达坐标系)
                pubExtractedCloud定义在 ImageProjection类的构造函数中，用来发布去畸变的点云.
                extractedCloud主要在cloudExtraction中被提取，点云被去除了畸变
                cloudHeader.stamp 来源于currentCloudMsg,cloudHeader在cachePointCloud中被赋值currentCloudMsg.header，  而currentCloudMsg是点云队列cloudQueue中提取的
                lidarFrame: "base_link"
        */
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame); 
        pubLaserCloudInfo.publish(cloudInfo);   //todo   pubExtractedCloud发布的只有点云信息，而pubLaserCloudInfo发布的为自定义的很多信息
    }
};

//!主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");       //初始化节点   

    ImageProjection IP;             //创建类对象
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");         //输出日志消息

    ros::MultiThreadedSpinner spinner(3);        //多线程，使用3个线程
    spinner.spin();                   //循环调用
    
    return 0;
}
