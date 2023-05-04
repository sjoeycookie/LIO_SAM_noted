#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
//!自定义点类型   
struct PointXYZIRPYT                   //创建点云结构体 
{
    PCL_ADD_POINT4D                         // preferred way of adding a XYZ+padding
    PCL_ADD_INTENSITY;                 //添加强度   
    //新增的类型名称
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned    Eigen编译_Eigen向量化_内存对齐（确保定义新类型点云内存与SSE对齐）
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment      强制SSE填充以正确对齐内存(16字节对齐)

//注册点类型宏
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam中的变量
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;         
    //*发布对象创建
    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubSLAMInfo;
    //*订阅对象
    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    //*创建服务对象变量
    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;   //创建容器
    lio_sam::cloud_info cloudInfo;                         //创建自定义消息变量
    // 历史所有关键帧的角点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;         //容器（类型：点云的智能指针）
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）
    
    //智能指针（pcl点云类型）
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;           // 存储关键帧的位置信息的点云 （历史关键帧位姿（位置））
    // 历史关键帧位姿
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;        // 存储关键帧的6D位姿信息的点云   （PointTypePose为自定义的点类型）
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 当前激光帧角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization  （当前激光帧平面点集合）
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization （当前激光帧角点集合，降采样，DS: DownSize）
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization （当前激光帧平面点集合，降采样）

    // 当前帧与局部map匹配上了的角点、平面点，加入同一集合；后面是对应点的参数
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    //*容器
    // 当前帧与局部map匹配上了的角点、参数、标记
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    // 当前帧与局部map匹配上了的平面点、参数、标记
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    // 局部地图的一个容器
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;      

    //智能指针（点云类型）
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;    // 局部map的角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;            // 局部map的平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;   // 角点局部地图的下采样后的点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;         // 面点局部地图的下采样后的点云

    //*智能指针（kdTree类型）
    // 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;     // 角点局部地图的kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;           // 面点局部地图的kdtree
 
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    //互斥锁
    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;      // 当前局部地图下采样后的角点数目
    int laserCloudSurfFromMapDSNum = 0;      // 当前局部地图下采样后的面点数目
    int laserCloudCornerLastDSNum = 0;      // 当前帧下采样后的角点的数目
    int laserCloudSurfLastDSNum = 0;           // 当前帧下采样后的面点数目

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    //*容器变量
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;           //双端队列

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;                //?当前帧位姿
    Eigen::Affine3f incrementalOdometryAffineFront;     // 前一帧位姿
    Eigen::Affine3f incrementalOdometryAffineBack;      // 当前帧位姿

    //*构造函数
    mapOptimization()
    {
        //*gtsam初始化
        //ISAM2的参数类
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;    //重新线性化
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);           //优化器根据参数重新生成优化器，并开辟堆区  （ISAM2 *isam）

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);   // 发布历史关键帧里程计
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);  // 发布局部关键帧map的特征点云
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);   //*发布激光里程计，rviz中表现为坐标轴
        //*发布激光里程计，它与上面的激光里程计基本一样，只是roll、pitch用imu数据加权平均了一下，z做了限制
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);   // 发布激光里程计路径，rviz中表现为载体的运行轨迹

        //*订阅当前激光帧点云信息，来自featureExtraction
        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        //!订阅GPS里程计(可以考虑注释掉，因为没有GPS)
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        //todo 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上(可以考虑注释掉)
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        //*订阅一个保存地图功能的服务(创建服务对象)
        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);
        
        // 发布闭环匹配关键帧局部map
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        // 发布闭环边，rviz中表现为闭环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        // 发布局部map的降采样平面点集合
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        // 发布历史帧（累加的）的角点、平面点降采样集合
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        // 发布当前帧原始点云配准之后的点云       
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);
        //发布地图点云信息
        pubSLAMInfo           = nh.advertise<lio_sam::cloud_info>("lio_sam/mapping/slam_info", 1);

        //*设置降采样体素网格，滤波叶子的大小
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);  //mappingCornerLeafSize: 0.2， 0.2*0.2*0.2
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize); //mappingSurfLeafSize: 0.4
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);       //0.4*0.4*0.4
        //surroundingKeyframeDensity: 2.0 
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
        //分配内存
        allocateMemory();
    }

    void allocateMemory()
    {
        //*智能指针使用reset函数初始化
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());
        
        //*容器预先开辟空间大小
        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);        //将容器起始到结束全部填充为0
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        //*智能指针使用reset函数初始化
        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;          //将float类型数组初始化为0
        }
        //*创建一个6*6的图像矩阵，元素全部初始化为0
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0)); //参数1：高(行)  ，参数2：宽（列)， 参数3：数据类型 （此处为单通道，float类型），参数4：填充的值
    }
    
    //*订阅当前激光帧点云信息的回调函数，来自featureExtraction
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;                      // 提取当前激光帧时间戳
        timeLaserInfoCur = msgIn->header.stamp.toSec();          //转成ROS的时间格式

        // extract info and feature cloud
        // 提取cloudinfo中的角点和面点（提取当前激光帧角点、平面点集合）
        cloudInfo = *msgIn;                                                        
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);        //ROS消息格式转成pcl格式
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);                 //锁保护器

        static double timeLastProcessing = -1;
        // 控制后端频率，两帧处理一帧
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)      //mappingProcessInterval: 0.15 
        { 
            timeLastProcessing = timeLaserInfoCur;
            /* 当前帧位姿初始化
                1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
                2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光初始位姿
            */
            updateInitialGuess();       // 更新当前匹配结果的初始位姿

            /*提取局部角点、平面点云集合，加入局部map
                 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
                 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
            */
            extractSurroundingKeyFrames();         // 提取当前帧相关的关键帧并且构建点云局部地图

           // 当前激光帧角点、平面点集合降采样
            downsampleCurrentScan();

             /* scan-to-map优化当前帧位姿
                1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
                2、迭代30次（上限）优化
                    1) 当前激光帧角点寻找局部map匹配点
                        a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
                        b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
                    2) 当前激光帧平面点寻找局部map匹配点
                        a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
                        b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
                    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
             */
            scan2MapOptimization();       // 对点云配准进行优化问题构建求解
          
            /* 设置当前帧为关键帧并执行因子图优化
                1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
                2、添加激光里程计因子、GPS因子、闭环因子
                3、执行因子图优化
                4、得到当前帧优化后位姿，位姿协方差
                5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            */
            saveKeyFramesAndFactor();     // 根据配准结果确定是否是关键帧

            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
            correctPoses();         // 调整全局轨迹

            // 将lidar里程记信息发送出去（ 发布激光里程计）
            publishOdometry();

            /* 发布里程计、点云、轨迹
                1、发布历史关键帧位姿集合
                2、发布局部map的降采样平面点集合
                3、发布历史帧（累加的）的角点、平面点降采样集合
                4、发布里程计轨迹
            */
            publishFrames();         // 发送可视化点云信息
        }
    }
    //收集GPS信息（gps回调函数）
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }
    // 根据当前帧位姿，变换到世界坐标系（map系）下
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        // 使用openmp进行并行加速 
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            // 每个点都施加RX+t这样一个过程
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        //note:getTransformation的变换矩阵  T = Z1Y2X3  （RPY形式）
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      //!存储文件的路径，可以修改
      //*如果是空，说明是程序结束后的自动保存，否则是中途调用ros的service发布的保存指令
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;     //savePCDDirectory: "/Downloads/LOAM/" 
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;

      // create directory and remove old files;
      // *删掉之前有可能保存过的地图
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str()); //删除旧的文件夹
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());  //创建新的文件夹

      // save key frame transformations
      // 首先保存关键帧轨迹
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      //创建智能指针
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

       // 遍历所有关键帧，将点云全部转移到世界坐标系下去
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          // 类似进度条的功能  参考文章链接：https://blog.csdn.net/Aliven888/article/details/111184891
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";  //std::flush(刷新流缓存区)
      }
      
      // 如果没有指定分辨率，就是直接保存
      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        // 使用指定分辨率降采样，分别保存角点地图和面点地图
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;
      // 保存全局地图
      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);    //mappingCornerLeafSize: 0.2   
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);   //mappingSurfLeafSize: 0.4  

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    // 全局可视化线程
    void visualizeGlobalMapThread()
    {
        // 更新频率设置为0.2hz
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }
        // 当ros被杀死之后，执行保存地图功能
        if (savePCD == false)          //savePCD: false    直接返回了
            return;
        //!下面代码没有执行
        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    // 发布可视化全局地图
    void publishGlobalMap()
    {
        // 如果没有订阅者就不发布，节省系统负载
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;
        // 没有关键帧自然也没有全局地图了
        if (cloudKeyPoses3D->points.empty() == true)
            return;
        
        //创建智能指针
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        //创建容器
        std::vector<int> pointSearchIndGlobalMap;   
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();

        // 把所有关键帧送入kdtree
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 寻找距离最新关键帧一定范围内的其他关键帧
        //*globalMapVisualizationSearchRadius: 1000.0  m
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        //把这些找到的关键帧的位姿保存起来
        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // downsample near selected key frames
        //做一个简单的下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        //*globalMapVisualizationPoseDensity: 10.0
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            // 找到这些下采样后的关键帧的索引，并保存下来
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)         //1000m
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 将每一帧的点云通过位姿转到世界坐标系下
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        // 转换后的点云也进行一个下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        //*globalMapVisualizationLeafSize: 1.0     
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // 最终发布出去
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }











    /** 闭环线程
      1、闭环scan-to-map，icp优化位姿
        1) 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
        2) 提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        3) 执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
      2、rviz展示闭环边
    */
    void loopClosureThread()
    {
        // 如果不需要进行回环检测，那么就退出这个线程
        if (loopClosureEnableFlag == false)
            return;
        // 设置回环检测的频率
        ros::Rate rate(loopClosureFrequency);    //loopClosureFrequency: 1.0  HZ
        while (ros::ok())
        {  
            // 执行完一次就必须sleep一段时间，否则该线程的cpu占用会非常高
            rate.sleep();
            // 执行回环检测
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        // 如果没有关键帧，就没法进行回环检测了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        // 把存储关键帧的位姿的点云copy出来，避免线程冲突
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;   // 当前关键帧索引
        int loopKeyPre;   //候选闭环匹配帧索引
        // 首先看一下外部通知的回环信息
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)  //*detectLoopClosureExternal()函数没有使用，结果为false
            // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧(然后根据里程记的距离来检测回环)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 提取当前关键帧特征点集合，降采样;
            //loopFindNearKeyframes形参分别为点云集合，当前帧的索引，搜索半径
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);    //historyKeyframeSearchNum: 25  
            // 如果特征点较少，返回
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            // 发布闭环匹配关键帧局部map
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                // 把局部地图发布出去供rviz可视化使用
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        // 使用简单的icp来进行帧到局部地图的配准
        // ICP参数设置    网页链接：https://pcl.readthedocs.io/projects/tutorials/en/master/iterative_closest_point.html#iterative-closest-point
        static pcl::IterativeClosestPoint<PointType, PointType> icp;  //创建ICP实例
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);    // historyKeyframeSearchRadius: 15.0     
        icp.setMaximumIterations(100); //迭代次数已达到用户施加的最大迭代次数
        icp.setTransformationEpsilon(1e-6);  //先前转换和当前估计转换（即两次位姿转换）之间的 epsilon（差异）小于用户施加的值
        icp.setEuclideanFitnessEpsilon(1e-6);   //欧几里得平方误差的总和小于用户定义的阈值
        icp.setRANSACIterations(0); // 设置RANSAC运行次数

        // Align clouds
        // 设置两个点云
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());  
        // 执行点云配准
        icp.align(*unused_result); //存放结果点云

        // 未收敛，或者匹配不够好
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)      //icp.hasConverged() =true,意味着两个点云正确对齐，得分越小越好， historyKeyframeFitnessScore: 0.3
            return;

        // publish corrected cloud
        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());  //函数的作用是通过转换矩阵(icp.getFinalTransformation())将cureKeyframeCloud转换后存到closed_cloud中保存
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        // 获得两个点云的变换矩阵结果
        correctionLidarFrame = icp.getFinalTransformation();

        // transform from world origin to wrong pose
        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        // 闭环优化后当前帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);

        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // 闭环匹配帧的位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        // 使用icp的得分作为他们的约束的噪声项
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // 添加闭环因子需要的数据
        //这些内容会在函数addLoopFactor中用到
        mtx.lock();
        // 将两帧索引，两帧相对位姿和噪声作为回环约束送入队列
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        // 保存已存在的约束对
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 当前帧已经添加过闭环对应关系，不再继续添加( 检查一下较晚帧是否和别的形成了回环，如果有就算了)
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
        //配置文件中默认historyKeyframeSearchRadius=15m
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        // 把只包含关键帧位移信息的点云填充kdtree
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 根据最后一个关键帧的平移信息，寻找离他一定距离内的其他关键帧
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        

        // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
        //配置文件中默认30s
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)     //遍历找到的候选关键帧
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)  //historyKeyframeSearchTimeDiff: 30.0    
            {
                loopKeyPre = id;
                break;
            }
        }

        // 如果没有找到回环或者回环找到自己身上去了，就认为是此次回环寻找失败
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)      //没找到候选匹配帧
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    //!函数没有使用(外部闭环检测)
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        // 外部的先验回环消息队列
        if (loopInfoVec.empty())          //队列为空返回false
            return false;
        // 取出回环信息，这里是把时间戳作为回环信息
        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();
        // 如果两个回环帧之间的时间差小于30s就算了
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)     //historyKeyframeSearchTimeDiff: 30.0    
            return false;
        // 如果现存的关键帧小于2帧那也没有意义
        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        // 遍历所有的关键帧，找到离后面一个时间戳更近的关键帧的id
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        // 同理，找到距离前一个时间戳更近的关键帧的id
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
        // 两个是同一个关键帧，就没什么意思了吧
        if (loopKeyCur == loopKeyPre)
            return false;
        // 检查一下较晚的这个帧有没有被其他时候检测了回环约束，如果已经和别的帧形成了回环就算了
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;
        // 两个帧的索引输出
        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        //通过-searchNum 到 +searchNum，搜索key两侧内容
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            // 如果超出范围就算了
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            // 否则把对应角点和面点的点云转到世界坐标系下去
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }
        // 如果没有有效的点云就算了
        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        // 降采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    // 将已有的回环约束可视化出来
    void visualizeLoopClosure()
    {
        // 如果没有回环约束就算了
        if (loopIndexContainer.empty())
            return;
        
        //visualization_msgs::Marker相关知识链接：http://docs.ros.org/en/api/visualization_msgs/html/msg/Marker.html
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        // 回环约束的两帧作为node添加
        // 先定义一下node的信息
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;     //球序列
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        // 两帧之间约束作为edge添加
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;                           //坐标系
        markerEdge.header.stamp = timeLaserInfoStamp;                     //时间戳
        markerEdge.action = visualization_msgs::Marker::ADD;              //ADD=0          0 添加/修改一个对象，1（不推荐使用），2删除一个对象，3删除所有对象 
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;    //对象类型(LINE_LIST=5) 线条寻列
        markerEdge.ns = "loop_edges";                       //为对象创建唯一名称
        markerEdge.id = 1;                                                 //对象 ID 与命名空间一起用于稍后操作和删除对象
        markerEdge.pose.orientation.w = 1;               //位姿中的四元数
        markerEdge.scale.x = 0.1;                               //对象的比例1,1,1表示默认（通常为1米square)
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0; //Color [0.0-1.0]
        markerEdge.color.a = 1;      //透明度
        // 遍历所有回环约束
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            // 添加node和edge
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        // 最后发布出去供可视化使用
        pubLoopConstraintEdge.publish(markerArray);
    }







    


    // 作为基于优化方式的点云匹配，初始值是非常重要的，一个好的初始值会帮助优化问题快速收敛且避免局部最优解的情况
    void updateInitialGuess()
    {
        // save current transformation before any processing
        // 前一帧的位姿，注：这里指lidar的位姿，后面都简写成位姿 (transformTobeMapped数组中初始值为0)
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped); //trans2Affine3f函数主要用于获取位姿变换信息(x,y,z,roll,pitch,yaw)
        
        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // 如果关键帧集合为空，继续进行初始化
        if (cloudKeyPoses3D->points.empty())
        {
            //当前帧位姿的旋转部分，用激光帧信息中的RPY（来自imu原始数据）初始化）
            transformTobeMapped[0] = cloudInfo.imuRollInit;          
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;
            // 无论vio还是lio 系统的不可观都是4自由度，平移+yaw角，这里虽然有磁力计将yaw对齐，但是也可以考虑不使用yaw
            if (!useImuHeadingInitialization)       //!useImuHeadingInitialization: true, 需要改动，因为没有GPS
                transformTobeMapped[2] = 0;
            // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的imu里程计位姿（旋转部分）
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
        
        // use imu pre-integration estimation for pose guess
        //* 用当前帧和前一帧对应的imu里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        /*odomAvailable和imuAvailable均来源于imageProjection.cpp中赋值，
            imuAvailable是遍历激光帧前后起止时刻0.01s之内的imu数据，
            如果都没有那就是false，因为imu频率一般比激光帧快，因此这里应该是都有的。
            odomAvailable同理，是监听imu里程计的位姿，如果没有紧挨着激光帧的imu里程计数据，那么就是false；
            这俩应该一般都有
        */
        if (cloudInfo.odomAvailable == true)                    
        {
            /* cloudInfo来自featureExtraction.cpp发布的lio_sam/feature/cloud_info,
                而其中的initialGuessX等信息本质上来源于ImageProjection.cpp发布的deskew/cloud_info信息，
                而deskew/cloud_info中的initialGuessX则来源于ImageProjection.cpp中的回调函数odometryHandler，
                odometryHandler订阅的是imuPreintegration.cpp发布的odometry/imu_incremental话题，
                该话题发布的xyz是imu在前一帧雷达基础上的增量位姿
                纠正一个观点：增量位姿，指的绝不是预积分位姿！！是在前一帧雷达的基础上(包括该基础!!)的（基础不是0）的位姿
            */
           //当前帧的初始估计位姿（来自imu里程计），后面用来计算增量位姿变换
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);                                                 
            if (lastImuPreTransAvailable == false)                                    
            {
                // 赋值给前一帧
                //lastImuPreTransAvailable是一个静态变量，初始被设置为false,之后就变成了true
                //也就是说这段只调用一次，就是初始时，把imu位姿赋值给lastImuPreTransformation
                lastImuPreTransformation = transBack;       
                lastImuPreTransAvailable = true;                    
            } else {
                /* 当前帧相对于前一帧的位姿变换，imu里程计计算得到
                   lastImuPreTransformation就是上一帧激光时刻的imu位姿,transBack是这一帧时刻的imu位姿
                   求完逆相乘以后才是增量，绝不可把imu_incremental发布的当成是两激光间的增量
                */
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;       
                // 前一帧的位姿  
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);    
                // 当前帧的位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;   
                //将transFinal传入，结果输出至transformTobeMapped中
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                // 当前帧初始位姿赋值作为前一帧
                lastImuPreTransformation = transBack;    
                // 虽然有里程记信息，仍然需要把imu磁力计得到的旋转记录下来
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        //只在第一帧调用（注意上面的return），用imu数据初始化当前帧imu里程计位姿，仅初始化旋转部分
        if (cloudInfo.imuAvailable == true)
        {
            /*注：这一时刻的transBack和之前if (cloudInfo.odomAvailable == true)内部的transBack不同，
                之前获得的是initialGuessRoll等，但是在这里是imuRollInit，它来源于imageProjection中的imuQueue，直接存储原始imu数据的。
                那么对于第一帧数据，目前的lastImuTransformation是initialGuessX等，即imu里程计的数据；
                而transBack是imuRollInit是imu的瞬时原始数据roll、pitch和yaw三个角。
                那么imuRollInit和initialGuessRoll这两者有啥区别呢？
                imuRollInit是imu姿态角，在imageProjection中一收到，就马上赋值给它要发布的cloud_info，
                而initialGuessRoll是imu里程计发布的姿态角。
                直观上来说，imu原始数据收到速度是应该快于imu里程计的数据的，因此感觉二者之间应该有一个增量，
                那么lastImuTransformation.inverse() * transBack算出增量，增量再和原先的transformTobeMapped计算,
                结果依旧以transformTobeMapped来保存
            */
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        //智能指针变量创建
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        //容器变量创建
        std::vector<int> pointSearchInd;     // 保存kdtree提取出来的元素的索引（默认排列距离从小到大的索引）
        std::vector<float> pointSearchSqDis; // 保存距离查询位置的距离的数组

        // extract all the nearby key poses and downsample them
        // kdtree的输入，全局关键帧位姿集合（历史所有关键帧集合）
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        /*创建Kd树然后搜索  半径在配置文件中  (surroundingKeyframeSearchRadius: 50.0米)
            指定半径范围查找近邻
            球状固定距离半径近邻搜索
            surroundingKeyframeSearchRadius是搜索半径，pointSearchInd应该是返回的index，pointSearchSqDis应该是依次距离中心点的距离
        */
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            //保存附近关键帧,加入相邻关键帧位姿集合中
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
        //降采样
        //把相邻关键帧位姿集合，进行下采样，滤波后存入surroundingKeyPosesDS
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            //k近邻搜索,找出最近的k个节点（这里是1）
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            /*把强度替换掉，注意是从原始关键帧数据中替换的,相当于是把下采样以后的点的强度，换成是原始点强度
                注意，这里的intensity应该不是强度，因为在函数saveKeyFramesAndFactor中:
                thisPose3D.intensity = cloudKeyPoses3D->size();
                就是索引，只不过这里借用intensity结构来存放
            */
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        
        }

        // also extract some latest key frames in case the robot rotates in one position
        //提取了一些最新的关键帧，以防机器人在一个位置原地旋转
        int numPoses = cloudKeyPoses3D->size();
        // 把10s内的关键帧也加到surroundingKeyPosesDS中,注意是“也”，原先已经装了下采样的位姿(位置)
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)        //timeLaserInfoCur是当前激光帧的时间
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }
        //对降采样后的点云进行提取出边缘点和平面点对应的localmap
        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        // 分别存储角点和面点相关的局部地图
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 遍历当前帧（实际是取最近的一个关键帧来找它相邻的关键帧集合）时空维度上相邻的关键帧集合
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {   
            // 距离超过阈值，丢弃
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)      //  (surroundingKeyframeSearchRadius: 50.0米)
                continue;
            // 相邻关键帧索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            //* 如果这个关键帧对应的点云信息已经存储在一个地图容器里
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())  //laserCloudMapContainer是一个map容器  参考文章链接：https://blog.csdn.net/Augenstern_QXL/article/details/117253848#t21
            {
                // transformed cloud available
                // 直接从容器中取出来加到局部地图中
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                // *如果这个点云没有实现存取，那就通过该帧对应的位姿，把该帧点云从当前帧的位姿转到世界坐标系下
                // 相邻关键帧对应的角点、平面点云，通过6D位姿变换到世界坐标系下
                //transformPointCloud输入的两个形参，分别为点云和变换，返回变换位姿后的点
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
               // 点云转换之后加到局部map中
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
               // 把转换后的面点和角点存进这个容器中，方便后续直接加入点云地图，避免点云转换的操作，节约时间
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        // 降采样局部角点map
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        // 降采样局部平面点map
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // 太大了，清空一下内存
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    //提取局部角点、平面点云集合，加入局部map（提取当前帧相关的关键帧并且构建点云局部地图）
    void extractSurroundingKeyFrames()
    {
        // 如果当前没有关键帧，就return了
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        //对当前帧点云降采样  刚刚完成了周围关键帧的降采样  
        //大量的降采样工作无非是为了使点云稀疏化 加快匹配以及实时性要求
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    /* 当前激光帧角点寻找局部map匹配点
      1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
      2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
    */
    void cornerOptimization()
    {
        //实现transformTobeMapped的矩阵形式转换 下面调用的函数就一行就不展开了  工具类函数
        //  把结果存入transPointAssociateToMap中
        updatePointAssociateToMap();
        // 使用openmp并行加速
        #pragma omp parallel for num_threads(numberOfCores)      //使用多线程对for循环处理点云加速  numberOfCores:4
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)           // 遍历当前帧的角点
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            //第i帧的点转换到第一帧坐标系下
            //这里就调用了第一步中updatePointAssociateToMap中实现的transPointAssociateToMap，
            //然后利用这个变量，把pointOri的点转换到pointSel下，pointSel作为输出
            pointAssociateToMap(&pointOri, &pointSel);          // 将该点从当前帧通过初始的位姿转换到地图坐标系下去
            //kd树的最近搜索
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);    // 在角点地图里寻找距离当前点比较近的5个点
            
            //!涉及公式推导的一些代码
            // 参考文章链接1：https://wykxwyc.github.io/2019/08/01/The-Math-Formula-in-LeGO-LOAM/
            // 参考文章链接2：https://blog.csdn.net/weixin_37835423/article/details/111587379
            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {      // 计算找到的点中距离当前点最远的点，如果距离太大那说明这个约束不可信，就跳过  
                float cx = 0, cy = 0, cz = 0;
                // 先求5个样本的平均值
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;
                // 下面求矩阵matA1=(1/5) *[ax,ay,az]^t*[ax,ay,az]
                // 更准确地说应该是在求协方差matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
                // 求正交阵的特征值和特征向量
                // 特征值：matD1，特征向量：matV1中  对应于LOAM论文里雷达建图 特征值与特征向量那块
                cv::eigen(matA1, matD1, matV1);
                // 边缘：与较大特征值相对应的特征向量，代表边缘线的方向（一大两小，大方向）
                // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
                // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
                // 如果最大的特征值相比次大特征值，大很多，认为构成了线，角点是合格的

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {  //最大特征值与次大特征值比较
                    // 当前帧角点坐标（map系下）
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 局部map对应中心角点，沿着特征向量（直线方向）方向，前后各取一个点
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                     
                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    // 因为[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                    // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
 
                    // a012，也就是三个点组成的三角形面积*2，叉积的模|axb|=a*b*sin(theta)
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    // line_12，底边边长
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 两次叉积，得到点到直线的垂线段单位向量，x分量，下面同理
                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // 得到底边上的高的方向向量[la,lb,lc]
                    // LLL=[la,lb,lc]是V1[0]这条高上的单位法向量。||LLL||=1；
 
                    //如不理解则看图：
                    //        A
                    //   B        C
                    // 这里ABxAC，代表垂直于ABC面的法向量，其模长为平行四边形面积
                    //因此BCx(ABxAC),代表了BC和（ABC平面的法向量）的叉乘，那么其实这个向量就是A到BC的垂线的方向向量
                    //那么(ABxAC)/|ABxAC|,代表着ABC平面的单位法向量
                    //BCxABC平面单位法向量，即为一个长度为|BC|的（A到BC垂线的方向向量），因此再除以|BC|，得到A到BC垂线的单位方向向量
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                    
                    // 三角形的高，也就是点到直线距离
                    // 计算点pointSel到直线的距离
                    // 这里需要特别说明的是ld2代表的是点pointSel到过点[cx,cy,cz]的方向向量直线的距离
                    float ld2 = a012 / l12;
                    // 下面涉及到一个鲁棒核函数，作者简单地设计了这个核函数。
                    //*距离越大，s越小，是个距离惩罚因子（权重）
                    float s = 1 - 0.9 * fabs(ld2);          //fabs()函数用于对float或者double或者int取绝对值
                    // coeff代表系数的意思
                    // coeff用于保存距离的方向向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                    // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                    coeff.intensity = s * ld2;
                    // 程序末尾根据s的值来判断是否将点云点放入点云集合laserCloudOri以及coeffSel中。
                    // 所以就应该认为这个点是边缘点
                    // s>0.1 也就是要求点到直线的距离ld2要小于1m
                    // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }
    
    // * 当前激光帧平面点寻找局部map匹配点
    void surfOptimization()
    {
        //实现transformTobeMapped的矩阵形式转换 下面调用的函数就一行就不展开了  工具类函数
        //  把结果存入transPointAssociateToMap中
        updatePointAssociateToMap();
        
        // 遍历当前帧平面点集合
        #pragma omp parallel for num_threads(numberOfCores)  //numberOfCores: 4    
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            // 平面点（坐标还是lidar系）
            pointOri = laserCloudSurfLastDS->points[i];
            // 根据当前帧位姿，变换到世界坐标系（map系）下
            pointAssociateToMap(&pointOri, &pointSel); 
            // 寻找5个紧邻点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();      //初始化为0
            matB0.fill(-1);            //全部初始化为-1
            matX0.setZero();      //初始化为0
            // 只考虑附近1.0m内
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // 求maxA0中点构成的平面法向量
                //matB0是-1，这个函数用来求解AX=B的X，
                //也就是AX+BY+CZ+1=0
                matX0 = matA0.colPivHouseholderQr().solve(matB0);       //QR分解
                // 假设平面方程为ax+by+cz+1=0，这里就是求方程的系数abc，d=1
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;
                // 单位法向量
                // 对[pa,pb,pc,pd]进行单位化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);       //计算法向量的模长
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 检查平面是否合格，如果5个点中有点到平面的距离超过0.2m，那么认为这些点太分散了，不构成平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    // 当前激光帧点到平面距离
                    //点(x0,y0,z0)到了平面Ax+By+Cz+D=0的距离为：d=|Ax0+By0+Cz0+D|/√(A^2+B^2+C^2) =|Ax0+By0+Cz0+D|       (note: A^2+B^2+C^2 = 1,因为向量已经单位化了)
                    //但是会发现下面的分母开了两次方，不知道为什么，分母多开一次方会更小，这因此求出的距离会更大
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    //* 距离越大，s越小，是个距离惩罚因子（权重）
                    //todo 分母看不懂 sqrt(sqrt(pointOri.x * pointOri.x+ pointOri.y * pointOri.y + pointOri.z * pointOri.z))
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));      
                    // 保存点到平面垂线单位法向量（其实等价于平面法向量）
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;
                    // 当前激光帧平面点，加入匹配集合中
                    //如果s>0.1,代表fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x+ pointOri.y * pointOri.y + pointOri.z * pointOri.z))这一项<1,即"伪距离"<1
                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    //*提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        // 遍历当前帧角点集合，提取出与局部map匹配上了的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        // 遍历当前帧平面点集合，提取出与局部map匹配上了的平面点
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        // 清空标记
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);       //将容器中内容全部用0填充
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    //!scan-to-map优化
    //*对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
    bool LMOptimization(int iterCount)
    {
        //*原始的loam代码是将lidar坐标系转到相机坐标系，这里把原先loam中的代码拷贝了过来，但是为了坐标系的统一，就先转到相机系优化，然后结果转回lidar系
        //*由于LOAM里雷达的特殊坐标系 所以这里也转了一次
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---   camera <- lidar
        // x = z                            ---     x = y
        // y = x                            ---     y = z
        // z = y                            ---     z = x
        // roll = yaw                 ---     roll = pitch
        // pitch = roll               ---     pitch = yaw
        // yaw = pitch              ---     yaw = roll

        // lidar -> camera
        //*roll = pitch
        float srx = sin(transformTobeMapped[1]);        //transformTobeMapped[6]={roll,pitch,yaw,x,y,z}
        float crx = cos(transformTobeMapped[1]);
        //*pitch = yaw
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        //*yaw = roll
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // 当前帧匹配特征点数太少
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        //初始化矩阵，用0填充
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;
        // 遍历匹配特征点，构建Jacobian矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            // 首先将当前点以及点到线（面）的单位向量转到相机系
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;            //coeffSel中放的值和向量方向有关
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
           
            // *in camera
            //! 相机系下的旋转顺序是Y - X - Z对应lidar系下Z -Y -X (note:将雷达系中点变换到全局地图坐标系的变换矩阵T中的旋转矩阵R = ZYX，当转到相机坐标系后，旋转矩阵R' = YXZ)
            //https://wykxwyc.github.io/2019/08/01/The-Math-Formula-in-LeGO-LOAM/
            // 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
            //各种cos sin的是旋转矩阵对roll求导，pointOri.x是点的坐标，coeff.x等是距离到局部点的偏导，也就是法向量（建议看链接）
            //注意：链接当中的R0-5公式中，ex和ey是反的
            //另一个链接https://blog.csdn.net/weixin_37835423/article/details/111587379#commentBox当中写的更好
            //!雅可比矩阵的推导（旋转部分）
            //*求解的是对roll的偏导量
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;
            //*同上，求解的是对pitch的偏导量
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
            //*同上，求解的是对yaw的偏导量
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            // camera -> lidar
            //matA就是误差对旋转和平移变量的雅克比矩阵
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            //对平移求误差就是法向量，见链接
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            // 残差项
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        //*构造JTJ以及-JTe矩阵
        // 将矩阵由matA转置生成matAt
        // 先进行计算，以便于后边调用 cv::solve求解
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        //*利用高斯牛顿法进行求解，
        // 高斯牛顿法的原型是J^(T)*J * delta(x) = -J^(T)*f(x)
        // J是雅克比矩阵，这里是A，f(x)是优化目标，这里是-B(符号在给B赋值时候就放进去了)
        // 通过QR分解的方式，求解matAtA*matX=matAtB，得到解matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);            //参数1：左侧矩阵A      参数2：右侧矩阵B     参数3：待求变量X   参数4：分解方式
    
        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0) {
            //创建矩阵并初始化为0
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            // 对近似的Hessian矩阵求特征值和特征向量， （Hessian矩阵：多元函数的二阶偏导数构成的矩阵）
            //matE特征值,matV是特征向量
            //*退化方向只与原始的约束方向  A有关，与原始约束的位置 b 无关     (线性化系统的结论)
            //算这个的目的是要判断退化，即约束中较小的偏移会导致解所在的局部区域发生较大的变化
            cv::eigen(matAtA, matE, matV);               //参数1：输入的矩阵(要求是转置矩阵)   参数2：特征值(结果为从大到小)     参数3：特征向量(按行排列)
            matV.copyTo(matV2);                              

            isDegenerate = false;
            //初次优化时，特征值门限设置为100，小于这个值认为是退化了
            //系统退化与否和系统是否存在解没有必然联系，即使系统出现退化，系统也是可能存在解的，
            //*因此需要将系统的解进行调整，一个策略就是将解进行投影，
            //对于退化方向，使用优化的状态估计值，对于非退化方向，依然使用方程的解。
            //*另一个策略就是直接抛弃解在退化方向的分量。
            //对于退化方向，我们不考虑，直接丢弃，只考虑非退化方向解的增量。
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {          //特征值小于阈值
                    for (int j = 0; j < 6; j++) {                                
                        matV2.at<float>(i, j) = 0;                              //特征向量全置为0
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            //以下这步可以参考链接：
            //https://zhuanlan.zhihu.com/p/258159552
            matP = matV.inv() * matV2;                      //非线性求解（只使用非线性优化解在非退化方向的分量，不考虑退化方向的分量）
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);                     //matX为高斯牛顿迭代法算出的增量
            matX = matP * matX2;                    //Vf.inv()*Vu*deltaX
        }
        //Xf = Xf + Vf.inv()*Vu*deltaX  (获取更可靠的位姿)
        transformTobeMapped[0] += matX.at<float>(0, 0);                  //roll
        transformTobeMapped[1] += matX.at<float>(1, 0);                  //pitch
        transformTobeMapped[2] += matX.at<float>(2, 0);                  //yaw
        transformTobeMapped[3] += matX.at<float>(3, 0);                  //x
        transformTobeMapped[4] += matX.at<float>(4, 0);                  //y
        transformTobeMapped[5] += matX.at<float>(5, 0);                  //z 
        // 计算更新的旋转和平移大小
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +           //弧度转成度
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
        // 旋转或者平移量足够小就停止这次迭代过程
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        // 否则继续优化
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        /*根据现有地图与最新点云数据进行配准从而更新机器人精确位姿与融合建图，
            它分为角点优化、平面点优化、配准与更新等部分。
            优化的过程与里程计的计算类似，是通过计算点到直线或平面的距离，构建优化公式再用LM法求解
        */
        // 如果没有关键帧，那也没办法做当前帧到局部地图的匹配
        if (cloudKeyPoses3D->points.empty())
            return;
        // 判断当前帧的角点数和面点数是否足够
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)  //edgeFeatureMinValidNum: 10，surfFeatureMinValidNum: 100
        {  
           // 分别把角点面点局部地图构建kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
            //迭代30次
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();
                //边缘点匹配优化
                cornerOptimization();
                //平面点匹配优化
                surfOptimization();
                //组合优化多项式系数
                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
            }
            //使用了9轴imu的orientation与transformTobeMapped做插值，并且roll和pitch收到常量阈值约束（权重）
            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }
    
    //用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    void transformUpdate()
    {
        // 可以获取九轴imu的世界系下的姿态
        if (cloudInfo.imuAvailable == true)
        {
            // 俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;               //  imuRPYWeight: 0.01
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);           // lidar匹配获得的roll角转成四元数
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);                  // imu获得的roll角
                //*slerp()四元数的球面线性插值，当两个四元数夹角很小时，近似于线性插值 slerp(p,q,t) = (1-t)p+tq    (t为比例系数)   imuWeight = 0.01
                // 参考文章链接：https://zhuanlan.zhihu.com/p/538653027
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);           
                transformTobeMapped[0] = rollMid;        // 插值结果作为roll的最终结果

                // slerp pitch
                // pitch角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }
        // 更新当前帧位姿的roll, pitch, z坐标；因为是小车，roll、pitch是相对稳定的，
        // 不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);        //rotation_tollerance: 1000 rad
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);       
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);                        //z_tollerance: 1000    m
        // 当前帧位姿
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    // 当前帧是否为关键帧
    bool saveFrame()
    {
        // 如果没有关键帧，那直接认为是关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;
        //*代码用不到
        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }
        // 前一帧位姿
        //注：最开始没有的时候，在函数extractCloud里面有
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 位姿变换增量
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 旋转和平移量都较小，当前帧不设为关键帧
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&                            //surroundingkeyframeAddingAngleThreshold: 0.2   rad
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)                  //surroundingkeyframeAddingDistThreshold: 1.0  m
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            // 第一帧初始化先验因子(置信度就设置差一点，尤其是不可观的平移和yaw角)
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 增加先验约束，对第0个节点增加约束
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 变量节点设置初始值
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            // *添加激光里程计因子
            // 如果不是第一帧，就增加帧间约束
            // 这时帧间约束置信度就设置高一些
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 变量节点设置初始值
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        // 如果没有gps信息就算了
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // 位姿协方差很小，没必要加入GPS数据进行校正
        //?3和4我猜可能是x和y？（6维，roll，pitch，yaw，x，y，z）
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)           //poseCovThreshold: 25.0
            return;

        // last gps position
        static PointType lastGPSPoint;
        while (!gpsQueue.empty())
        {
            // 删除当前帧0.2s之前的里程计
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            // 超过当前帧0.2s之后，退出
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                // GPS噪声协方差太大，不能用
                float noise_x = thisGPS.pose.covariance[0];           //covariance为float64[36]型数组（数据为6*6的协防差矩阵）（x,y,z,roll,pitch,yaw)
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)    //gpsCovThreshold: 2.0 
                    continue;

                // GPS里程计位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)                                                           //useGpsElevation: false     
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                // gps的x或者y太小说明还没有初始化好
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                // 每隔5m添加一个GPS里程计
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;
                // 添加GPS因子
                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                // 加入gps之后等同于回环，需要触发较多的isam update
                aLoopIsClosed = true;
                break;
            }
        }
    }

    //*添加闭环因子
    void addLoopFactor()
    {
        // 有一个专门的回环检测线程会检测回环，检测到就会给这个队列塞入回环结果
        if (loopIndexQueue.empty())
            return;
       // 把队列里所有的回环约束都添加进来
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            // 闭环边对应两帧的索引
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 闭环边的位姿变换
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // 回环的置信度就是icp的得分
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }
        // 清空回环相关队列
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        // 标志位置true
        aLoopIsClosed = true;
    }

    //*设置当前帧为关键帧并执行因子图优化
    void saveKeyFramesAndFactor()
    {  
        // 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        if (saveFrame() == false)      //为true继续向下执行
            return;
        // 如果是关键帧就给isam增加因子
        // odom factor
        addOdomFactor();

        // gps factor
        //!没有可以考虑注释掉
        addGPSFactor();       

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        // 执行优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        // 如果加入了gps的约束或者回环约束，isam需要进行更多次的优化
        if (aLoopIsClosed == true)            //初始aLoopIsClosed = false; 添加GPS因子和回环检测因子函数中将aLooplsClosed设置为true
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        // 下面保存关键帧信息
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        // 优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        // 当前帧位姿结果
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");
        
        // cloudKeyPoses3D加入当前帧位姿( 平移信息取出来保存进cloudKeyPoses3D这个结构中，其中索引作为intensity值)
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        //索引
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);
        
        // cloudKeyPoses6D加入当前帧位姿
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;               //当前激光帧的时间
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 位姿协方差(保存当前位姿的置信度)
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        //  transformTobeMapped更新当前帧位姿( 将优化后的位姿更新到transformTobeMapped数组中，作为当前最佳估计值)
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        // 当前帧激光角点、平面点，降采样集合
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // 保存特征点降采样集合( 关键帧的点云保存下来)
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // 更新里程计轨迹( 根据当前最新位姿更新rviz可视化)
        updatePath(thisPose6D);
    }

    //*更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
    void correctPoses()
    {
        // 没有关键帧，自然也没有什么意义
        if (cloudKeyPoses3D->points.empty())
            return;
        // 只有回环以及gps信息这些会触发全局调整信息才会触发
        if (aLoopIsClosed == true)
        {
            // clear map cache
            // 清空局部map(很多位姿会变化，因子之前的容器内转到世界坐标系下的很多点云就需要调整，因此这里索性清空)
            laserCloudMapContainer.clear();
            // clear path
            // 清空里程计轨迹
            globalPath.poses.clear();
            // update key poses
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 更新所有关键帧的位姿
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                // 同时更新path
                updatePath(cloudKeyPoses6D->points[i]);
            }
            // 标志位置位
            aLoopIsClosed = false;
        }
    }

    //*更新里程计轨迹
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    //*发布激光里程计
    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        // 发布激光里程计，odom等价map（ 发送当前帧的位姿）
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;       //当前激光帧时间戳
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        // 发布TF，odom->lidar（发送lidar在odom坐标系下的tf）
        static tf::TransformBroadcaster br;      //创建 TF 广播器
        //获取变换信息
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        // tf::StampedTransform是tf::Transform的一个特例      StampedTransform(参数1：输入，参数2：时间戳，参数3：父坐标系，参数4：子坐标系)
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);      //广播器发布数据

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        //第一次数据直接用全局里程计初始化
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            // 当前帧与前一帧之间的位姿变换
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            // 位姿增量叠加到上一帧位姿上
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    // roll姿态角加权平均
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    // pitch姿态角加权平均
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        // 发布历史关键帧位姿集合
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        // 发送周围局部地图点云信息
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        // 发布当前帧的角点、平面点降采样集合
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 把当前点云转换到世界坐标系下去
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            // 发送当前点云
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            // 发送原始点云
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        // 发送path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                lio_sam::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                // 发布地图点云信息
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};

//!主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");   //节点初始化

    mapOptimization MO;       //创建类对象

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");    //输出日志信息
    //把某个类中的某个函数作为线程的入口地址   （参考文章链接：https://blog.csdn.net/qq_38231713/article/details/106091372）
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);                                              //开一个回环检测线程 （参数1：取地址，参数2：引用，当然也可以直接传值）
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);             //开一个可视化地图线程

    ros::spin();                    //循环调用

    loopthread.join();             
    visualizeMapThread.join();

    return 0;
}
