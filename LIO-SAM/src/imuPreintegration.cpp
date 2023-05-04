#include "utility.h"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)


// 订阅激光里程计（来自MapOptimization）和IMU里程计，
//根据前一时刻激光里程计，和该时刻到当前时刻的IMU里程计变换增量，
//计算当前时刻IMU里程计；
//rviz展示IMU里程计轨迹（局部）
class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;
    //*创建订阅对象和发布对象
    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;           //创建仿射变换对象
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;              //定义监听器tfListenner
    tf::StampedTransform lidar2Baselink;  //定义存放变换关系的变量lidar2Baselink

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;           //创建双端队列imuOdomQueue

    TransformFusion()    //类的构造函数
    {
        // 如果lidar系与baselink系不同（激光系和载体系），需要外部提供二者之间的变换关系(通常baselink指车体系）
        if(lidarFrame != baselinkFrame)
        {
            /*
                由于tf的会把监听的内容存放到一个缓存中，然后再读取相关的内容，而这个过程可能会有几毫秒的延迟，也就是，
                tf的监听器并不能监听到“现在”的变换，所以如果不使用try,catch函数会导致报错
            */     
            try
            {
                // 等待3s
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                // baselink系到lidar系的变换
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)               
            {
                ROS_ERROR("%s",ex.what());
            }
        }
        //!订阅IMU里程计+激光里程计，发布IMU里程计+IMU路径
        // 订阅地图优化节点的全局位姿和预积分节点的增量位姿
        // 订阅激光里程计，来自mapOptimization
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅imu里程计，来自IMUPreintegration(IMUPreintegration.cpp中的类IMUPreintegration)
        //topic name: odometry/imu_incremental
        //注意区分lio_sam/mapping/odometry_incremental
        //目前可以明确的一点，odometry/imu_incremental是增量内容，即两帧激光里程计之间的预积分内容,（加上开始的激光里程计本身有的位姿）
        //imuIntegratorImu_本身是个积分器，只有两帧之间的预积分，但是发布的时候实际是结合了前述里程计本身有的位姿
        //如这个predict里的prevStateOdom:
        //currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom)
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());
        // 发布imu里程计，用于rviz展示
        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        // 发布imu里程计轨迹
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }

    //里程计对应变换矩阵
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;          //创建tf的四元数变量
        //ROS消息的四元数转换成tf的四元数，然后再转换成tf的欧拉角  （参考文章链接：https://blog.csdn.net/haigujiujian/article/details/117878718）
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation); 
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    //*订阅激光里程计的回调函数，来自mapOptimization
    // 将全局位姿保存下来
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 激光里程计对应变换矩阵
        lidarOdomAffine = odom2affine(*odomMsg);
        // 激光里程计时间戳
        lidarOdomTime = odomMsg->header.stamp.toSec();
        //这二者里面保存的都是最近的一个雷达激光里程计的变换和时间戳（不再是用一个vector之类的东西保存起来）
    }

    /*
     * 订阅imu里程计，来自IMUPreintegration
     * 1、以最近一帧激光里程计位姿为基础，计算该时刻与当前时刻间imu里程计增量位姿变换，相乘得到当前时刻imu里程计位姿
     * 2、发布当前时刻里程计位姿，用于rviz展示；发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
    */

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf(发布tf，map与odom系设为同一个系)
        static tf::TransformBroadcaster tfMap2Odom;  //创建tf的广播器tfMap2Odom
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));// 定义存放转换信息（平移，旋转）的变量map_to_odom，初始化tf数据
        // 发送静态tf，odom系和map系将他们重合(广播map坐标系与odometry坐标系之间的tf数据)
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);
        // imu得到的里程记结果送入这个队列中( 添加imu里程计到队列，注：imu里程计由本cpp中的另一个类imuPreintegration来发布)
        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        // 如果没有收到lidar位姿就return( lidarOdomTime初始化为-1，在收到lidar里程计数据后，在回调函数lidarOdometryHandler中被赋值时间戳)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            // 弹出时间戳小于最新lidar位姿时刻之前的imu里程记数据(  从imu里程计队列中删除当前（最近的一帧）激光里程计时刻之前的数据)
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        // 计算最新队列里imu里程记的增量(最近的一帧激光里程计时刻对应imu里程计位姿)
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        // 当前时刻imu里程计位姿
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        // imu里程计增量位姿变换(当前时刻imu里程计位姿=最近的一帧激光里程计位姿 * imu里程计增量位姿变换)
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        // 增量补偿到lidar的位姿上去，就得到了最新的预测的位姿(lidarOdomAffine在本类的lidarOdometryHandler回调函数中被赋值，消息来源于mapOptimization.cpp发布,是激光里程计)
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        // 分解成平移+欧拉角的形式
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        // 发送全局一致位姿的最新位姿
        /*
            发布名为"odometry/imu"的话题
            也就是说，这个函数先监听了类IMUPreintegration中发布的odometry/imu_incremental，
            然后计算imu二者之间的增量，然后在激光的基础上加上增量，重新发布
            可以看出发布的并非odometry/imu_incremental的一部分子数据，而是把x，y，z，roll，pitch，yaw都经过了激光里程计的修正，才发布
            可以看出这次发布的内容，是当前时刻里程计位姿.发布名称为"odometry/imu"
        */
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // 更新tf(发布tf，当前时刻odom与baselink系变换关系)
        //由于之前把map和odom坐标系固定了，因此这里我认为发布的就是真正的最终位姿关系
        static tf::TransformBroadcaster tfOdom2BaseLink;//创建tf的广播器tfOdom2BaseLink
        tf::Transform tCur;       // 定义存放转换信息（平移，旋转）的变量tcur（note:odom系相对lidar系的位姿）
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);    //ros下的位姿信息转换为TF位姿信息
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;      //map优化提供激光，预积分提供imu，imu之间变换再乘以激光里程计得到各个时刻精确位姿
        // 更新odom到baselink的tf(广播odometry坐标系与baselink坐标系之间的tf数据)
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发送imu里程记的轨迹(发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段)
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // 控制一下更新频率，不超过10hz
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            // 将最新的位姿送入轨迹中
            imuPath.poses.push_back(pose_stamped);
            // 把lidar时间戳之前的轨迹全部擦除(  删除最近一帧激光里程计时刻之前的imu里程计)
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            // 发布轨迹，这个轨迹实际上是可视化imu预积分节点输出的预测值
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

//todo IMUPreintegration类
class IMUPreintegration : public ParamServer             //一个类，继承paramServer，公共继承(类继承链接：https://blog.csdn.net/Augenstern_QXL/article/details/117253730?spm=1001.2014.3001.5501)
{
public:
    //*变量声明
    std::mutex mtx;                      //定义mtx互斥量

    //订阅和发布变量定义
    ros::Subscriber subImu;          
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;

    bool systemInitialized = false;          //系统初始化为false
    //噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;  // Diagonal对角线  先验位姿噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;     // 先验速度噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;   // 先验偏差噪声
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;    //修正噪声
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;  //修正噪声2
    gtsam::Vector noiseModelBetweenBias;      

    //*预积分
    // 这个和优化有关(负责预积分两个激光里程计之间的imu数据，作为约束加入因子图，并且优化出bias)
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;  
    // 这个和imu预测有关( 用来根据新的激光里程计到达后已经优化好的bias，预测从当前帧开始，下一帧激光里程计到达之前的imu里程计增量)
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;  

    // 下面这两个双端队列，用来存储上面的两个指针（或者叫数据来源）
    // 给imuIntegratorOpt_提供数据来源，不要的就弹出（从队头开始出发，比当前激光里程计数据早的imu通通积分，用一个仍一个）
    std::deque<sensor_msgs::Imu> imuQueOpt;
    // 给imuIntegratorImu_提供数据来源，不要的就弹出（弹出当前激光里程计之前的imu数据，预计分用完一个弹一个）
    std::deque<sensor_msgs::Imu> imuQueImu;

    // imu因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    // imu状态
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;     // 这个应该是判断是否是第一次来优化的（第一次优化为false)        
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    // iSAM2优化器
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;     //总的因子图模型
    gtsam::Values graphValues;   //因子图模型中的值

    const double delta_t = 0;

    int key = 1;

    // *imu-lidar位姿变换
    //头文件的imuConverter中，也只有一个旋转变换。这里绝对不可以理解为把imu数据转到lidar下的变换矩阵。
    //事实上，作者后续是把imu数据先用imuConverter旋转到雷达系下（但其实还差了个平移）。
    //作者真正是把雷达数据又根据lidar2Imu反向平移了一下，和转换以后差了个平移的imu数据在“中间系”对齐，
    //之后算完又从中间系通过imu2Lidar挪回了雷达系进行publish
    //https://blog.csdn.net/zkk9527/article/details/117957067
    
    // T_b'_l: tramsform points from lidar frame to imu frame 
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));  //从雷达系移动到中间系
    // T_l_b': tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));  //从中间系移动到雷达系(可以联系t12理解)
    
    IMUPreintegration()  
    {
        //!订阅imu+odometry和发布imuodometry
        //订阅imu原始数据，用下面因子图优化的结果，施加两帧之间的imu预计分量，预测每一时刻（imu频率）的imu里程计
        subImu  = nh.subscribe<sensor_msgs::Imu>  (imuTopic,   2000, &IMUPreintegration::imuHandler,  this, ros::TransportHints().tcpNoDelay()); // 最后这个参数是ros里面信息的传输方式，这里用的是tcp传输，而且是无延迟的
        //订阅激光里程计，来自mapOptimization，用两帧之间的imu预积分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的imu里程计，以及下一次因子图优化）
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        //发布imu里程计: odometry/imu_incremental
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);

        //*imu预积分的噪声协方差
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        //imuAccNoise和imuGyrNoise都是定义在头文件中的高斯白噪声，由配置文件中写入  
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous    pow(x,y)是求x的y次方
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        // 积分的协方差
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        //假设没有初始的bias
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        //*噪声先验
        //Diagonal对角线矩阵
        //发现diagonal型一般调用.finished(),注释中说finished()意为设置完所有系数后返回构建的矩阵 (博客链接：https://blog.csdn.net/xuezhisdc/article/details/54619853#t4)
        // 初始位姿置信度设置比较高
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        // 初始速度置信度就设置差一些
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        // 零偏的置信度也设置高一些
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good

        // 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        //imu预积分器，用于预测每一时刻（imu频率）的imu里程计（转到lidar系了，与激光里程计同一个系）  
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        //imu预积分器，用于因子图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    //todo重新设置图优化的参数
    void resetOptimization()
    {
        //*gtsam初始化
        //ISAM2的参数类
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1; // relinearize 重新线性化
        optParameters.relinearizeSkip = 1;
        //优化器 根据参数重新生成优化器
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }
    //todo重新设置参数
    void resetParams()
    {
        //将上一帧imu数据的时间戳更新成-1（代表还没有imu数据进来）
        lastImuT_imu = -1;
        //false代表需要重新进行一次odom优化（跟imuHandler联系起来）
        doneFirstOpt = false;
        //系统关闭
        systemInitialized = false;
    }

    // !订阅的是激光里程计,"lio_sam/mapping/odometry_incremental"( 订阅地图优化节点的增量里程记消息)（核心)
    /*
    里程计回调函数
    每隔100帧激光里程计，重置iSAM2优化器，添加里程计、速度、偏置先验因子，执行优化
    计算前一帧激光里程计与当前帧激光里程计之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，添加来自mapOptimization的
    当前位姿，进行因子图优化，更新当前帧状态。
    优化之后，获得imu真实的bias，用来计算当前时刻之后的imu预积分。
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);        //std::lock_guard是锁保护器,保证在抛出异常时正确地解锁互斥对象,避免死锁的情况
        // 当前帧激光里程计时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        // 确保imu队列中有数据
        if (imuQueOpt.empty())                   //empty()：如果 queue 中没有元素的话，返回 true,有元素就是false
            return;
        // 获取里程记位姿( 当前帧激光位姿，来自scan-to-map匹配、因子图优化后的位姿)
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        // 该位姿是否出现退化
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        // 把位姿转成gtsam的格式
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system(第一帧)
        //todo首先初始化系统
        if (systemInitialized == false)
        {
            // 重置ISAM2优化器
            resetOptimization();

            // pop old IMU message
            //从imu优化队列中删除当前帧激光里程计时刻之前的imu数据,delta_t=0
            while (!imuQueOpt.empty())
            {
                // 最终得到了离currentCorrectionTime - delta_t 最近的那个imu时间
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    //将队首元素的时间戳记录下来，更新lastImuT_opt，再将元素弹出
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }

            // 添加里程计位姿先验因子
            // *initial pose                                                  
            // 将lidar的位姿转移到imu坐标系(中间坐标系)下
            prevPose_ = lidarPose.compose(lidar2Imu);      //lidarPose 为本回调函数收到的激光里程计数据，重组成gtsam的pose格式  //todo T_w_b' = T_w_l * T_l_b'
            // 设置其初始位姿和置信度  （ PriorFactor 概念可看gtsam  包括了位姿 速度  bias ，加入PriorFactor在图优化中基本都是必需的前提）
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);   //priorPoseNoise先验位姿噪声
            // 约束加入到因子中
            graphFactors.add(priorPose);         // 添加先验位姿 先验是计算得到的，后验是观测得到的

            // 添加速度先验因子
            //* initial velocity
            // 初始化速度，这里就直接赋0了
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            // 将对速度的约束也加入到因子图中
            graphFactors.add(priorVel);

            // 添加imu偏置先验因子
            //*initial bias
            // 初始化零偏
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            // 零偏加入到因子图中
            graphFactors.add(priorBias);

            // 以上把约束加入完毕，下面开始添加状态量
            //* add values
            // 将各个状态量赋成初始值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            //*optimize once  
            // 约束和状态量更新进isam优化器
            optimizer.update(graphFactors, graphValues);  // 优化一次
            // 进优化器之后保存约束和状态量的变量就清零 （图和节点均清零  为什么要清零不能继续用吗? 是因为节点信息保存在gtsam::ISAM2 optimizer，所以要清理后才能继续使用）
            graphFactors.resize(0);
            graphValues.clear();

            // 重置优化之后的偏置
            // 预积分的接口，使用初始零偏进行初始化
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            //key = 1 表示有一帧
            //系统初始化成功
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        // 当isam优化器中加入了较多的约束后，为了避免运算时间变长，就直接清空(  每隔100帧激光里程计，重置ISAM2优化器，保证优化效率)(//?滑动窗口的实现方式？)
        if (key == 100)
        {
            // get updated noise before reset
            // 取出最新时刻位姿 速度 零偏的协方差矩阵( 前一帧的位姿、速度、偏置噪声模型)
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            // 复位整个优化问题
            resetOptimization();

            // 添加位姿先验因子，用前一帧的值初始化
            // add pose
            // 将最新的位姿，速度，零偏以及对应的协方差矩阵加入到因子图中
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);

            // add velocity  添加速度先验因子，用前一帧的值初始化
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);

            //添加偏置先验因子，用前一帧的值初始化
            // add bias  
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);

            // 变量节点赋初值，用前一帧的值初始化
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            // 优化一次
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        //todo 1. integrate imu data and optimize（第二帧IMU数据)
        // 将两帧之间的imu做积分 (计算前一帧与当前帧之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计)
        //  添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            // 将imu消息取出来( 提取前一帧与当前帧之间的imu数据，计算预积分)
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            // 时间上小于当前lidar位姿的都取出来
            if (imuTime < currentCorrectionTime - delta_t)  //currentCorrectionTime是当前回调函数收到的激光里程计数据的时间,delta_t=0
            {
                // 计算两个imu量之间的时间差
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                // 调用预积分接口将imu数据送进去处理(imu预积分数据输入：加速度、角速度、dt)
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                // 记录当前imu时间( 在推出一次数据前保存上一个数据的时间戳)
                lastImuT_opt = imuTime;
                //从队列中删除已经处理的imu数据
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // 添加imu预积分因子
        //* add imu factor to graph(利用两帧之间的IMU数据完成了预积分后增加imu因子到因子图中)
        // 两帧间imu预积分完成之后，就将其转换成预积分约束
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        // 预积分约束对相邻两帧之间的位姿 速度 零偏形成约束
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu); // 参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧偏置，预积分量
        // 加入因子图中
        graphFactors.add(imu_factor);

        // 添加imu偏置因子
        // add imu bias between factor(添加imu偏置因子，前一帧偏置B(key - 1)，当前帧偏置B(key)，观测值，噪声协方差；deltaTij()是积分段的时间)
        // 零偏的约束，两帧间零偏相差不会太大，因此使用常量约束
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
       

        // 添加位姿因子
        // add pose factor 
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);       //todo T_w_b' = T_w_l * T_l_b'  (转到中间系)
        // lidar位姿补偿到imu坐标系(中间坐标系)下，同时根据是否退化选择不同的置信度，作为这一帧的先验估计
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        // 加入因子图中去
        graphFactors.add(pose_factor);

        // insert predicted values      
        // 根据上一时刻的状态，结合预积分结果，对当前状态进行预测( 用前一帧的状态、偏置，施加imu预计分量，得到当前帧的状态)
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 预测量作为初始值插入因子图中
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        // 执行优化
        optimizer.update(graphFactors, graphValues);
        optimizer.update();  //update两次，优化效果可能会好一点(作者的经验)

        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        // 获取优化后的当前状态作为当前帧的最佳估计
        // 更新当前帧位姿、速度 --> 上一帧 这步就是当前帧的变成了上一帧的
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        // 更新当前帧状态--> 上一帧
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        // 更新当前帧imu偏置
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        // 当前约束任务已经完成，预积分约束复位，同时需要设置一下零偏作为下一次积分的先决条件(重置预积分器，设置新的偏置，这样下一帧激光里程计进来的时候，预积分量就是两帧之间的增量)
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        // 一个简单的失败检测( imu因子图优化结果，速度或者偏置过大，认为失败)
        if (failureDetection(prevVel_, prevBias_))
        {
            // 状态异常就直接复位了
            resetParams();
            return;
        }


        //todo2. after optiization, re-propagate imu odometry preintegration
        // 优化之后，根据最新的imu状态进行传播( 用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿)
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        /*
            注意，这里是要“删除”当前帧“之前”的imu数据，是想根据当前帧“之后”的累积递推。
            而前面imuIntegratorOpt_做的事情是，“提取”当前帧“之前”的imu数据，用两帧之间的imu数据进行积分。处理过的就弹出来。
            因此，新到一帧激光帧里程计数据时，imuQueOpt队列变化如下：
            当前帧之前的数据被提出来做积分，用一个删一个（这样下一帧到达后，队列中就不会有现在这帧之前的数据了）
            那么在更新完以后，imuQueOpt队列不再变化，剩下的原始imu数据用作下一次优化时的数据。
            而imuQueImu队列则是把当前帧之前的imu数据都给直接剔除掉，仅保留当前帧之后的imu数据，
            用作两帧lidar里程计到达时刻之间发布的imu增量式里程计的预测。
            imuQueImu和imuQueOpt的区别要明确,imuIntegratorImu_和imuIntegratorOpt_的区别也要明确,见imuhandler中的注释

        */

        // 首先把lidar帧之前的imu状态全部弹出去
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate(理由：对剩余的imu数据计算预积分，因为bias改变了)
        // 如果有新于lidar状态时刻的imu
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // 这个预积分变量复位( 重置预积分器和最新的偏置，使用最新的偏差更新bias值)
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 然后把剩下的imu状态重新积分(利用imuQueImu中的数据进行预积分 主要区别旧在于上一行的更新了bias)
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);
                // 注意:加入的是这个用于传播的的预积分器imuIntegratorImu_,(之前用来计算的是imuIntegratorOpt_,）
                //注意加入了上一步算出的dt
                //结果被存放在imuIntegratorImu_中
                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }
        //记录帧数
        ++key;
        //优化器开启(设置成True，用来通知另一个负责发布imu里程计的回调函数imuHandler“可以发布了”)
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        // 如果当前速度大于30m/s，108km/h就认为是异常状态
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        // 如果零偏太大，那也不太正常
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    /*
     !订阅imu原始数据
     * 1、用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预计分量，得到当前时刻的状态，也就是imu里程计
     * 2、imu里程计位姿转到lidar系，发布里程计
    */

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 首先把imu的状态做一个简单的转换( imu原始测量数据转换到lidar中间系，加速度、角速度、RPY)
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);
        // 添加当前帧imu数据到队列
        // 注意这里有两个imu的队列，作用不相同，一个用来执行预积分和位姿的优化，一个用来更新最新imu状态(两个双端队列分别装着优化前后的imu数据)
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // 要求上一次imu因子图优化执行成功，确保更新了上一帧（激光里程计帧）的状态、偏置，预积分已经被重新计算
        // 这里需要先在odomhandler中优化一次后再进行该函数后续的工作
        if (doneFirstOpt == false)   
            return;

        double imuTime = ROS_TIME(&thisImu);
        //lastImuT_imu变量初始被赋值为-1
        // 获得时间间隔, 第一次为1/500,之后是两次imuTime间的差
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        // 每来一个imu值就加入预积分状态中( imu预积分器添加一帧imu数据，注：这个预积分器的起始时刻是上一帧激光里程计时刻)
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        // 根据这个值预测最新的状态( 用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预计分量，得到当前时刻的状态)
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        // 发布imu里程计（转到lidar系，与激光里程计同一个系）
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // 将这个状态转到lidar坐标系下去发送出去(预测值currentState获得imu位姿, 再由imu到雷达变换, 获得雷达位姿)
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);        //todo  T_w_l =  T_w_b'  *  T_b'_l            (中间系转到雷达系)

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};

//!主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");  //节点初始化
    
    IMUPreintegration ImuP;         //创建类对象ImuP

    TransformFusion TF;                  //创建类对象TF

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");         //输出日志消息
    
    ros::MultiThreadedSpinner spinner(4);                          //多线程，4个线程
    spinner.spin();                                                  //循环调用
    
    return 0;
}
