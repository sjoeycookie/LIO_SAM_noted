#include "utility.h"
#include "lio_sam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    //*创建订阅对象和发布对象
    ros::Subscriber subLaserCloudInfo; 
    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    //创建pcl点云的智能指针
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;   //创建体素网格对象

    lio_sam::cloud_info cloudInfo;  // 当前激光帧点云信息，包括的历史数据有：运动畸变校正，点云数据，初始位姿，姿态角，有效点云数据，角点点云，平面点点云等
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;         //创建容器

    //创建一些指针变量
    float *cloudCurvature;          //用来做曲率计算的中间变量
    int *cloudNeighborPicked;          //特征提取标记，1表示遮挡、平行，或者已经进行特征提取的点，0表示还未进行特征提取处理
    int *cloudLabel;            //  1表示角点，-1表示平面点

    FeatureExtraction()      //构造函数
    {
        // 订阅当前帧激光运动畸变校正后的点云信息（订阅的话题来自imageProjection类中的发布话题)
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布当前激光帧提取特征之后的点云信息
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        // 发布当前激光帧的角点点云
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        // 发布当前激光帧的面点点云
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        // 初始化
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);     //容器开辟空间大小
        // 进行体素滤波实现降采样，叶子大小为0.4*0.4*0.4
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);      //odometrySurfLeafSize: 0.4  单位m
        
        //使用智能指针的reset进行初始化
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        //开辟堆区
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    //!回调函数(核心)
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        // 当前激光帧运动畸变校正后的有效点云
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction          //将msg信息转成点云形式

        //*计算当前激光帧点云中每个点的曲率
        calculateSmoothness();
        //* 标记属于遮挡、平行两种情况的点，不做特征提取
        markOccludedPoints();

        /*点云角点、平面点特征提取
            1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
            2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
        */
        extractFeatures();
        //发布角点、面点点云，发布带特征点云数据的当前激光帧点云信息
        publishFeatureCloud();
    }

    // 计算曲率
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();  // 遍历当前激光帧运动畸变校正后的有效点云
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 计算当前点和周围十个点的距离差
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            
            // 距离差值平方作为曲率
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;     //0表示还未进行特征提取处理,1表示遮挡、平行，或者已经进行特征提取的点
            cloudLabel[i] = 0;   //1表示角点，-1表示平面点
            // cloudSmoothness for sorting
            // 存储该点曲率值、激光点一维索引
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    // 标记一下遮挡的点
    void markOccludedPoints()       
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            // 取出相邻两个点距离信息
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
             /*
                两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；
                如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
                如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
            */
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));     // 计算两个有效点之间的列id差
            // 只有比较靠近才有意义
            if (columnDiff < 10){
                // 10 pixel diff in range image
                /*
                    两个点在同一扫描线上，且距离相差大于0.3，认为存在遮挡关系（也就是这两个点不在同一平面上，如果在同一平面上，距离相差不会太大）
                    远处的点会被遮挡，标记一下该点以及相邻的5个点，后面不再进行特征提取
                */
                if (depth1 - depth2 > 0.3){          // 这样depth1容易被遮挡，因此其之前的5个点都设置为无效点（depth1比较远）
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){          //同理，depth2比较远
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            //用前后相邻点判断当前点所在平面是否与激光束方向平行（如果两点距离比较大 就很可能是平行的点，也很可能失去观测）
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));   //前一个点与当前点之间的距离
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));   //后一个点与当前点之间的距离
            //如果当前点距离左右邻点都过远，则视其为瑕点，因为入射角可能太小导致误差较大
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    // 提取特征
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        //创建pcl点云的智能指针  
        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)  //遍历每一根扫描
        {
            surfaceCloudScan->clear();
            // 将一条扫描线扫描一周的点云数据，划分为6段，每段分开提取有限数量的特征，保证特征均匀分布
            for (int j = 0; j < 6; j++)       
            {
                /*
                 根据之前得到的每个scan的起始和结束id来均分：（startRingIndex和 endRingIndex 在imageProjection.cpp中的 cloudExtraction函数里被填入）
                    startRingIndex为扫描线起始第5个激光点在一维数组中的索引（注意：所有的点云在这里都是以"一维数组"的形式保存）   
                    
                    !假设 当前ring在一维数组中起始点是m，结尾点为n（不包括n），那么6段的起始点分别为：
                    *m + [(n-m)/6]*j  （ j从0～5） 
                    *化简为 [（6-j)*m + nj ]/6
                    !6段的终止点分别为：
                    *m + [(n-m)/6]*j+ (n-m)/6  -1           （每段终=每段起+1/6均分段-1）（ j从0～5）(减去1,避免每段首尾重复)
                    *化简为 [（5-j)*m + (j+1)*n ]/6 -1
                    这块不必细究边缘值到底是不是划分的准（例如考虑前五个点是不是都不要，还是说只不要前四个点），
                    只是尽可能的分开成六段，首位相接的地方不要。因为庞大的点云中，一两个点其实无关紧要
                */
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;
                // 这种情况就不正常（起始>结束）
                if (sp >= ep)      
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());     //按照曲率从小到大升序排列点云

                int largestPickedNum = 0;
                //按照曲率从大到小遍历每段
                for (int k = ep; k >= sp; k--)       
                {
                    int ind = cloudSmoothness[k].ind;    // 找到这个点对应的原先的idx
                    //当前激光点还未被处理，且曲率大于阈值（1.0)，则认为是角点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)    // edgeThreshold: 1.0
                    {
                        largestPickedNum++;     //角点计数
                        // 每段最多找20个角点（ 每段只取20个角点，如果单条扫描线扫描一周是1800个点，则划分6段，每段300个点，从中提取20个角点）
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;                            //*标签置1表示是角点（初始赋值为0）
                            cornerCloud->push_back(extractedCloud->points[ind]);    // 这个点收集进存储角点的点云中
                        } else {
                            break;
                        }                   
                        cloudNeighborPicked[ind] = 1;    // 标记已被处理

                        // 将这个点周围的几个点（10个点）设置成遮挡点，避免选取太集中
                        for (int l = 1; l <= 5; l++)      // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            // 列idx距离太远就算了，空间上也不会太集中
                            if (columnDiff > 10)           //大于10，说明距离远，则不作标记
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)       //同理（同一条扫描线上前5个点标记一下，不再处理，避免特征聚集）
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 开始收集面点（曲率从小到大遍历每段）
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    // 当前激光点还未被处理，且曲率小于阈值，则认为是平面点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)        //surfThreshold: 0.1
                    {

                        cloudLabel[ind] = -1;            //-1表示面点
                        cloudNeighborPicked[ind] = 1;
                        // 同理 把周围的点都设置为遮挡点
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }

                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {  // 注意这里是小于等于0,也就是说不是角点的都认为是面点了（cloudLabel初始值为0）
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
            /*
                用surfaceCloudScan来装数据，然后放到downSizeFilter里，
                再用downSizeFilter进行.filter()操作，把结果输出到*surfaceCloudScanDS里。
                最后把DS装到surfaceCloud中。DS指的是DownSample。
                同样角点（边缘点）则没有这样的操作，直接就用cornerCloud来装点云
            */
            // 因为面点太多了，所以做一个下采样
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);      //设置需要过滤的点云给滤波对象(参数：指针)
            downSizeFilter.filter(*surfaceCloudScanDS);            //执行滤波处理，存储输出
            // 加入平面点云集合
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    // 将一些不会用到的存储空间释放掉
    void freeCloudInfoMemory()
    {
        //分区用
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        // 判断点是否遮挡和平行用
        cloudInfo.pointColInd.clear();
        //计算曲率用
        cloudInfo.pointRange.clear();
    }

    //发布角点、面点点云，发布带特征点云数据的当前激光帧点云信息
    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        // 发布角点、面点点云，用于rviz展示
        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        /*
        发布当前激光帧点云信息，加入了角点、面点点云数据，发布给mapOptimization
            和imageProjection.cpp发布的不是同一个话题，
            image发布的是"lio_sam/deskew/cloud_info"，
            这里发布的是"lio_sam/feature/cloud_info"，
            因此不用担心地图优化部分的冲突
        */
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

//!主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");     //节点初始化

    FeatureExtraction FE;               //创建类对象

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");  //打印日至信息
   
    ros::spin();               //循环调用

    return 0;
}