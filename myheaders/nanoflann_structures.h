#ifndef NANOFLANN_STRUCTURES_H
#define NANOFLANN_STRUCTURES_H(args)

#include <nanoflann.hpp>
#include "helper_functions.h"

struct PointId{
    double x;
    double y;
    int id;
};

struct PointIdCloud{
    std::vector<PointId> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim==0)
            return pts[idx].x;
        else
            return pts[idx].y;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Adaptor<double, PointIdCloud>,
        PointIdCloud,2> pointid_kd_tree;



typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointIdCloud > ,
        PointIdCloud, 2> pointIdDynamic_kd_tree;



struct PointVector{
    double x;
    double y;
    std::vector<double> values;
};

struct PointVectorCloud{
    std::vector<PointVector> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim==0)
            return pts[idx].x;
        else
            return pts[idx].y;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Adaptor<double, PointVectorCloud>,
        PointVectorCloud,2> pointVector_kd_tree;

void interpolateVectorCloud(PointVectorCloud& cloud,
                            std::shared_ptr<pointVector_kd_tree>& index,
                            double power, double radius, double thres,
                            double x, double y,
                            std::vector<double>& out){
    const double query_pt[2] = {x, y};
    std::vector<std::pair<size_t,double> > ret_matches;
    nanoflann::SearchParams params;
    params.sorted = false;
    std::vector< std::vector<double> > values;
    std::vector<double> distances;
    out.clear();
    const int nMatches = index->radiusSearch(&query_pt[0], radius, ret_matches, params);
    for (int i = 0; i < nMatches; ++i) {
        values.push_back(cloud.pts[ret_matches[i].first].values);
        distances.push_back(std::sqrt(ret_matches[i].second));
    }
    out = IDWinterp(values,distances,power,thres);
}

#endif //NANOFLANN_STRUCTURES_H