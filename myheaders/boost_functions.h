#ifndef BOOST_FUNCTIONS_H
#define BOOST_FUNCTIONS_H

#include <deque>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/algorithms/assign.hpp>
#include <boost/foreach.hpp>

#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/assign.hpp>

//#include <boost/utility.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
//#include <boost/graph/visitors.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef std::pair<int, int> Edge;



template <typename Point>
void list_coordinates(Point const& p){
    using boost::geometry::get;
    std::cout << "x = " << get<0>(p) << " y = " << get<1>(p) << std::endl;
}

template <typename Point>
void list_coordinatesX_oneline(Point const& p){
    using boost::geometry::get;
    std::cout << get<0>(p) << " ";
}

template <typename Point>
void list_coordinatesY_oneline(Point const& p){
    using boost::geometry::get;
    std::cout << get<1>(p) << " ";
}


/*!
 * \brief polyXpoly Finds the intersection between two polygons. This use boost library
 * \param x1 x coordinates of 1st polygon
 * \param y1 y coordinates of 1st polygon
 * \param x2 x coordinates of 2nd polygon
 * \param y2 y coordinates of 2nd polygon
 * \param xc x coordinate of the barycenter of the intersection of the polygons
 * \param yc y coordinate of the barycenter of the intersection of the polygons
 * \return the area of the intersection.
 */
double polyXpoly(std::vector<double>& x1, std::vector<double>& y1,
                 std::vector<double>& x2, std::vector<double>& y2,
                 double& xc, double& yc ){
    double area = 0;
    typedef boost::geometry::model::d2::point_xy<double> strm_point;
    typedef boost::geometry::model::polygon<strm_point> strm_polygon;
    std::vector<strm_point> pnts;

    for (unsigned int i = 0; i < x1.size(); ++i){
        pnts.push_back(strm_point(x1[i], y1[i]));
    }
    strm_polygon Poly1;
    boost::geometry::assign_points(Poly1, pnts);
    boost::geometry::correct(Poly1);

    std::vector<strm_point> pnts2;
    for (unsigned int i = 0; i < x2.size(); ++i){
        pnts2.push_back(strm_point(x2[i], y2[i]));
    }
    strm_polygon Poly2;
    boost::geometry::assign_points(Poly2, pnts2);
    boost::geometry::correct(Poly2);

    std::deque<strm_polygon> output;
    boost::geometry::intersection(Poly1, Poly2, output);

    BOOST_FOREACH(strm_polygon const& p, output){
        area = boost::geometry::area(p);
        strm_point cntr(0,0);
        boost::geometry::centroid(p,cntr);
        // Plot intersction ------------
        //std::cout << "plot([";
        //boost::geometry::for_each_point(p,list_coordinatesX_oneline<strm_point>);
        //std::cout << "],[";
        //boost::geometry::for_each_point(p,list_coordinatesY_oneline<strm_point>);
        //std::cout << "])" << std::endl;
        //------------------------------
        xc = cntr.x();
        yc = cntr.y();
        break;
    }
    return area;
}

/*!
 * \brief simplify_polyline simplifies a polyline based on the threshold using the
 * Ramer–Douglas–Peucker algorithm
 * https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm.
 * \param thres Is the threshold
 * \param x A vector of the x coordinates of the polyline
 * \param y A vector of the y coordinates of the polyline
 * \param z A vector of the z coordinates of the polyline
 */
void simplify_polyline(double thres,
                       std::vector<double>& x,
                       std::vector<double>& y,
                       std::vector<double>& z){
    if (x.size() != y.size() || x.size() != z.size() || z.size() != y.size())
         std::cerr << "The x, y and z vectors must have the same size" << std::endl;

    using namespace boost::assign;
    using boost::geometry::get;
    typedef boost::geometry::model::point<double,3,
            boost::geometry::cs::cartesian> pnt;
    boost::geometry::model::linestring<pnt> pline;
    boost::geometry::model::linestring<pnt> simplified;
    for (unsigned int i = 0; i < x.size(); ++i){
        pline += pnt(x[i], y[i], z[i]);
    }

    boost::geometry::simplify(pline, simplified, thres);

    std::vector<pnt> const& points = simplified;
    x.clear();
    y.clear();
    z.clear();
    for (std::vector<pnt>::size_type i = 0; i < points.size(); ++i){
        x.push_back(get<0>(points[i]));
        y.push_back(get<1>(points[i]));
        z.push_back(get<2>(points[i]));
    }

}

#endif // BOOST_FUNCTIONS_H
