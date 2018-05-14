#ifndef POINT_SET_H
#define POINT_SET_H

#include <algorithm> // Definition for the min_element
#include <deal.II/base/point.h>

using namespace dealii;

/*!
 * \brief The MyPointSet class is used as container for a list of unique points. The points can be associated with values for
 * nearest neighbor interpolation. This class is used dusring the top and bottom values assigned in the mesh structure routine.
 * This replace the mixed_mesh approach.
 *
 */
template <int dim>
class MyPointSet{
public:
    //! This initializes the point set by setting the number of values per point and the threshold.
    //! The Points with distance less than #threshold are treated as identical
    MyPointSet();

    //! This initialize the set with number of values per point and the threshold
    void initialize(int Nval, double thres);

    //! Resets the point set. Deletes everything and is ready to accept points and values
    void reset();

    //! A method which asks if the node already exists
    bool point_exist(Point<dim> p);

    //! Adds a node in the set that we are sure it does not exist. This can be called under two circumstances.
    //! i) when we add the first node in the list. If we are sure that this node doesnt already exists by having called
    //! the  point_exist method before
    void add_non_existing_point(Point<dim> p, std::vector<double> Val);

    //! This adds the point only if doesnt exist. In fact this method calls first the point_exist method and then the
    //! add_non_existing_point if the point doesnt exist
    bool add_point(Point<dim> p, std::vector<double> val);

    //! This is the method that does the interpolation
    bool interpolate(Point<dim> p, std::vector<double>& val);

    //! Returns true if this set is empty
    bool is_empty();

private:
    //! This is the number of values associated with each point in the set
    int Nvalues;

    //! This is the threshold value.
    //! //! The Points with distance less than #threshold are treated as identical
    double threshold;

    //! THis is a the list of points in the set
    std::vector<Point<dim>>  points;

    //! The values associated with the points
    std::vector<std::vector<double>> values;

};

template <int dim>
MyPointSet<dim>::MyPointSet(){
    Nvalues = 1;
    threshold = 0.001;
}

template <int dim>
void MyPointSet<dim>::initialize(int Nval, double thres){
    Nvalues = Nval;
    threshold = thres;
    reset();
}

template<int dim>
void MyPointSet<dim>::reset(){
    points.clear();
    values.clear();
}

template<int dim>
void MyPointSet<dim>::add_non_existing_point(Point<dim> p, std::vector<double> Val){
    if (Val.size() != Nvalues){
        std::cerr << "The number of values per point is not correct" << std::endl;
    }
    points.push_back(p);
    values.push_back(Val);
}

template <int dim>
bool MyPointSet<dim>::add_point(Point<dim> p, std::vector<double> val){

    if (point_exist(p)){
        add_non_existing_point(p, val);
        return true;
    }
    else
        return false;
}

template<int dim>
bool MyPointSet<dim>::point_exist(Point<dim> p){
    for (unsigned int i = 0; i < points.size(); ++i){
        double dst = p.distance(points[i]);
        if (dst < threshold)
            return true;
    }
    return false;
}

template <int dim>
bool MyPointSet<dim>::interpolate(Point<dim> p, std::vector<double>& val){
    val.clear();

    std::vector<double> dst;
    for (unsigned int i = 0; i < points.size(); ++i)
        dst.push_back(p.distance(points[i]));


    std::vector<double>::iterator it = std::min_element(dst.begin(), dst.end());
    if(*it < threshold){
        int ii = static_cast<int>(std::distance(dst.begin(), it));
        val = values[ii];
        return true;
    }
    else return false;

}

template <int dim>
bool MyPointSet<dim>::is_empty(){
    return points.size() == 0;
}





#endif // POINT_SET_H
