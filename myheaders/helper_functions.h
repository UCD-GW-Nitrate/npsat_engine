#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <math.h>
#include <string>

/*!
 * \brief linspace generate a linearly spaced vector between the two numbers min and max
 * \param min is the lower end
 * \param max is the upper end
 * \param n is the number of numbers to generate between the min and max
 * \return a vector of n numbers linearly spaced
 */
std::vector<double> linspace(double min, double max, int n){
    std::vector<double> result;
    int iterator = 0;
    for (int i = 0; i <= n-2; i++){
        double temp = min + i*(max-min)/(floor(static_cast<double>(n)) - 1);
        result.insert(result.begin() + iterator, temp);
        iterator += 1;
    }
    result.insert(result.begin() + iterator, max);
    return result;
}

/*!
 * \brief is_input_a_scalar check if the string can be converted into a scalar value
 * \param input is the string to test
 * \return true if the input can be a scalars
 */
bool is_input_a_scalar(std::string input){
    // try to convert the input to scalar
    bool outcome;
    try{
        double value = stod(input);
        value++; // something to surpress the warning
        outcome = true;
    }
    catch(...){
        outcome = false;
    }
    return outcome;
}

#endif // HELPER_FUNCTIONS_H
