#ifndef MY_MACROS_H
#define MY_MACROS_H

#define _DIM 3

enum class wellParticleDistributionType{
    LAYERED, /**<The particles are organized into layers equally distributed */
    SPIRAL,  /**< The Particles are distributed on a spiral around the well */
    LAYROT, /**<This is similar to the layer, however each layer is rotated slightly */
    INVALID /**< This is a placeholder for not valid types*/
};


#endif // MY_MACROS_H
