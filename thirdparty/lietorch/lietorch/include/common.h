#ifndef COMMON_H
#define COMMON_H

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
// EIGEN_RUNTIME_NO_MALLOC removed: causes access violations on GPU
// when Eigen internally needs temporaries in device code (e.g. inv())

#define EPS 1e-6
#define PI 3.14159265358979323846


#endif

