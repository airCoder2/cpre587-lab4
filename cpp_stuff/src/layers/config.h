// config.h
#include "../Types.h"

#ifndef CONFIG_H
#define CONFIG_H

#include <array>

// A read-only array safe to include in multiple source files
inline constexpr std::array<float, 12>  SW_VALS = {419.3088582098988, 260.8992077660963, 0, 183.42577326208328, 234.51324444962967, 0, 236.64045885476003, 248.70011854276902, 0, 0, 227.76791452790815, 95.91284026290671};
// Layer index: 0 conv2d, 1 conv2d_1, 2 mp, 3 conv2d_2, 4 conv2d_3, 5 mp, 6 conv2d_4, 7 conv2d_5, 8 mp, 9 flatten, 10 dense, 11 dense_1
// SI/SZ are now calibrated on each layer's INPUT (fixed in quantizator.ipynb)
// old SI_VALS = {77.48698028248077, 30.711071488247477, 0, 35.757093755677076, 29.98245171430904, 0,  22.069204744996554, 14.039217546749096, 0, 0, 8.156033096333804, 127.73635003263591};
// old SZ_VALS = {-3, -1, 0, -1, -1,0, -3, -1, 0, 0, -5, -1};
inline constexpr std::array<float, 12>  SI_VALS = {228.2462633602231, 77.4869802658987, 0, 30.863847964644368, 35.757091355232916, 0, 30.157240469096152, 22.06919743390229, 0, 0, 14.144781844598391, 8.15603009772199};
inline constexpr std::array<int8_t, 12> SZ_VALS = {-101, -3, 0, -2, -1, 0, -2, -3, 0, 0, -2, -5};


// If using traditional C-style arrays:
// inline constexpr int MY_C_ARRAY[] = {10, 20, 30, 40, 50};

#endif