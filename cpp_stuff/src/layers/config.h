// config.h
#include "../Types.h"

#ifndef CONFIG_H
#define CONFIG_H

#include <array>

// A read-only array safe to include in multiple source files
inline constexpr std::array<float, 9>  SW_VALS = {419.3088582098988, 260.8992077660963, 0, 183.42577326208328, 234.51324444962967, 236.64045885476003, 248.70011854276902, 227.76791452790815, 95.91284026290671};
inline constexpr std::array<float, 9>  SI_VALS = {77.48698028248077, 30.711071488247477, 0, 35.757093755677076, 29.98245171430904, 22.069204744996554, 14.039217546749096, 8.156033096333804, 127.73635003263591};
inline constexpr std::array<int8_t, 9> SZ_VALS = {-3, -1, 0, -1, -1, -4, -1, -5, -1};


// If using traditional C-style arrays:
// inline constexpr int MY_C_ARRAY[] = {10, 20, 30, 40, 50};

#endif