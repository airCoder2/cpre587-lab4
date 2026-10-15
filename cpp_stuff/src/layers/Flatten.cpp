
#include <iostream>
#include <cmath>
#include <algorithm>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"
#include "Flatten.h"

namespace ML
{

    // --- Begin Student Code ---

    // {4, 4, 128}
    // ME: the LayerData object passed has been loaded before (by img.loadData() no need to do it again)
    void FlattenLayer::computeNaive(const LayerData &dataIn, const int layer_num) const
    {
        // flatten was one line, because I added a new = operator that takes care of it in LayerData object
//        getOutputData() = dataIn;
        
        size_t input_h = dataIn.getParams().dims[0];
        size_t input_w = dataIn.getParams().dims[1]; 
        size_t input_d = dataIn.getParams().dims[2]; 

        for (size_t i = 0; i < input_h; i++)
        {
            for (size_t j = 0; j < input_w; j++)
            {
                for (size_t k = 0; k < input_d; k++)
                {
                    getOutputData().get<i8>(i * input_w * input_d + j * input_d + k) = dataIn.get<i8>(k*input_h * input_w + i * input_w + j);
                }
            }
            
        }
        
    }

    void FlattenLayer::computeThreaded(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void FlattenLayer::computeTiled(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void FlattenLayer::computeSIMD(const LayerData &dataIn, const int layer_num) const
    {
        FlattenLayer::computeNaive(dataIn, layer_num);
        // TODO: Your Code Here...
    }
} // namespace ML
