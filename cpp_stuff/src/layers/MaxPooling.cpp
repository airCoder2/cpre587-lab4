#include "MaxPooling.h"

#include <iostream>
#include <cmath>
#include <algorithm>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    inline size_t get_input_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t image_w, size_t image_d){
        return image_w * image_d * height_idx + image_d * width_idx + depth_idx;
    }

    inline size_t get_out_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t out_w, size_t out_d){
        return out_w * out_d * height_idx + out_d * width_idx + depth_idx;
    }


    inline size_t get_out_flat_idx_cm(size_t height_idx, size_t width_idx, size_t depth_idx, size_t out_w, size_t out_h){
        return out_w * out_h * depth_idx + out_w * height_idx + width_idx;
    }

    inline size_t get_input_flat_idx_cm(size_t height_idx, size_t width_idx, size_t depth_idx, size_t img_w, size_t img_h){
        //return kernel_w * kernel_d * kernel_b * height_idx + kernel_d * kernel_b * width_idx + kernel_b * depth_idx + batch_idx;
        return (depth_idx * img_w * img_h + height_idx * img_w + width_idx);
    }

    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    // ME: the LayerData object passed has been loaded before (by img.loadData() no need to do it again)
    void MaxPoolingLayer::computeNaive(const LayerData &dataIn, const int layer_num) const
    {
        size_t input_h = getInputParams().dims[0];
        size_t input_w = getInputParams().dims[1];
        size_t input_d = getInputParams().dims[2];

        size_t output_h = getOutputParams().dims[0];
        size_t output_w = getOutputParams().dims[1];
        size_t output_d = getOutputParams().dims[2];

        size_t i = 0, j = 0, k = 0;

        for (k = 0; k < input_d; k++)
        {
            for (i = 0; i < input_h; i+=2)
            {
                for (j = 0; j < input_w; j+=2)
                {
                    getOutputData().get<i8>(get_out_flat_idx_cm(i/2, j/2, k, output_w, output_h)) = std::max
                    ({
                        dataIn.get<i8>(get_input_flat_idx_cm(i, j, k, input_w, input_h)),
                        dataIn.get<i8>(get_input_flat_idx_cm(i, j+1, k, input_w, input_h)),
                        dataIn.get<i8>(get_input_flat_idx_cm(i+1, j, k, input_w, input_h)),
                        dataIn.get<i8>(get_input_flat_idx_cm(i+1, j+1, k, input_w, input_h))
                    });
                }
                
            }
            
        }
        
    }

    void MaxPoolingLayer::computeThreaded(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void MaxPoolingLayer::computeTiled(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void MaxPoolingLayer::computeSIMD(const LayerData &dataIn, const int layer_num) const
    {
        MaxPoolingLayer::computeNaive(dataIn, layer_num);
        // TODO: Your Code Here...
    }
} // namespace ML
