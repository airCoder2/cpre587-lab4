#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Flatten.h"
#include "layers/Softmax.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML
{

    // Build our ML toy model
    Model buildToyModel(const Path modelPath)
    {
        Model model;
        logInfo("--- Building Toy Model ---");

        // --- Conv 1: L1 ---
        // Input shape: 64x64x3
        // Output shape: 60x60x32

        // You can pick how you want to implement your layers, both are allowed:

        // LayerParams conv1_inDataParam(sizeof(fp32), {64, 64, 3});
        // LayerParams conv1_outDataParam(sizeof(fp32), {60, 60, 32});
        // LayerParams conv1_weightParam(sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin");
        // LayerParams conv1_biasParam(sizeof(fp32), {32}, modelPath / "conv1_biases.bin");
        // auto conv1 = new ConvolutionalLayer(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam);

        // ME: I should add a checker for conv layers to make sure out and in shapes are good given the kernel size

//        model.addLayer<ConvolutionalLayer>(
//            LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
//            LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
//            LayerParams{sizeof(fp32), {5, 5, 3, 32}, modelPath / "model_data" / "conv2d_weights.bin"}, // Weights
//            LayerParams{sizeof(fp32), {32}, modelPath / "model_data" / "conv2d_biases.bin"}            // Bias
//        );

        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
            LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
            LayerParams{sizeof(i8), {5, 5, 3, 32}, modelPath / "quantized_model_data"/"weights_8bit" / "conv2d_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {32}, modelPath / "quantized_model_data"/ "biases_8bit" / "conv2d_biases.bin"}            // Bias
        );

//        model.addLayer<ConvolutionalLayer>(
//            LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
//            LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
//            LayerParams{sizeof(i8), {5, 5, 3, 32}, modelPath / "quantized_model_data" / "conv2d_quantized_weights.bin"}, // Weights
//            LayerParams{sizeof(i32), {32}, modelPath / "quantized_model_data" / "conv2d_quantized_combined_biases.bin"}            // Bias
//        );



   //     // --- Conv 2: L2 ---
   //     // Input shape: 60x60x32
   //     // Output shape: 56x56x32
   //     model.addLayer<ConvolutionalLayer>(
   //         LayerParams{sizeof(fp32), {60, 60, 32}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {56, 56, 32}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {5, 5, 32, 32}, modelPath / "conv2_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {32}, modelPath / "conv2_biases.bin"}            // Bias
   //     );

   //     // --- Conv 2: L2 ---
   //     // Input shape: 60x60x32
   //     // Output shape: 56x56x32
   //     model.addLayer<ConvolutionalLayer>(
   //         LayerParams{sizeof(fp32), {60, 60, 32}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {56, 56, 32}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {5, 5, 32, 32}, modelPath / "conv2_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {32}, modelPath / "conv2_biases.bin"}            // Bias
   //     );

   //     // --- MPL 1: L3 ---
   //     // Input shape: 56x56x32
   //     // Output shape: 28x28x32
   //     model.addLayer<MaxPoolingLayer>(
   //         LayerParams{sizeof(fp32), {56, 56, 32}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {28, 28, 32}}                                    // Output Data
   //     );

   //     // --- Conv 3: L4 ---
   //     // Input shape: 28x28x32
   //     // Output shape: 26x26x64
   //     model.addLayer<ConvolutionalLayer>(
   //         LayerParams{sizeof(fp32), {28, 28, 32}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {26, 26, 64}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {3, 3, 32, 64}, modelPath / "conv3_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {64}, modelPath / "conv3_biases.bin"}            // Bias
   //     );

   //     // --- Conv 4: L5 ---
   //     // Input shape: 26x26x64
   //     // Output shape: 24x24x64
   //     model.addLayer<ConvolutionalLayer>(
   //         LayerParams{sizeof(fp32), {26, 26, 64}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {24, 24, 64}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv4_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {64}, modelPath / "conv4_biases.bin"}            // Bias
   //     );

   //     // --- MPL 2: L6 ---
   //     // Input shape: 24x24x64
   //     // Output shape: 12x12x64
   //     model.addLayer<MaxPoolingLayer>(
   //         LayerParams{sizeof(fp32), {24, 24, 64}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {12, 12, 64}}                                    // Output Data
   //     );

   //     // --- Conv 5: L7 ---
   //     // Input shape: 12x12x64
   //     // Output shape: 10x10x64
   //     model.addLayer<ConvolutionalLayer>(
   //         LayerParams{sizeof(fp32), {12, 12, 64}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {10, 10, 64}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv5_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {64}, modelPath / "conv5_biases.bin"}            // Bias
   //     );

   //     // --- Conv 6: L8 ---
   //     // Input shape: 10x10x64
   //     // Output shape: 8x8x128
   //     model.addLayer<ConvolutionalLayer>(
   //         LayerParams{sizeof(fp32), {10, 10, 64}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {8, 8, 128}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {3, 3, 64, 128}, modelPath / "conv6_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {128}, modelPath / "conv6_biases.bin"}            // Bias
   //     );

   //     // --- MPL 3: L9 ---
   //     // Input shape: 8x8x128
   //     // Output shape: 4x4x128
   //     model.addLayer<MaxPoolingLayer>(
   //         LayerParams{sizeof(fp32), {8, 8, 128}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {4, 4, 128}}                                    // Output Data
   //     );

   //     // --- Flatten 1: L10 ---
   //     // Input shape: 4x4x128
   //     // Output shape: 2048
   //     model.addLayer<FlattenLayer>(
   //         LayerParams{sizeof(fp32), {4, 4, 128}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {2048}}                                         // Output Data
   //     );

   //     // --- Dense 1: L11 ---
   //     // Input shape: 2048
   //     // Output shape: 256
   //     model.addLayer<DenseLayer>(
   //         LayerParams{sizeof(fp32), {2048}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {256}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {2048, 256}, modelPath / "dense1_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {256}, modelPath / "dense1_biases.bin"},            // Bias
   //         DenseLayer::ActivationType::ReLU
   //     );

   //     // --- Dense 2: L12 ---
   //     // Input shape: 256
   //     // Output shape: 200
   //     model.addLayer<DenseLayer>(
   //         LayerParams{sizeof(fp32), {256}},                                    // Input Data
   //         LayerParams{sizeof(fp32), {200}},                                   // Output Data
   //         LayerParams{sizeof(fp32), {256, 200}, modelPath / "dense2_weights.bin"}, // Weights
   //         LayerParams{sizeof(fp32), {200}, modelPath / "dense2_biases.bin"},            // Bias
   //         DenseLayer::ActivationType::SoftMax

   //     );

        // --- Softmax 1: L13 ---
        // Input shape: 200
        // Output shape: 200

        // I added softmax as an activation funciton to the previous one. So no need to implement it. I don't know why they have it here
        // as a separate layer, since ReLU activation function was always done before the output of a layer

        return model;
    }

    void runBasicTest(const Model &model, const Path &basePath)
    {
        logInfo("--- Running Basic Test ---");

        // Load an image
        // ME:  this line constructs an image and sets up its private LayerParams params attribute
        LayerData img = {{sizeof(fp32), {64, 64, 3}, basePath / "test_input" / "test_input_image.bin"}};
        // ME: this internally allocates and fills the data buffer with whatever is inside the file path passed when constructing the img object
        img.loadData();

        // Compare images
        std::cout << "Comparing image 0 to itself (max error): " << img.compare<fp32>(img) << std::endl
                  << "Comparing image 0 to itself (T/F within epsilon " << ML::Config::EPSILON << "): " << std::boolalpha
                  << img.compareWithin<fp32>(img, ML::Config::EPSILON) << std::endl;

        // Test again with a modified copy
        std::cout << "\nChange a value by 0.1 and compare again" << std::endl;

        LayerData imgCopy = img;
        imgCopy.get<fp32>(0) += 0.1;

        // Compare images
        img.compareWithinPrint<fp32>(imgCopy);

        // Test again with a modified copy
        log("Change a value by 0.1 and compare again...");
        imgCopy.get<fp32>(0) += 0.1;

        // Compare Images
        img.compareWithinPrint<fp32>(imgCopy);
    }

//                data[i] = 


    void runLayerTest(const std::size_t layerNum, const Model &model, const Path &input_bin_path, const Path &output_ground_truth_path)
    {
        // Load an image
        logInfo(std::string("\n\n\n--- Running Layer Test ") + std::to_string(layerNum) + "---");

        // Construct a LayerData object from a LayerParams one
        // LayerData img(model[layerNum].getInputParams(), test_image_files[layerNum].first);

        dimVec input_dimensions = model.getLayer(layerNum).getInputParams().dims;
        std::size_t input_element_size = model.getLayer(layerNum).getInputParams().elementSize;

        // See if you can initialize this is quantization path, and quantization type. So if you do that,
        // rather than allocating data for non quantized one, it does it for quantized
        LayerData input_data({input_element_size, input_dimensions, input_bin_path});
        input_data.loadData();

        Timer timer("Layer Inference");

        // Run inference on the model
        timer.start();
        const LayerData &output = model.inferenceLayer(input_data, layerNum, Layer::InfType::NAIVE);
        timer.stop();

        // Compare the output
        // Construct a LayerData object from a LayerParams one
        LayerData expected(output.getParams(), output_ground_truth_path);
        expected.loadData();
        output.compareWithinPrint<fp32>(expected);
    }

    void runInferenceTest(const Model &model, const Path& input_image_bin, const Path& ground_truth_pred_path)
    {
        // Load an image
        logInfo("\n\n\n--- Running Inference Test ---");

        // Construct a LayerData object from a LayerParams one
        LayerData img(model[0].getInputParams(), input_image_bin);
        img.loadData();

        Timer timer("Full Inference");

        // Run inference on the model
        timer.start();
        const LayerData &output = model.inference(img, Layer::InfType::NAIVE);
        timer.stop();

        // Compare the output
        // Construct a LayerData object from a LayerParams one
        LayerData expected(model.getOutputLayer().getOutputParams(), ground_truth_pred_path);
        expected.loadData();
        output.compareWithinPrint<fp32>(expected);
    }

    void runTests()
    {
        // Base input data path (determined from current directory of where you are running the command)
        Path basePath("../python_stuff"); // May need to be altered for zedboards loading from SD Cards

        // Build the model and allocate the buffers
        Model model = buildToyModel(basePath);
        model.allocLayers(); // ME: allocates memory to store out_data from each leayer

        // Run some framework tests as an example of loading data
        runBasicTest(model, basePath);

        // Run a layer inference test
        runLayerTest(0,  model, basePath / "test_input" / "test_input_image.bin", basePath / "test_input_feature_maps" / "conv2d_feature.bin");
        // Give the path of the actual image rather than the quantized

        // Clean up
        model.freeLayers();
        std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
    }

} // namespace ML

#ifdef ZEDBOARD
extern "C" int main()
{
    try
    {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK)
        {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n"
                  << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main()
{
    ML::runTests();
}
#endif