#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cmath>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "NvInferPlugin.h"
#include <sys/time.h>
#include <fstream>
#include "Gplugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME1 = "global_pool";
const char* OUTPUT_BLOB_NAME2 = "last_bn";
const char* OUTPUT_BLOB_NAME3 = "feature";


long long ustime(void) {
    struct timeval tv;
    long long ust;
    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust;
}

/* Return the UNIX time in milliseconds */
long long mstime(void) {
    return ustime()/1000;
}

int main(int argc, char** argv){
	const char* model_def = argv[1];//"det3.prototxt";//
    const char* weights_def = argv[2];//"det3.caffemodel";//
    const char* image_name = argv[3]; //"/home/tmp_data_dir/zhaoyu/wider_face/WIDER_train/images/0--Parade/0_Parade_marchingband_1_173.jpg";//
    
    PluginFactory pluginFactory;
    caffeToGIEModel(model_def,weights_def,{OUTPUT_BLOB_NAME1,OUTPUT_BLOB_NAME2,OUTPUT_BLOB_NAME3},1,10<<20,&pluginFactory,"engine.binbin");

    ICudaEngine* engine;

    std::shared_ptr<char> engine_buffer {nullptr};
    int engine_buffer_size = 0;
    ReadModel("engine.binbin", engine_buffer, engine_buffer_size);

    std::cout << "IO done" << std::endl;

    PluginFactory pluginFactoryserialize;
    IRuntime* runtime = createInferRuntime(gLogger);
	engine = runtime->deserializeCudaEngine(engine_buffer.get(), engine_buffer_size, &pluginFactoryserialize);
    std::cout << "RT deserialize done!" << std::endl;

    IExecutionContext *context = engine->createExecutionContext();
    std::cout << "RT createExecutionContext done!" << std::endl;

    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME), 
        outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2),
        outputIndex3 = engine->getBindingIndex(OUTPUT_BLOB_NAME3);
     
    DimsCHW inputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(inputIndex)), 
            outputDims1 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex1)),
            outputDims2 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex2)),
            outputDims3 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex3));
    

    std::cout<<inputDims.c()<<" "<<inputDims.h()<<" "<<inputDims.w()<<std::endl;
    std::cout<<outputDims1.c()<<" "<<outputDims1.h()<<" "<<outputDims1.w()<<std::endl;
    std::cout<<outputDims2.c()<<" "<<outputDims2.h()<<" "<<outputDims2.w()<<std::endl;
    std::cout<<outputDims3.c()<<" "<<outputDims3.h()<<" "<<outputDims3.w()<<std::endl;

    cv::Mat img = cv::imread(image_name,1);
    cv::Size dsize = cv::Size(inputDims.h(),inputDims.w());
    cv::Mat imgResize;
    cv::resize(img, imgResize, dsize, 0, 0 , cv::INTER_LINEAR);

    float means[3] = {104.0, 117.0, 123.0};
    float *data = new float[inputDims.c() * inputDims.h() * inputDims.w()];

    float *Res1 = new float[outputDims1.c() * outputDims1.h() *outputDims1.w()];
    float *Res2 = new float[outputDims2.c() * outputDims2.h() *outputDims2.w()]; 
    float *Res3 = new float[outputDims3.c() * outputDims3.h() *outputDims3.w()]; 

    /*for (int i = 0; i < imgResize.rows; ++i){
        for (int j = 0; j < imgResize.cols; ++j){
            data[0*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[0]) - means[0];
            data[1*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[1]) - means[1];
            data[2*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[2]) - means[2];
        }
    }*/

    for (int i=0;i<inputDims.c();i++){
    	for (int j=0;j< inputDims.h()*inputDims.w();j++){
    		data[i*inputDims.h()*inputDims.w()+j] = i;
    	}
    }
    for (int i = 0; i < 100; ++i){
        std::cout << data[i] << " ";
    }


    void* buffers[4];
    int batchSize = 1;
    // create GPU buffers and a stream
    cudaMalloc(&buffers[inputIndex], batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float));
    cudaMalloc(&buffers[outputIndex1], batchSize * outputDims1.c() * outputDims1.h() * outputDims1.w() * sizeof(float));
    cudaMalloc(&buffers[outputIndex2], batchSize * outputDims2.c() * outputDims2.h() * outputDims2.w() * sizeof(float));
    cudaMalloc(&buffers[outputIndex3], batchSize * outputDims3.c() * outputDims3.h() * outputDims3.w() * sizeof(float));

    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    long long llStart = mstime();
    for (int i=0;i<10;i++){
    
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[inputIndex], data, batchSize * inputDims.c() * inputDims.h() * inputDims.w()  * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(batchSize, buffers, stream, nullptr);
    cudaMemcpyAsync(Res1, buffers[outputIndex1], batchSize * outputDims1.c() * outputDims1.h() * outputDims1.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(Res2, buffers[outputIndex2], batchSize * outputDims2.c() * outputDims2.h() * outputDims2.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(Res3, buffers[outputIndex3], batchSize * outputDims3.c() * outputDims3.h() * outputDims3.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    }
    long long llEnd = mstime();
    std::cout<<llEnd-llStart<<"ms"<<std::endl;

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
    cudaFree(buffers[outputIndex3]);

    std::cout << "\n\n";

    for (int i = 0; i < 100; ++i){
        std::cout << Res1[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 100; ++i){
        std::cout << Res2[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 100; ++i){
        std::cout << Res3[i] << " ";
    }
    std::cout << std::endl;

	context->destroy();
    engine->destroy();
    runtime->destroy();

	std::cout << "8" << std::endl;
    pluginFactoryserialize.destroyPlugin();
    std::cout << "9" << std::endl;
	
	return 0;
}