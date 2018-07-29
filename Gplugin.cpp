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
#include <ctype.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "NvInferPlugin.h"
#include <sys/time.h>
#include <fstream>
#include "Gplugin.h"
#include "GpluginGPU.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;



PreluPlugin::PreluPlugin(const Weights *weights, int nbWeights){
    assert(nbWeights==1);
    mWeights = weights[0];
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    mWeights.values = malloc(mWeights.count*type2size(mWeights.type));
    memcpy(const_cast<void*>(mWeights.values),weights[0].values,mWeights.count*type2size(mWeights.type));
}

PreluPlugin::PreluPlugin(const void* buffer, size_t size)
{
    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    read<int>(d,input_c);
    read<int>(d,input_h);
    read<int>(d,input_w);
    read<int>(d,input_count);
    read<bool>(d,channel_shared_);
    read<int64_t>(d,mWeights.count);
    read<DataType>(d,mWeights.type);
    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));//deserializeToDevice(d,mDeviceKernel,mWeights.count);
    memcpy(const_cast<void*>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    d += mWeights.count * type2size(mWeights.type);
    assert(d == a + size);
}

PreluPlugin::~PreluPlugin()
{   

    //std::cout << "~PreluPlugin  "<< mWeights.values << std::endl;
    if (mWeights.values){
        free(const_cast<void*>(mWeights.values));
    }
}

Dims PreluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}


void PreluPlugin::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int){
    input_c = inputs[0].d[0]; 
    input_h = inputs[0].d[1];
    input_w = inputs[0].d[2];
    input_count = input_c * input_h * input_w;
}

size_t PreluPlugin::getSerializationSize() {
    return 4*sizeof(int) + sizeof(bool) + sizeof(mWeights.count) 
    + sizeof(mWeights.type) +  mWeights.count * type2size(mWeights.type);
}

void PreluPlugin::serialize(void* buffer) {
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, input_c);
    write(d, input_h);
    write(d, input_w);
    write(d, input_count);
    write(d, channel_shared_);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d,mWeights);
    assert(d == a + getSerializationSize());
}

int PreluPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    const float *bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float *top_data = reinterpret_cast<float*>(outputs[0]);

    const int count = batchSize * input_count;
    const int dim = input_h*input_w;
    const int channels = input_c;
    const int div_factor = channel_shared_ ? channels : 1; //channel_shared_ default is false

    PReLUForward(count,channels,dim,bottom_data,top_data,mDeviceKernel,div_factor);

    return 0;
}

int PreluPlugin::initialize(){
    //std::cout << "~initialize  "<< mDeviceKernel << std::endl;
    cudaMalloc(&mDeviceKernel,mWeights.count*type2size(mWeights.type));
    cudaMemcpy(mDeviceKernel,mWeights.values,mWeights.count*type2size(mWeights.type),cudaMemcpyHostToDevice);
    return 0;
}

void PreluPlugin::terminate(){
    if (mDeviceKernel){
        //std::cout << "~terminate  "<< mDeviceKernel << std::endl;
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}

bool PluginFactory::isPlugin(const char* name)
{
    std::string strName {name};
    std::transform(strName.begin(),strName.end(),strName.begin(),::tolower);
    return(strName.find("prelu") != std::string::npos || strName.find("slice") != std::string::npos );
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights){
    assert(isPlugin(layerName));

    std::string strName {layerName};
    std::transform(strName.begin(),strName.end(),strName.begin(),::tolower);

    if (strName.find("prelu") != std::string::npos){
        _nvPlugins[layerName] = (IPlugin*)(new PreluPlugin(weights,nbWeights));
        return _nvPlugins.at(layerName);
    }
    else if (strName.find("slice") != std::string::npos){
        _nvPlugins[layerName] = (IPlugin*)(new SliceLayer<5>({3,6,9,12,15}));
        return _nvPlugins.at(layerName);
    }
    else{
        std::cout << "warning : " << layerName << std::endl;
        assert(0);  
        return nullptr;  
    }
}
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    assert(isPlugin(layerName));

    std::string strName {layerName};
    std::transform(strName.begin(),strName.end(),strName.begin(),::tolower);

    if (strName.find("prelu") != std::string::npos){
        _nvPlugins[layerName] = (IPlugin*)(new PreluPlugin(serialData,serialLength));
        return _nvPlugins.at(layerName);
    }
    else if (strName.find("slice") != std::string::npos){
        _nvPlugins[layerName] = (IPlugin*)(new SliceLayer<5>(serialData,serialLength));
        return _nvPlugins.at(layerName);
    }
    else{
        std::cout << "warning : " << layerName << std::endl;
        assert(0);  
        return nullptr;  
    }
}

void PluginFactory::destroyPlugin(){
    for (auto it = _nvPlugins.begin(); it!=_nvPlugins.end(); it++){
        if (strstr(it->first.c_str(),"prelu")){
            delete (PreluPlugin*)(it->second);
        }
        else if (strstr(it->first.c_str(),"slice")){
            delete (SliceLayer<5>*)(it->second);
        }
        _nvPlugins.erase(it);
    }
}


void caffeToGIEModel(const std::string& deployFile,             // name for caffe prototxt
                     const std::string& modelFile,              // name for model 
                     const std::vector<std::string>& outputs,   // network outputs
                     unsigned int maxBatchSize,
                     unsigned int workSpaceSize,
                     nvcaffeparser1::IPluginFactory* pluginFactory,                 // batch size - NB must be at least as large as the batch we want to run with)
                     const std::string& serializeFile)    // output buffer for the GIE model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    IHostMemory* gieModelStream {nullptr};
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(workSpaceSize);

    ICudaEngine* engine = builder->buildCudaEngine(*network);  
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();

    std::cout << "RT init done!" << std::endl;
    
    std::ofstream out(serializeFile.c_str(),std::ios::out|std::ios::binary);
    out.write((const char*)(gieModelStream->data()),gieModelStream->size());
    out.close();

    if (gieModelStream) 
    {
        gieModelStream->destroy();
        gieModelStream = nullptr;
    }
}

void ReadModel(const std::string& fileName, std::shared_ptr<char>& engine_buffer, int& engine_buffer_size){
    std::ifstream in(fileName.c_str(),std::ios::in | std::ios::binary);
    if (!in.is_open()){
        engine_buffer_size = 0;
        engine_buffer = nullptr;
    }

    in.seekg(0,std::ios::end);
    engine_buffer_size = in.tellg();
    in.seekg(0,std::ios::beg);
    engine_buffer.reset(new char[engine_buffer_size]);
    in.read(engine_buffer.get(),engine_buffer_size);
    in.close();
}
