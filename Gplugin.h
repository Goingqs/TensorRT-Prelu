#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

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
//#include "fp16.h"
#include "NvInferPlugin.h"
#include <sys/time.h>


using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;


static Logger gLogger;

void caffeToGIEModel(const std::string& deployFile,             // name for caffe prototxt
                     const std::string& modelFile,              // name for model 
                     const std::vector<std::string>& outputs,   // network outputs
                     unsigned int maxBatchSize,					// easy to learn
                     unsigned int workSpaceSize,				// max workspace for this engine
                     nvcaffeparser1::IPluginFactory* pluginFactory,          // arg for parse custom layer
                     const std::string& serializeFile);

/*
	read model from file
	if the file doesn't exist
	the engine_buffer will be null
	engine_buffer_size will be 0
	you should check them after the func return;

	coded by Qiushan Guo
*/
void ReadModel(const std::string& fileName, std::shared_ptr<char>& engine_buffer, int& engine_buffer_size);



/*
	SliceLayer when you construct the layer by std::vector<int>& channels
	std::vector<int>& channels should contain slice_points and the num of input's channel.
	This code only support slice axis = 1 ! 
	e.g.

	slice_param {
    axis: 1
    slice_point: 3
    slice_point: 6
    slice_point: 9
    slice_point: 12
  	}
	
	means new SliceLayer<5>({3,6,9,12,15})

	coded by Qiushan Guo
*/

template<int OutC>
class SliceLayer : public IPlugin
{
public:
    SliceLayer(const std::vector<int>& channels){
    	assert(channels.size() == OutC);
    	sliceChannel[0] = 0;
    	for (int i=0; i<channels.size(); ++i){
    		sliceChannel[i+1] = channels[i];
    	}
    }
    SliceLayer(const void* buffer,size_t size)
    {
        assert(size == (3 + OutC +1) * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
        for (int i=0; i<OutC+1; i++){
        	sliceChannel[i] = d[3+i];
        }
    }

    inline int getNbOutputs() const override { return OutC; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(3 == inputs[0].nbDims);
        assert(index < OutC);
        return DimsCHW(sliceChannel[index+1]-sliceChannel[index], inputs[0].d[1], inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
    	for (int i=0; i<OutC; i++){
    		for (int j=0;j<batchSize;j++){
    			int batchOffset = dimBottom.c()*dimBottom.h()*dimBottom.w();
    			cudaMemcpyAsync(outputs[i] + j * (sliceChannel[i+1] - sliceChannel[i]) * _size * sizeof(float), 
    				inputs[0] + (batchOffset * j + sliceChannel[i] * _size) * sizeof(float) , 
    				(sliceChannel[i+1] - sliceChannel[i]) * _size * sizeof(float) , 
    				cudaMemcpyDeviceToDevice,stream);
    		}
    	}
        return 0;
    }


    size_t getSerializationSize() override
    {
        return (3+OutC+1) * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
        for (int i=0;i < OutC+1; i++){
        	d[3+i] = sliceChannel[i];
        }
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
    int sliceChannel[OutC+1];
};



/*
	Prelu layer 
	My code doesn't channel_shared_ (only one param), 
	that is a case of Leaky ReLU ( you can implement it by nvinfer1::plugin::createPReLUPlugin)
	coded by Qiushan Guo
*/


class PreluPlugin : public IPlugin
{
public:
	PreluPlugin(const Weights *weights, int nbWeights);
	PreluPlugin(const void* buffer, size_t size);
	~PreluPlugin();
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims);
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream);
	int getNbOutputs() const override
    {
        return 1;
    };
    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;
    void serialize(void* buffer) override;
    size_t getSerializationSize() override;
    inline size_t getWorkspaceSize(int) const override { return 0; }
    int initialize() override;
    void terminate() override;
protected:
	int input_c;
	int input_h;
	int input_w;
	int input_count;
	bool channel_shared_ {false};
	Weights mWeights;
	void* mDeviceKernel{nullptr};

private:
	void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        cudaMalloc(&deviceData, count);
        cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice);
        return deviceData;
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    size_t type2size(DataType type) { return sizeof(float); }

    void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
    {
        memcpy(buffer, weights.values, weights.count * type2size(weights.type));
        buffer += weights.count * type2size(weights.type);
    }
};


/*
	PluginFactory 
	My code only support PReLU and SliceLayer now, 
	Some custom layers are implemented in the SSDDetection.
	if you wanna merge them, you can do it yourself~~~~
	
	coded by Qiushan Guo
*/

class PluginFactory: public nvinfer1::IPluginFactory,
                      public nvcaffeparser1::IPluginFactory {
public:
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override  ;
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override ; 
    bool isPlugin(const char* name) override;
    void destroyPlugin();
private:
    std::map<std::string, IPlugin* > _nvPlugins; 
};

#endif