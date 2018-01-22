#include <stdlib.h>
#include <cmath>
#include <numeric>
#include <tgmath.h>
#include <iostream>
#include <iomanip>
#include <ctime>
using namespace std;

class Neuron {
    public:
        Neuron(){rawValue = 0; delta = 0;};
        void setValue(double value){rawValue = activatedValue = value;};
        double getValue(){return activatedValue;};
        void setDelta(double value){delta = value;};
        double getDelta(){return delta;};
        void activate();
    private:
        double rawValue;
        double activatedValue;
        double delta;
};

void Neuron::activate(){
    activatedValue = 1/(1 + exp(-1 * rawValue));
}

class Layer {
    public:
        Layer(int size);
        int getSize(){return layerSize;};
        Neuron* getNeuron(int index){ return neurons[index];};
        double getNeuronValue(int index){ return neurons[index]->getValue();};
        double getNeuronDelta(int index){ return neurons[index]->getDelta();};
        double* getNeuronValues(bool includeBias);
        double* getNeuronDeltas();
        void setNeuronValue(int index, double value){ neurons[index]->setValue(value);};
        void setNeuronDelta(int index, double value){neurons[index]->setDelta(value);};
        void initializeTheta(int sizeOfLayer, int sizeOfNextLayer);
        void initializeThetaDeltaAccumulator(int sizeOfLayer, int sizeOfNextLayer);
        void activateNeuron(int index){ neurons[index]->activate();};
        void activateLayer();
        double** getThetas(){return theta;};
        double getThetaValue(int i, int j){return theta[i][j];};
        double** getThetaDeltas(){return thetaDeltas;};
        void setTheta(int i, int j, double value){theta[i][j] = value;};
        void computeNeuronDeltas(Layer* nextLayer);
        void addToThetaDeltaAccumulator(Layer* nextLayer);
        double getThetaDeltaAccumulatorValue(int i, int j){ return thetaDeltaAccumulator[i][j];};
        double** getThetaTranspose(int sizeOfLayer, int sizeOfNextLayer);
        void setThetaDelta(int i, int j, double value){thetaDeltas[i][j] = value;};
    private:
        Neuron** neurons;
        double** theta;
        double** thetaDeltaAccumulator;
        double** thetaDeltas;
        int layerSize;

};

Layer::Layer(int size){
    layerSize = size;
    neurons = new Neuron*[layerSize+1];
    for (int l = 0; l < layerSize+1; l++){
        neurons[l] = new Neuron;
    }
}

double* Layer::getNeuronValues(bool includeBias){
    int vectorSize = includeBias ? layerSize + 1 : layerSize;
    double* neuronValues = new double[vectorSize];
    neuronValues[0] = 1;
    for(int i = 0; i < vectorSize; i++){
        neuronValues[i] = getNeuronValue(includeBias ? i: i+1);
    }
    return neuronValues;
}

double* Layer::getNeuronDeltas(){
    double* neuronDeltas = new double[layerSize];
    for(int i = 1; i < layerSize+1; i++){
        neuronDeltas[i] = getNeuronDelta(i);
    }
    return neuronDeltas;
}

void Layer::initializeTheta(int sizeOfLayer, int sizeOfNextLayer){
    theta = new double*[sizeOfNextLayer];
    for (int i = 0; i < sizeOfNextLayer; i++){
        theta[i] = new double[sizeOfLayer + 1];
        for(int j = 0; j < sizeOfLayer + 1; j++){
            theta[i][j] = rand() % 2 - 0.5;
        }
    }
};

void Layer::initializeThetaDeltaAccumulator(int sizeOfLayer, int sizeOfNextLayer){
    thetaDeltaAccumulator = new double*[sizeOfNextLayer];
    for (int i = 0; i < sizeOfNextLayer; i++){
        thetaDeltaAccumulator[i] = new double[sizeOfLayer + 1];
        for(int j = 0; j < sizeOfLayer + 1; j++){
            thetaDeltaAccumulator[i][j] = 0;
        }
    }
};

void Layer::activateLayer(){
    for(int i = 1; i < layerSize+1; i++){
        activateNeuron(i);
    }
}

double** Layer::getThetaTranspose(int sizeOfLayer, int sizeOfNextLayer){
    double** thetaT = new double*[sizeOfLayer + 1];
    for(int i = 0; i < sizeOfLayer + 1; i++){
        for(int j = 0; j < sizeOfNextLayer; j++){
            thetaT[i][j] = theta[j][i];
        }
    }
    return thetaT;
}
void Layer::computeNeuronDeltas(Layer* nextLayer){
    double* nextLayerNeuronDeltas = nextLayer->getNeuronDeltas();
    double sizeOfNextLayer = nextLayer->getSize();
    double* thisLayerValues = getNeuronValues(true);
    double neuronDelta = 0;
    for(int i = 0; i < layerSize + 1; i++){
        for(int j = 0; j < sizeOfNextLayer; j++){
            neuronDelta += nextLayerNeuronDeltas[j] * theta[j][i] * thisLayerValues[i] * (1 - thisLayerValues[i]);
        }
        setNeuronDelta(i, neuronDelta);
    }

}

void Layer::addToThetaDeltaAccumulator(Layer* nextLayer){
    double* nextLayerDeltas = nextLayer->getNeuronDeltas();
    double sizeOfNextLayer = nextLayer->getSize();
    double* thisLayerValues = getNeuronValues(true);
    for(int i = 0; i < sizeOfNextLayer; i++){
        for(int j = 0; j < layerSize + 1; j++){
            thetaDeltaAccumulator[i][j] += thisLayerValues[j] * nextLayerDeltas[i];
        }
    }
}

class Network {
    public:
        Network(int sizeOfInputLayer, int numberOfHiddenLayers, int sizeOfHiddenLayers, int sizeOfOutputLayer);
        void setInputValues(double input[]);
        double* getOutput(double* input);
        Neuron* getNeuron(int indexOfLayer, int indexOfNeuron){return layers[indexOfLayer]->getNeuron(indexOfNeuron);};
        double getNeuronValue(int indexOfLayer, int indexOfNeuron){return getNeuron(indexOfLayer, indexOfNeuron)->getValue();};
        int getSizeOfInputLayer(){ return inputLayerSize;};
        int getSizeOfHiddenLayers() {return hiddenLayerSize;};
        int getSizeOfOutputLayer() {return outputLayerSize;};
        int getNumberOfHiddenLayers() {return numOfHiddenLayers;};
        int getNumberOfLayers() {return numOfHiddenLayers + 2;};
        Layer* getLayer(int index){return layers[index];};
        void feedForward();
        void backPropagate(double* label);
        void resetDeltaAccumulators();
        void train(double** X, double** y, int numOfTrainingExamples, double lambda, int iterations);
    private:
        Layer** layers;
        Layer* inputLayer;
        Layer* outputLayer;
        int inputLayerSize;
        int hiddenLayerSize;
        int outputLayerSize;
        int numOfHiddenLayers;
};

Network::Network(int sizeOfInputLayer, int numberOfHiddenLayers, int sizeOfHiddenLayers, int sizeOfOutputLayer){
    srand(time(0));
    numOfHiddenLayers = numberOfHiddenLayers;
    inputLayerSize = sizeOfInputLayer;
    hiddenLayerSize = sizeOfHiddenLayers;
    outputLayerSize = sizeOfOutputLayer;
    layers = new Layer*[numberOfHiddenLayers + 2]; // Initialize layer array
    layers[0] = new Layer(sizeOfInputLayer); // Initialize input layer
    layers[0]->initializeTheta(layers[0]->getSize(), sizeOfHiddenLayers); // Initialize input layer's theta matrix
    layers[0]->initializeThetaDeltaAccumulator(layers[0]->getSize(), sizeOfHiddenLayers);
    inputLayer = layers[0];
    for(int l = 1; l < numberOfHiddenLayers + 1; l++){
        layers[l] = new Layer(sizeOfHiddenLayers); // Initialize hidden layer l
        layers[l]->initializeTheta(layers[l]->getSize(), l == numberOfHiddenLayers ? sizeOfOutputLayer : sizeOfHiddenLayers); // Initialize hidden layer l's theta matrix
        layers[l]->initializeThetaDeltaAccumulator(layers[l]->getSize(), l == numberOfHiddenLayers ? sizeOfOutputLayer : sizeOfHiddenLayers);
    }
    layers[getNumberOfLayers() - 1] = new Layer(sizeOfOutputLayer); // Initialize output layer
    outputLayer = layers[numberOfHiddenLayers + 1];
}

void Network::setInputValues(double *input){
    for(int i = 0; i < inputLayer->getSize(); i++){
        inputLayer->setNeuronValue(i, input[i]); // Assign input values to neurons in input layer
    }
};

void Network::feedForward(){
    for(int l = 0; l < getNumberOfLayers() - 1; l++){
        Layer* thisLayer = getLayer(l);
        Layer* nextLayer = getLayer(l+1);
        double* neuronValues = thisLayer->getNeuronValues(true);
        double** thetaValues = thisLayer->getThetas();
        int rowCount = nextLayer->getSize();
        int colCount = thisLayer->getSize() + 1;
        for(int i = 0; i < rowCount; i++){
            double outputVal = inner_product(neuronValues, neuronValues + colCount, thetaValues[i], 0.0);
            nextLayer->setNeuronValue(i+1, outputVal);
            nextLayer->activateLayer();
        }
    }
    getLayer(getNumberOfLayers() - 1)->activateLayer(); // Activate the output layer
}

void Network::backPropagate(double* label){
    Layer* outputLayer = getLayer(getNumberOfLayers()-1);
    for(int i = 0; i < getSizeOfOutputLayer(); i++){
        outputLayer->setNeuronDelta(i+1, outputLayer->getNeuronValue(i+1) - label[i]);
    }
    for(int l = getNumberOfLayers() - 2; l >= 0; l--){
        Layer* thisLayer = getLayer(l);
        Layer* nextLayer = getLayer(l+1);
        thisLayer->computeNeuronDeltas(nextLayer);
        thisLayer->addToThetaDeltaAccumulator(nextLayer);
    }
}

void Network::resetDeltaAccumulators(){
    for(int l = 0; l < getNumberOfLayers() - 2; l++){
        Layer* layer = getLayer(l);
        layer->initializeThetaDeltaAccumulator(layer->getSize(), getLayer(l+1)->getSize());
    }
}

void Network::train(double** X, double** y, int numOfTrainingExamples, double lambda, int iterations){
    for(int iter = 0; iter < iterations; iter++){
        resetDeltaAccumulators();
        for(int i = 0; i < numOfTrainingExamples; i++){
            setInputValues(X[i]);
            feedForward();
            backPropagate(y[i]);
        }
        for(int l = 0; l < getNumberOfLayers() - 2; l++){
            Layer* thisLayer = getLayer(l);
            Layer* nextLayer = getLayer(l+1);
            for(int i = 0; i < nextLayer->getSize(); i++){
                for(int j = 0; j < thisLayer->getSize()+1; j++){
                    double thetaDelta;
                    double thetaValue = thisLayer->getThetaValue(i, j);
                    if(j == 0){
                        thetaDelta = thisLayer->getThetaDeltaAccumulatorValue(i,j) / numOfTrainingExamples;
                    } else {
                        thetaDelta = thisLayer->getThetaDeltaAccumulatorValue(i,j) / numOfTrainingExamples + lambda * thetaValue;
                    }
                    //cout << "Layer " << l << " ThetaDelta[" << i << "][" << j << "]: " << thetaDelta << endl;
                    thisLayer->setTheta(i, j, thetaValue - lambda * thetaDelta); // Update theta value through gradient descent
                }
            }
        }

    }
}

double* Network::getOutput(double input[]){
    setInputValues(input);
    feedForward();
    return getLayer(getNumberOfLayers()-1)->getNeuronValues(false);
}

int main(int argc, const char* argv[]){
    double** X = new double*[4];
    double ** y = new double*[4];
    for (int i = 0; i < 4; i++){
        X[i] = new double[2];
        y[i] = new double[1];
        y[i][0] = i == 3 ? 1 : 0;
    }
    X[0][0] = 0; X[0][1] = 0; X[1][0] = 0; X[1][1] = 1; X[2][0] = 1; X[2][1] = 0; X[3][0] = 1; X[3][1] = 1;

    Network *net = new Network(2, 1, 2, 1); // Initialize neural net
    net->train(X, y, 4, 0.01, 100000);

    for(int i = 0; i < 4; i++){
        double input[2] = {X[i][0], X[i][1]};
        double output = *(net->getOutput(input));
        double result = output >= 0.5 ? 1 : 0;
        cout << "Input: [" << input[0] << ", " << input[1] << "]";
        cout << " Output: " << output << endl;
    }

    // for(int l = 0; l < net->getNumberOfLayers(); l++){
    //     Layer* layer = net->getLayer(l);
    //     cout << "Layer " << l << endl;
    //     double* neuronVals = layer->getNeuronValues(false);
    //     for(int i = 0; i < layer->getSize(); i++){
    //         cout << "Neuron " << i << ": " << neuronVals[i] << endl;
    //     }
    //     if (l == net->getNumberOfLayers() - 1){
    //         break;
    //     }
    //     cout << "Theta " << l << endl;
    //     double** thetaVals = layer->getThetas();
    //     for(int i = 0; i < net->getLayer(l+1)->getSize(); i++){
    //         for(int j = 0; j < layer->getSize() + 1; j++){
    //             cout << setw(10) << thetaVals[i][j];
    //         }
    //         cout << endl;
    //     }

    // }
    return 0;
}

