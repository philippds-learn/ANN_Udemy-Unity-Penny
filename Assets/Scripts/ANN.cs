using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
{
    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden; // number of neurons per hidden layer
    public double alpha; // determines how fast the network learns
    List<Layer> layers = new List<Layer>();

    public ANN(int numberOfInputs, int numberOfOutputs, int numberOfHiddenLayers, int numberOfNeuronsInHiddenLayers, double alpha)
    {
        this.numInputs = numberOfInputs;
        this.numOutputs = numberOfOutputs;
        this.numHidden = numberOfHiddenLayers;
        this.numNPerHidden = numberOfNeuronsInHiddenLayers;
        this.alpha = alpha;

        if(this.numHidden > 0)
        {
            // adding input layer
            this.layers.Add(new Layer(this.numNPerHidden, this.numInputs));

            for(int i = 0; i < this.numHidden - 1; i++)
            {
                // adding hidden layer
                this.layers.Add(new Layer(this.numNPerHidden, this.numNPerHidden));
            }

            // adding output layer
            this.layers.Add(new Layer(this.numOutputs, this.numNPerHidden));
        }
        else
        {
            // adding output layer
            this.layers.Add(new Layer(this.numOutputs, this.numInputs));
        }
    }

    public List<double> Go(List<double> inputValues, List<double> desiredOutput, bool updateWeights)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if(inputValues.Count != this.numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + this.numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);
        // loop through layers
        for(int i = 0; i < this.numHidden + 1; i++)
        {
            if(i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for(int j = 0; j < this.layers[i].numNeurons; j++)
            {
                double N = 0;
                this.layers[i].neurons[j].inputs.Clear();

                for(int k = 0; k < this.layers[i].neurons[j].numInputs; k++)
                {
                    this.layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k]; // dotproduct
                }

                N -= this.layers[i].neurons[j].bias;

                if(i == this.numHidden)
                {
                    this.layers[i].neurons[j].output = ActivationFunctionOutputLayer(N);
                }
                else
                {
                    this.layers[i].neurons[j].output = ActivationFunction(N);
                }
                
                outputs.Add(this.layers[i].neurons[j].output);
            }
        }

        if(updateWeights == true)
        {
            UpdateWeights(outputs, desiredOutput);
        }

        return outputs;
    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        // looping through layers backwards
        for(int i = this.numHidden; i >= 0; i--)
        {
            // looping through neurons
            for(int j = 0; j < this.layers[i].numNeurons; j++)
            {
                // if output layer
                if(i == this.numHidden)
                {
                    error = desiredOutput[j] - outputs[j];
                    this.layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                    // errorGradient calculated with Delta Rule: en.wikipedia.org/wiki/Delta_rule
                }
                else
                {
                    this.layers[i].neurons[j].errorGradient = this.layers[i].neurons[j].output * (1 - this.layers[i].neurons[j].output);
                    double errorGradSum = 0;
                    for(int p = 0; p < this.layers[i + 1].numNeurons; p++)
                    {
                        errorGradSum += this.layers[i + 1].neurons[p].errorGradient * this.layers[i + 1].neurons[p].weights[j];
                    }
                    this.layers[i].neurons[j].errorGradient *= errorGradSum;
                }

                // looping through inputs
                for(int k = 0; k < this.layers[i].neurons[j].numInputs; k++)
                {
                    // if in hidden layer
                    if(i == this.numHidden)
                    {
                        error = desiredOutput[j] - outputs[j];
                        this.layers[i].neurons[j].weights[k] += this.alpha * this.layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        this.layers[i].neurons[j].weights[k] += this.alpha * this.layers[i].neurons[j].inputs[k] * this.layers[i].neurons[j].errorGradient;
                    }
                }
                this.layers[i].neurons[j].bias += this.alpha * -1 * this.layers[i].neurons[j].errorGradient;
            }
        }
    }

    // for full list of activation functions
    // see en.wikipedia.org/wiki/Activation_function
    double ActivationFunction(double value)
    {
        return Sigmoid(value);
    }

    double ActivationFunctionOutputLayer(double value)
    {
        return Sigmoid(value);
    }

    // binary step function
    double Step(double value)
    {
        if (value < 0) return 0;
        else return 1;
    }

    // sigmopid activation function / logistic softstep
    double Sigmoid(double value)
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
    }

    // TanH activation function
    double TanH(double value)
    {
        return 2 * (Sigmoid(2 * value) - 1);
    }

    // ReLu (Rectified Linear Unit)
    double ReLu(double value)
    {
        if(value > 0)
        {
            return value;
        }
        else
        {
            return 0;
        }
    }

    // LeakyReLu (Rectified Linear Unit)
    double LeakyReLu(double value)
    {
        if(value < 0)
        {
            return 0.01 * value;
        }
        else
        {
            return value;
        }
    }

    //Sinusoid, ArcTan and SoftSign.

    // Sinusoid
    double Sinusoid(double value)
    {
        return Mathf.Sin((float)value);
    }

    // ArcTan
    double ArcTan(double value)
    {
        return Mathf.Atan((float)value);
    }

    // SoftSign
    double SoftSign(double value)
    {
        return value / (1 + Mathf.Abs((float)value));
    }
}
