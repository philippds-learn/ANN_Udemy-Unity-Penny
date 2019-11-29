using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    public int numInputs;
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    public Neuron(int nInputs)
    {
        this.bias = Random.Range(-0.1f, 0.1f);
        this.numInputs = nInputs;
        for(int i = 0; i < nInputs; i++)
        {
            this.weights.Add(Random.Range(-0.1f, 0.1f));
        }
    }
}
