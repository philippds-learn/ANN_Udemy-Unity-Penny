using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    ANN ann;
    double sumSquareError = 0;

    // Start is called before the first frame update
    void Start()
    {
        this.ann = new ANN(2, 1, 1, 2, 0.8);

        List<double> result;

        // 1000 epochs
        for(int i = 0; i < 10000; i++)
        {
            this.sumSquareError = 0;
            result = Train(1, 1, 0, true);
            this.sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(1, 0, 1, true);
            this.sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(0, 1, 1, true);
            this.sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(0, 0, 0, true);
            this.sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
        }
        Debug.Log("SSE: " + (float)sumSquareError);

        result = Train(1, 1, 0, false);
        Debug.Log(" 1 1 (0)" + result[0]);
        result = Train(1, 0, 1, false);
        Debug.Log(" 1 0 (1)" + result[0]);
        result = Train(0, 1, 1, false);
        Debug.Log(" 0 1 (1)" + result[0]);
        result = Train(0, 0, 0, false);
        Debug.Log(" 0 0 (0)" + result[0]);
    }

    List<double> Train(double i1, double i2, double o, bool updateWeights)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return this.ann.Go(inputs, outputs, updateWeights);
    }
}
