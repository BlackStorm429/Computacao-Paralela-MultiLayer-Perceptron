/*
MLP Library - Version 2.0 - August 2005

Copyright (c) 2005 Sylvain BARTHELEMY

Contact: sylbarth@gmail.com, www.sylbarth.com

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef _MLP_H_
#define _MLP_H_

struct Neuron {
  double  x;     /* saída */
  double  e;     /* erro */
  double* w;     /* pesos */
  double* dw;    /* último peso para os momentum */
  double* wsave; /* pesos salvos */
};

struct Layer {
  int     nNumNeurons;
  Neuron* pNeurons;
};

class MultiLayerPerceptron {
  int    nNumLayers;
  Layer* pLayers;

  double dMSE;
  double dMAE;

  void RandomWeights();

  void SetInputSignal(double* input);
  void GetOutputSignal(double* output);

  void SaveWeights();
  void RestoreWeights();

  void PropagateSignal();
  void ComputeOutputError(double* target);
  void BackPropagateError();
  void AdjustWeights();

  std::vector<std::vector<std::vector<double>>> weights;
  std::vector<std::vector<double>> biases;

public:
  double dEta;
  double dAlpha;
  double dGain;
  double dAvgTestError;

  MultiLayerPerceptron(int nl, int npl[]);
  ~MultiLayerPerceptron();

  int Train(const char* fnames);
  int Test(const char* fname);
  int Evaluate();

  void Run(const char* fname, const int& maxiter);
  void Simulate(double* input, double* output, double* target, bool training);

  int GetLayerCount() const;
  void AddWeightsFrom(const MultiLayerPerceptron& other);
  void ScaleWeights(double factor);
  int GetLayerSize(int layer) const;
  double GetWeight(int layer, int neuron, int input) const;
  double GetBias(int layer, int neuron) const;
  void SetWeight(int layer, int neuron, int input, double value);
  void SetBias(int layer, int neuron, double value);
  int Predict(const double* input) const;

};

#endif
