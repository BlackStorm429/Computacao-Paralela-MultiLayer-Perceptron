/*
MLP Library - Version 2.0 - May 2025 (Modified)

Copyright (c) 2005 Sylvain BARTHELEMY
Modified to fix MPI implementation issues

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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>


#include "mlp.h"

void InitializeRandoms()
{
  srand(time(0));
}

int RandomEqualINT(int Low, int High)
{
  return rand() % (High-Low+1) + Low;
}

double RandomEqualREAL(double Low, double High)
{
  return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

// Construtor de cópia: aloca novas estruturas e copia valores
MultiLayerPerceptron::MultiLayerPerceptron(const MultiLayerPerceptron& other)
  : nNumLayers(other.nNumLayers)
{
    // 1) Duplica o array de camadas
    pLayers = new Layer[nNumLayers];

    // 2) Copia os tamanhos e aloca neurônios
    for (int i = 0; i < nNumLayers; ++i) {
        pLayers[i].nNumNeurons = other.pLayers[i].nNumNeurons;
        pLayers[i].pNeurons    = new Neuron[ pLayers[i].nNumNeurons ];

        // Se não for camada de entrada, aloca w, dw, wsave
        if (i > 0) {
            int prevN = other.pLayers[i-1].nNumNeurons;
            for (int j = 0; j < pLayers[i].nNumNeurons; ++j) {
                pLayers[i].pNeurons[j].w     = new double[prevN];
                pLayers[i].pNeurons[j].dw    = new double[prevN];
                pLayers[i].pNeurons[j].wsave = new double[prevN];
                // 3) Copia pesos armazenados no vetor weights
                for (int k = 0; k < prevN; ++k) {
                    double v = other.pLayers[i].pNeurons[j].w[k];
                    pLayers[i].pNeurons[j].w[k]     = v;
                    pLayers[i].pNeurons[j].dw[k]    = other.pLayers[i].pNeurons[j].dw[k];
                    pLayers[i].pNeurons[j].wsave[k] = other.pLayers[i].pNeurons[j].wsave[k];
                }
            }
        }
        else {
            // entrada não tem arrays de pesos
            for (int j = 0; j < pLayers[i].nNumNeurons; ++j)
                pLayers[i].pNeurons[j].w = pLayers[i].pNeurons[j].dw = pLayers[i].pNeurons[j].wsave = nullptr;
        }
    }

    // 4) Copia demais campos escalares
    dMSE = other.dMSE;
    dMAE = other.dMAE;
    dEta = other.dEta;
    dAlpha = other.dAlpha;
    dGain = other.dGain;
    dAvgTestError = other.dAvgTestError;

    // 5) Copia vetores de compatibilidade (weights, biases)
    weights = other.weights;
    biases  = other.biases;
}

MultiLayerPerceptron& MultiLayerPerceptron::operator=(const MultiLayerPerceptron& other)
{
    if (this == &other) return *this;

    // 1) Liberar recursos atuais (mesmo que destrutor faria)
    this->~MultiLayerPerceptron();

    // 2) Reconstruir via cópia: usar placement new
    new (this) MultiLayerPerceptron(other);
    return *this;
}

MultiLayerPerceptron::MultiLayerPerceptron(int nl, int npl[]) :
  nNumLayers(0),
  pLayers(0),
  dMSE(0.0),
  dMAE(0.0),
  dEta(0.25),
  dAlpha(0.9),
  dGain(1.0),
  dAvgTestError(0.0)
{
  int i,j;

  /* --- criação das camadas */
  nNumLayers = nl;
  pLayers    = new Layer[nl];

  /* --- inicialize weights and biases vectors for compatibility with GetLayerSize, etc. */
  weights.resize(nl);
  biases.resize(nl);

  /* --- init das camadas */
  for ( i = 0; i < nl; i++ )
    {
      /* --- criação dos neurones */
      pLayers[i].nNumNeurons = npl[i];
      pLayers[i].pNeurons    = new Neuron[ npl[i] ];

      /* --- initialize weights and biases structures */
      if (i > 0) {
        weights[i].resize(npl[i]);
        biases[i].resize(npl[i]);
        for (int n = 0; n < npl[i]; n++) {
          weights[i][n].resize(npl[i-1], 0.0);
        }
      }

      /* --- init dos neurones */
      for( j = 0; j < npl[i]; j++ )
	{
	  pLayers[i].pNeurons[j].x  = 1.0;
	  pLayers[i].pNeurons[j].e  = 0.0;
	  if(i>0)
	    {
	      pLayers[i].pNeurons[j].w     = new double[ npl[i-1] ];
	      pLayers[i].pNeurons[j].dw    = new double[ npl[i-1] ];
	      pLayers[i].pNeurons[j].wsave = new double[ npl[i-1] ];
	    }
	  else
	    {
	      pLayers[i].pNeurons[j].w     = NULL;
	      pLayers[i].pNeurons[j].dw    = NULL;
	      pLayers[i].pNeurons[j].wsave = NULL;
	    }
	}
    }
}

MultiLayerPerceptron::~MultiLayerPerceptron()
{
  int i,j;
  for( i = 0; i < nNumLayers; i++ )
    {
      if ( pLayers[i].pNeurons )
	{
	  for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	    {
	      if ( pLayers[i].pNeurons[j].w )
		delete[] pLayers[i].pNeurons[j].w;
	      if ( pLayers[i].pNeurons[j].dw )
		delete[] pLayers[i].pNeurons[j].dw;
	      if ( pLayers[i].pNeurons[j].wsave )
		delete[] pLayers[i].pNeurons[j].wsave;
	    }
	}
      delete[] pLayers[i].pNeurons;
    }
  delete[] pLayers;
}

void MultiLayerPerceptron::RandomWeights()
{
  int i,j,k;
  for( i = 1; i < nNumLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	    {
	      double randValue = RandomEqualREAL(-0.5, 0.5);
	      pLayers[i].pNeurons[j].w [k]    = randValue;
	      pLayers[i].pNeurons[j].dw[k]    = 0.0;
	      pLayers[i].pNeurons[j].wsave[k] = 0.0;
          
          // Update weights vector for compatibility with GetWeight
          weights[i][j][k] = randValue;
	    }
      // Initialize bias in the biases vector
      biases[i][j] = 0.0;
	}
    }
}

void MultiLayerPerceptron::SetInputSignal(double* input)
{
  int i;
  for ( i = 0; i < pLayers[0].nNumNeurons; i++ )
    {
      pLayers[0].pNeurons[i].x = input[i];
    }
}

void MultiLayerPerceptron::GetOutputSignal(double* output)
{
  int i;
  for ( i = 0; i < pLayers[nNumLayers-1].nNumNeurons; i++ )
    {
      output[i] = pLayers[nNumLayers-1].pNeurons[i].x;
    }
}

void MultiLayerPerceptron::SaveWeights()
{
  int i,j,k;
  for( i = 1; i < nNumLayers; i++ )
    for( j = 0; j < pLayers[i].nNumNeurons; j++ )
      for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	pLayers[i].pNeurons[j].wsave[k] = pLayers[i].pNeurons[j].w[k];
}

void MultiLayerPerceptron::RestoreWeights()
{
  int i,j,k;
  for( i = 1; i < nNumLayers; i++ )
    for( j = 0; j < pLayers[i].nNumNeurons; j++ )
      for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ ) {
	pLayers[i].pNeurons[j].w[k] = pLayers[i].pNeurons[j].wsave[k];
        weights[i][j][k] = pLayers[i].pNeurons[j].wsave[k];
      }
}

/***************************************************************************/
/* calculate and feedforward outputs from the first layer to the last      */
void MultiLayerPerceptron::PropagateSignal()
{
  int i,j,k;

  /* --- la boucle commence avec la seconde couche */
  for( i = 1; i < nNumLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  /* --- calcul de la somme pondérée en entrée */
	  double sum = 0.0;
	  for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	    {
	      double out = pLayers[i-1].pNeurons[k].x;
	      double w   = pLayers[i  ].pNeurons[j].w[k];
	      sum += w * out;
	    }
	  /* --- application de la fonction d'activation (sigmoid) */
	  pLayers[i].pNeurons[j].x = 1.0 / (1.0 + exp(-dGain * sum));
	}
    }
}

void MultiLayerPerceptron::ComputeOutputError(double* target)
{
  int  i;
  dMSE = 0.0;
  dMAE = 0.0;
  for( i = 0; i < pLayers[nNumLayers-1].nNumNeurons; i++)
    {
      double x = pLayers[nNumLayers-1].pNeurons[i].x;
      double d = target[i] - x;
      pLayers[nNumLayers-1].pNeurons[i].e = dGain * x * (1.0 - x) * d;
      dMSE += (d * d);
      dMAE += fabs(d);
    }
  /* --- erreur quadratique moyenne */
  dMSE /= (double)pLayers[nNumLayers-1].nNumNeurons;
  /* --- erreur absolue moyenne */
  dMAE /= (double)pLayers[nNumLayers-1].nNumNeurons;
}

/***************************************************************************/
/* backpropagate error from the output layer through to the first layer    */

void MultiLayerPerceptron::BackPropagateError()
{
  int i,j,k;
  /* --- la boucle commence à l'avant dernière couche */
  for( i = (nNumLayers-2); i >= 0; i-- )
    {
      /* --- couche inférieure */
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  double x = pLayers[i].pNeurons[j].x;
	  double E = 0.0;
	  /* --- couche supérieure */
	  for ( k = 0; k < pLayers[i+1].nNumNeurons; k++ )
	    {
	      E += pLayers[i+1].pNeurons[k].w[j] * pLayers[i+1].pNeurons[k].e;
	    }
	  pLayers[i].pNeurons[j].e = dGain * x * (1.0 - x) * E;
	}
    }
}

/***************************************************************************/
/* update weights for all of the neurons from the first to the last layer  */

void MultiLayerPerceptron::AdjustWeights()
{
  int i,j,k;
  /* --- la boucle commence avec la seconde couche */
  for( i = 1; i < nNumLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	    {
	      double x  = pLayers[i-1].pNeurons[k].x;
	      double e  = pLayers[i  ].pNeurons[j].e;
	      double dw = pLayers[i  ].pNeurons[j].dw[k];
	      pLayers[i].pNeurons[j].w [k] += dEta * x * e + dAlpha * dw;
	      pLayers[i].pNeurons[j].dw[k]  = dEta * x * e;
          
          // Update weights vector for compatibility
          weights[i][j][k] = pLayers[i].pNeurons[j].w[k];
	    }
	}
    }
}

void MultiLayerPerceptron::Simulate(double* input, double* output, double* target, bool training)
{

  if(!input)  return;

  /* --- on fait passer le signal dans le réseau */
  SetInputSignal(input);
  PropagateSignal();
  if(output) GetOutputSignal(output);

  /* --- calcul de l'erreur en sortie par rapport à la cible */
  /*     ce calcul sert de base pour la rétropropagation     */
  if (target) {
    ComputeOutputError(target);

    /* --- si c'est un apprentissage, on fait une rétropropagation de l'erreur */
    if (training)
      {
        BackPropagateError();
        AdjustWeights();
      }
  }
}

bool read_number(FILE* fp, double* number)
{
  char szWord[256];
  int i = 0;
  int b;

  *number = 0.0;

  szWord[0] = '\0';
  while ( ((b=fgetc(fp))!=EOF) && (i<255) )
    {
      if( (b=='.') ||
	  (b=='0') ||
	  (b=='1') ||
	  (b=='2') ||
	  (b=='3') ||
	  (b=='4') ||
	  (b=='5') ||
	  (b=='6') ||
	  (b=='7') ||
	  (b=='8') ||
	  (b=='9') )
	{
	  szWord[i++] = (char)b;
	}
      else
	if(i>0) break;
    }
  szWord[i] = '\0';

  if(i==0) return false;

  *number = atof(szWord);

  return true;
}

int MultiLayerPerceptron::Train(const char* fname)
{
  int count = 0;
  int nbi   = 0;
  int nbt   = 0;
  double* input  = NULL;
  double* output = NULL;
  double* target = NULL;
  FILE*   fp = NULL;

  fp = fopen(fname,"r");
  if(!fp) return 0;

  input  = new double[pLayers[0].nNumNeurons];
  output = new double[pLayers[nNumLayers-1].nNumNeurons];
  target = new double[pLayers[nNumLayers-1].nNumNeurons];

  if(!input) return 0;
  if(!output) return 0;
  if(!target) return 0;


  while( !feof(fp) )
    {
      double dNumber;
      if( read_number(fp,&dNumber) )
	{
	  /* --- on le transforme en input/target */
	  if( nbi < pLayers[0].nNumNeurons )
	    input[nbi++] = dNumber;
	  else if( nbt < pLayers[nNumLayers-1].nNumNeurons )
	    target[nbt++] = dNumber;

	  /* --- on fait un apprentisage du réseau  avec cette ligne*/
	  if( (nbi == pLayers[0].nNumNeurons) &&
	      (nbt == pLayers[nNumLayers-1].nNumNeurons) )
	    {
	      Simulate(input, output, target, true);
	      nbi = 0;
	      nbt = 0;
	      count++;
	    }
	}
      else
	{
	  break;
	}
    }

  if(fp) fclose(fp);

  if(input)  delete[] input;
  if(output) delete[] output;
  if(target) delete[] target;

  return count;
}

int MultiLayerPerceptron::Test(const char* fname)
{
  int count = 0;
  int nbi   = 0;
  int nbt   = 0;
  double* input  = NULL;
  double* output = NULL;
  double* target = NULL;
  FILE*   fp = NULL;

  fp = fopen(fname,"r");
  if(!fp) return 0;

  input  = new double[pLayers[0].nNumNeurons];
  output = new double[pLayers[nNumLayers-1].nNumNeurons];
  target = new double[pLayers[nNumLayers-1].nNumNeurons];

  if(!input) return 0;
  if(!output) return 0;
  if(!target) return 0;

  dAvgTestError = 0.0;

  while( !feof(fp) )
    {
      double dNumber;
      if( read_number(fp,&dNumber) )
	{
	  /* --- on le transforme en input/target */
	  if( nbi < pLayers[0].nNumNeurons )
	    input[nbi++] = dNumber;
	  else if( nbt < pLayers[nNumLayers-1].nNumNeurons )
	    target[nbt++] = dNumber;

	  /* --- on fait un apprentisage du réseau  avec cette ligne*/
	  if( (nbi == pLayers[0].nNumNeurons) &&
	      (nbt == pLayers[nNumLayers-1].nNumNeurons) )
	    {
	      Simulate(input, output, target, false);
	      dAvgTestError += dMAE;
	      nbi = 0;
	      nbt = 0;
	      count++;
	    }
	}
      else
	{
	  break;
	}
    }

  dAvgTestError /= count;

  if(fp) fclose(fp);

  if(input)  delete[] input;
  if(output) delete[] output;
  if(target) delete[] target;

  return count;
}

int MultiLayerPerceptron::Evaluate()
{
  int count = 0;
  return count;
}

void MultiLayerPerceptron::Run(const char* fname, const int& maxiter)
{
  int    countTrain = 0;
  int    countLines = 0;
  bool   Stop = false;
  bool   firstIter = true;
  double dMinTestError = 0.0;

  /* --- init du générateur de nombres aléatoires  */
  /* --- et génération des pondérations aléatoires */
  InitializeRandoms();
  RandomWeights();

  /* --- on lance l'apprentissage avec tests */
  do {

    countLines += Train(fname);
    Test(fname);
    countTrain++;

    if(firstIter)
      {
	dMinTestError = dAvgTestError;
	firstIter = false;
      }

    //printf( "%i \t TestError: %f", countTrain, dAvgTestError);

    if ( dAvgTestError < dMinTestError)
      {
	//printf(" -> saving weights\n");
	dMinTestError = dAvgTestError;
	SaveWeights();
      }
    else if (dAvgTestError > 1.2 * dMinTestError)
      {
	//printf(" -> stopping training and restoring weights\n");
	Stop = true;
	RestoreWeights();
      }
    else
      {
	//printf(" -> ok\n");
      }

  } while ( (!Stop) && (countTrain<maxiter) );

  //printf("bye\n");

}
/// Retorna número de camadas (input + hidden + output)
int MultiLayerPerceptron::GetLayerCount() const {
  return nNumLayers;
}

// Soma pesos e biases de outra rede a esta
void MultiLayerPerceptron::AddWeightsFrom(const MultiLayerPerceptron& other) {
  // Para cada camada (exceto a de entrada):
  for (int l = 1; l < nNumLayers; ++l) {
      int N = pLayers[l].nNumNeurons;
      int M = pLayers[l-1].nNumNeurons;
      for (int i = 0; i < N; ++i) {
          Neuron& mine   = pLayers[l].pNeurons[i];
          const Neuron& th = other.pLayers[l].pNeurons[i];
          for (int j = 0; j < M; ++j) {
              mine.w[j] += th.w[j];
              weights[l][i][j] += other.weights[l][i][j];
          }
      }
  }
}

void MultiLayerPerceptron::cloneWeightsFrom(const MultiLayerPerceptron& other) {
  // Para cada camada (exceto a de entrada):
  for (int l = 1; l < nNumLayers; ++l) {
      int N = pLayers[l].nNumNeurons;
      int M = pLayers[l-1].nNumNeurons;
      for (int i = 0; i < N; ++i) {
          Neuron& mine   = pLayers[l].pNeurons[i];
          const Neuron& th = other.pLayers[l].pNeurons[i];
          for (int j = 0; j < M; ++j) {
              mine.w[j] = th.w[j];
              weights[l][i][j] = other.weights[l][i][j];
          }
      }
  }
}

// Escala todos os pesos e biases por um fator
void MultiLayerPerceptron::ScaleWeights(double factor) {
  // Multiplica todos os pesos por 'factor'
  for (int l = 1; l < nNumLayers; ++l) {
      int N = pLayers[l].nNumNeurons;
      int M = pLayers[l-1].nNumNeurons;
      for (int i = 0; i < N; ++i) {
          Neuron& neuron = pLayers[l].pNeurons[i];
          for (int j = 0; j < M; ++j) {
              neuron.w[j] *= factor;
              weights[l][i][j] *= factor;
          }
      }
  }
}

int MultiLayerPerceptron::GetLayerSize(int layer) const {
  if (layer < 0 || layer >= nNumLayers) return 0;
  return pLayers[layer].nNumNeurons;
}

double MultiLayerPerceptron::GetWeight(int layer, int neuron, int input) const {
  if (layer <= 0 || layer >= nNumLayers) return 0.0;
  if (neuron < 0 || neuron >= pLayers[layer].nNumNeurons) return 0.0;
  if (input < 0 || input >= pLayers[layer-1].nNumNeurons) return 0.0;
  return pLayers[layer].pNeurons[neuron].w[input];
}

double MultiLayerPerceptron::GetBias(int layer, int neuron) const {
  // In this implementation, biases are represented implicitly
  // Return 0 as a default value since we don't have explicit bias values
  return 0.0;
}

void MultiLayerPerceptron::SetWeight(int layer, int neuron, int input, double value) {
  if (layer <= 0 || layer >= nNumLayers) return;
  if (neuron < 0 || neuron >= pLayers[layer].nNumNeurons) return;
  if (input < 0 || input >= pLayers[layer-1].nNumNeurons) return;
  pLayers[layer].pNeurons[neuron].w[input] = value;
  weights[layer][neuron][input] = value;
}

void MultiLayerPerceptron::SetBias(int layer, int neuron, double value) {
  // In this implementation, biases are not explicitly separated
  // They could be implemented as an extra input with fixed value of 1.0
  biases[layer][neuron] = value;
}

int MultiLayerPerceptron::Predict(const double* input) const {
  // Create temporary arrays to store activations
  double* activations_in = new double[pLayers[0].nNumNeurons];
  double* activations_out = new double[pLayers[nNumLayers-1].nNumNeurons];
  
  // Copy input values
  for (int i = 0; i < pLayers[0].nNumNeurons; i++) {
    activations_in[i] = input[i];
  }
  
  // Create non-const version of this to use Simulate
  MultiLayerPerceptron* non_const_this = const_cast<MultiLayerPerceptron*>(this);
  
  // Forward pass through the network
  non_const_this->Simulate(activations_in, activations_out, nullptr, false);
  
  // Find the index of the maximum activation in the output layer
  int max_idx = 0;
  double max_val = activations_out[0];
  for (int i = 1; i < pLayers[nNumLayers-1].nNumNeurons; i++) {
    if (activations_out[i] > max_val) {
      max_val = activations_out[i];
      max_idx = i;
    }
  }
  
  // Clean up
  delete[] activations_in;
  delete[] activations_out;
  
  return max_idx;
}