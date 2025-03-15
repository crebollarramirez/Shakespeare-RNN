# Shakespearean Text Generation with RNN and LSTM

## Table of Contents
- [Introduction](#introduction)
  - [Detailed Report](#detailed-report)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Models](#models)
  - [Baseline Models](#baseline-models)
  - [Training Procedure](#training-procedure)
  - [Text Generation](#text-generation)
- [Results](#results)
  - [Performance Metrics](#performance-metrics)
- [Discussion](#discussion)
- [Contributions](#contributions)
- [References](#references)

## Introduction
This project leverages Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) architectures to generate text in the style of William Shakespeare, using the TinyShakespeare dataset, which consists of 40,000 lines from his plays. We aim to replicate Shakespeare's unique literary style and delve into the intricacies of modeling temporal dependencies in text. By exploring various neural network architectures and training strategies, particularly focusing on the effectiveness of teacher forcing in accelerating model convergence, the study seeks to understand how different network configurations influence the quality of text generation. This approach not only tests the capability of RNNs and LSTMs in handling long-range dependencies but also provides insights into optimizing machine learning models for creative text generation.

### Detailed Report
You can access the comprehensive analysis by clicking here: [Detailed Report](./detailed-report.pdf)

## Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)


## Dataset
The project utilizes the [TinyShakespeare](./data/tiny_shakespeare.txt) dataset, which consists of 40,000 lines from Shakespeare's plays. This dataset challenges the model with character-level inputs and outputs to replicate the literary style.

## Models

### Baseline Models
- **RNN Model**: Single hidden layer with 150 neurons, embeddings of size 100, tanh activation, and a fully connected output layer.
- **LSTM Model**: Similar structure as the RNN but incorporates LSTM cells to improve retention of long-range dependencies.

### Training Procedure
- **Teacher Forcing**: Used to speed up training by using actual output tokens from the dataset as input for the next training step, rather than the model's predictions.
- **Without Teacher Forcing**: Trains the model by using its own predictions as inputs for subsequent steps to enhance its generative capabilities.

### Text Generation
Text generation is performed by sampling from the output distribution of the model, adjusted by a temperature parameter that affects randomness and diversity in the generated text.

### Sample Output from LSTM with 128 Sequence Length and 300 Neurons: High-Quality Text Generation
```
Macbeth
 by William Shakespeare
 Edited by Barbara A. Mowat and Paul Werstines to speak
As the means of the sacriving spirits of him
Assurance to assurance that the fice
To part the welcome of the shadow of the same,
Have had the sun for what you must feel the world
Than the sister than the worthy service,
And court that the while the princes that we well
Are all the present proclamation
Of the clamour of the ground of the duke.

GREMIO:
The matter have made me shall we die that sighs to him;
And then may be a man's part thou to all, and so still.

GRUMIO:
How camest thou holier that I am possible?

CAPULET:
And then I pardon it: he shall speak the house
As the sicks interruption of the north.

DUCHESS OF YORK:
Ay, and therefore were I so no man to the gods.

KING RICHARD II:
My lord forth thy command still to my heart.

ROMEO:
As the argy of the way that I have had done
That the parliam lies, the gods will be not breath
That makes her here they shall answer the prince.

BUCKINGHAM:
My lord, I will do him stand a pair and like a thoughts
Of my heart am not comp
```


## Results
The LSTM models generally outperformed RNNs, especially with higher sequence lengths and when using teacher forcing. Detailed performance metrics are discussed with respect to different model configurations and training approaches.

### Performance Metrics
| Model                         | Sequence Length | Loss   |
|-------------------------------|-----------------|--------|
| RNN (Baseline)                | 16              | 1.5309 |
| RNN                           | 128             | 1.5473 |
| RNN                           | 512             | 1.5267 |
| LSTM (Baseline)               | 16              | 1.4054 |
| LSTM                          | 128             | 1.3932 |
| **LSTM (Larger Hidden)**          | **128**             | **1.3237** |
| LSTM                          | 512             | 1.3773 |
| LSTM (No Teacher Forcing)     | 4               | 3.1663 |
| LSTM (No Teacher Forcing)     | 8               | 3.1266 |
| LSTM (No Teacher Forcing)     | 16              | 3.3049 |
| LSTM (No Teacher Forcing)     | 32              | 3.3174 |
| RNN (2 Hidden Layers)         | 16              | 1.5242 |
| LSTM (2 Hidden Layers)        | 16              | 1.3857 |


## Discussion
The effectiveness of LSTMs over RNNs in handling long-range dependencies is evident, with significant differences in performance metrics under various experimental conditions. The impact of temperature on the coherence and creativity of the generated text is also analyzed.

## Contributions
- [Anthony Tong](https://github.com/atong28)
- [Brandon Park](https://github.com/brandonmpark)
- [Chi Zhang](https://github.com/Ayaaa99)
- [Christopher Rebollar-Ramirez](https://github.com/crebollarramirez)

## References
1. Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. "Scheduled sampling for sequence prediction with recurrent neural networks," 2015.
2. Jeffrey L. Elman. "Finding structure in time." Cognitive Science, 14(2):179–211, 1990.
3. Felix Gers, Jürgen Schmidhuber, and Fred Cummins. "Learning to forget: Continual prediction with LSTM." Neural Computation, 12:2451–2471, October 2000.
4. Sepp Hochreiter. "The vanishing gradient problem during learning recurrent neural nets and problem solutions." International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 06(02):107–116, 1998.
5. Andrej Karpathy. "char-rnn." https://github.com/karpathy/char-rnn, 2015.
6. D. E. Rumelhart, G. E. Hinton, and R. J. Williams. "Learning internal representations by error propagation." MIT Press, Cambridge, MA, USA, 1986, pp. 318–362.
7. Ronald J. Williams and David Zipser. "A learning algorithm for continually running fully recurrent neural networks." Neural Computation, 1:270–280, 1989.