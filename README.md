# ONLINE LEARNING PERCEPTRON

Matlab implementation of online perceptron algorithm and its variants.

Let’s start with a brief overview of perceptron:

### McCulloch-Pitts Neuron
The birth of artificial neural nets started with the 1943 paper “a Logical Calculus of the Ideas Immanent in Nervous Activity”. Two researchers, McCulloch a neurologist, Pitts a logician, joined forces to sketch out the first artificial neurons.
McCulloch reasoned that nervous activity had an ‘all-or-nothing’ activity: A neuron would fire (output “1”) once it’s activation threshold was reached, or it wouldn’t fire (output “0”).
Pitts saw and understood the potential to capture propositional logic using such neurological principles.

## Perceptron
Invented in 1957 by cognitive psychologist Frank Rosenblatt, the perceptron algorithm was the first artificial neural net implemented in hardware. Visit Wikipedia for more information: https://en.wikipedia.org/wiki/Perceptron.

In 1960 researchers at Cornell Aeronautical Laboratory, with funding from the US Office of Naval Research, randomly hooked 400 photocells to a perceptron and the “Mark 1 perceptron” was born. It was capable of basic image recognition.

#### Weights
A Perceptron works by assigning weights to incoming connections. With the McCulloch-Pitts Neuron we took the sum of the values from the incoming connections, then looked if it was over or below a certain threshold. With the Perceptron we instead take the dotproduct. We multiply each incoming value with a weight and take the sum: (value1 * weight1) + (value2 * weight2), etc.

#### Learning
A perceptron is a supervised classifier. It learn by first making a prediction: Is the dotproduct over or below the threshold? If it over the threshold it predicts a “1”, if it is below threshold it predicts a “0”.
Then the perceptron looks at the label of the sample. If the prediction was correct, then the error is “0”, and it leaves the weights alone. If the prediction was wrong, the error is either “-1” or “1” and the perceptron will update the weights like:

weights[feature_index] += learning_rate * error * feature_value

#### Example
In this example a perceptron learns to model a NAND function:
https://www.mathworks.com/matlabcentral/fileexchange/32949-a-perceptron-learns-to-perform-a-binary-nand-function

And there we have it, a simple linear classifying single node neural net: the perceptron.

## Online Learning Perceptron
The perceptron is capable of online learning (learning from samples one at a time). This is useful for larger datasets since you do not need entire datasets in memory.
Several authors have proposed online Perceptron variants that feature both the margin and kernel properties. This repository is a Matlab implementation of the online perceptron algorithm and its variants, including:

-	Online perceptron algorithm,
-	Online budget perceptron algorithm with fixed size of budget B,
-	Online budget perceptron algorithm with variable size of budget B
-	Forgetron algorithm
-	Online passive aggressive algorithm,
-	Online kernel passive aggressive algorithm,
-	Online least budget perceptron algorithm,
-	Online random budget perceptron algorithm,
-	Online tight budget perceptron algorithm,
-	Online margin infused relaxed algorithm (MIRA).


### References:

“Online (and Offline) on an Even Tighter Budget” by J. Weston, A. Bordes, and L. Bottou.

“Online Classification on a budget” by K. Crammer, J. Kaudola, and Y. Singer.

“Online Passive-Aggressive Algorithms on a Budget” by Z. Wang and S. Vucetic.

“Online Passive-Aggressive Algorithms” by K. Crammer et al.

“The Forgetron: A Kernel-Based Perceptron on a Fixed Budget” by O. Dekel, Sh. Shalev-Shwartz, and Y. Singer.

“Tighter Perceptron with Improved Dual Use of Cached for Model Representation and Validation” by Z. Wang and S. Vucetic.

“Margin-infused Relaxed Algorithm (MIRA)” by K. Crammer and Y. Singer.

