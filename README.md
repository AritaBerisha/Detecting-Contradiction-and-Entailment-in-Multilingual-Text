# Detecting-Contradiction-and-Entailment-in-Multilingual-Text

https://www.kaggle.com/competitions/contradictory-my-dear-watson

Two phrases can be connected in one of three ways: one can imply the other, one can contradict the other, or they can have no relation at all. A common NLP issue is Natural Language Inferencing (NLI), which entails figuring out the relationships between phrases that contain a premise and a hypothesis.

The task is to creare an NLI model that labels pairings of premises and hypotheses with labels of 0, 1, or 2 (equivalent to entailment, neutrality, and contradiction).

Deep Learning will be utilized:
  1. To create a fitting representation for the model's input by implementing BERT,
  2. To create an RNN Model with LSTM for classifying the premises and hypotheses pairs.

## Multilingual NLI Dataset
This is a dataset of natural language inference (NLI) examples in multiple languages, including English, Spanish, French, and German. Each example consists of a premise and a hypothesis, along with a label indicating whether the hypothesis is entailed by, contradicted by, or neutral with respect to the premise.

## Dataset Details
The dataset contains approximately 12,000 examples, with a roughly equal distribution of labels across languages. The examples were sourced from various public datasets and preprocessed to remove any personally identifiable information.

## BERT Encoding Technique
To encode the examples for input to an RNN model, we use the BERT (Bidirectional Encoder Representations from Transformers) language model. Specifically, we use the bert-base-multilingual-uncased pre-trained model from the Hugging Face Transformers library.

We tokenize the premise and hypothesis using the BERT tokenizer, and encode them as input_ids and attention_masks. We then pass them through the BERT model to obtain a fixed-size representation of the input sequence, which is passed to an RNN model for prediction.

## RNN with LSTM Layer Technique
Our model consists of an LSTM layer followed by a fully connected layer with softmax activation. The LSTM layer takes the BERT-encoded input sequence and learns a representation of it by processing it sequentially. The output of the LSTM layer is passed through the fully connected layer to predict the label of the example.

We use the sparse categorical cross-entropy loss function and the Adam optimizer during training. We train the model using batches of 32 examples for 10 epochs.

## Model Retraining and Parameter Tuning

In order for improving and fine tuning the model, we have a few different parameters that can have various values:

- *lstm_dim*: The number of output units in the LSTM layer.
- *learning_rate*: The size of the steps taken to reach a (local) minimum in the optimization process.
- *dropout_rate*: The proportion of neurons in a layer that are randomly ignored (or "dropped out") during training.
- *clipnorm*: A threshold value for gradient clipping, which can prevent exploding gradients in neural networks.
- *optimizer_type*: The algorithm used to change the attributes of the neural network such as weights and learning rate in order to reduce the losses.
- *regularizer_type*: Regularization strategy used to prevent overfitting. L1 or L2 refers to the order of the norm used in the regularization term.
- *regularization_rate*: The scalar multiplier for the regularization term in the loss function.
- *batch_size*: The number of training examples utilized in one iteration.
- *epochs*: An epoch is a complete pass through the entire training dataset.

**Note**: You can find the results in the PDF File.
