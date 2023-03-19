# Detecting-Contradiction-and-Entailment-in-Multilingual-Text

https://www.kaggle.com/competitions/contradictory-my-dear-watson

Two phrases can be connected in one of three ways: one can imply the other, one can contradict the other, or they can have no relation at all. A common NLP issue is Natural Language Inferencing (NLI), which entails figuring out the relationships between phrases that contain a premise and a hypothesis.

The task is to creare an NLI model that labels pairings of premises and hypotheses with labels of 0, 1, or 2 (equivalent to entailment, neutrality, and contradiction).

Deep Learning will be utilized:
  1. To create a fitting representation for the model's input by implementing BERT,
  2. To create an RNN Model with LSTM for classifying the premises and hypotheses pairs.