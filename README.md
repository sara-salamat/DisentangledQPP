# Contrastive Disentangled Representation for Query Performance Prediction (CoDiR-QPP)

## Overview
CoDiR-QPP is a novel approach designed to enhance search query performance prediction by disentangling the semantic content of queries from their inherent difficulty. This project leverages neural disentanglement techniques and contrastive learning to isolate the information need expressed in search queries, which often impacts retrieval performance due to various complexities.

Our approach hypothesizes that separating content semantics from query difficulty can significantly improve the accuracy of predicting query performance. By distinguishing between well-performing and poorly performing query variants, CoDiR-QPP enables more precise estimation of a query’s potential success.

## Features
- **Neural Disentanglement**: Isolates semantic content from difficulty aspects of queries.
- **Contrastive Learning**: Enhances the model's ability to differentiate between varying performances of query formulations.
- **Improved Metrics**: Achieves higher correlation with standard performance metrics like Kendall τ, Spearman ρ, and scaled Mean Absolute Ranking Error (sMARE).
- **Extensive Validation**: Tested on four standard benchmark datasets, demonstrating superior performance over state-of-the-art baselines.

## Installation

'''
pip install -r requirements.txt
'''

To train a disentanglement model, rum the following command:
'''
python TrainT.py --base-model /path/to/model/ 
'''
To evaluate your trained model:
'''
python Evaluate.py --model-path /path/to/trained/model --data /path/to/data
