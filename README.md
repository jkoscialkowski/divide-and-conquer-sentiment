# Divide (Text) and Conquer (Sentiment)
## Improved Sentiment Classification by Constituent Conflict Resolution

## Abstract
Sentiment classification, a complex task in natural language processing, becomes even more challenging when analyzing passages with multiple conflicting tones. Typically, longer passages exacerbate this issue, leading to decreased model performance. This paper introduces novel methodologies for isolating conflicting sentiments and effectively predicting the overall sentiment of such passages. Our approach involves a Multi-Layer Perceptron (MLP) model which significantly outperforms baseline models across various datasets, including Amazon, Twitter, and SST. In addition, the MLP model can be considered as an approximation of more advanced models and offers insights into standard sentiment classification models' inner workings.


## Setup
Run the following commads

```bash
python3 -m venv venv
./venv/bin/activate
pip3 install uv
uv pip install .
pre-commit install
```

To run tests, simply run `pytest`.

## Notebooks

## Paper
