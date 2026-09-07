---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:41105
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Segmentation for radar images based on active contour
  sentences:
  - 'We devise and analyze algorithms for the empirical policy evaluation problem

    in reinforcement learning. Our algorithms explore backward from high-cost

    states to find high-value ones, in contrast to forward approaches that work

    forward from all states. While several papers have demonstrated the utility of

    backward exploration empirically, we conduct rigorous analyses which show that

    our algorithms can reduce average-case sample complexity from $O(S \log S)$ to

    as low as $O(\log S)$.'
  - 'Vegetation is a relevant feature in the urban scenery and its awareness can

    be measured in an image by the Green View Index (GVI). Previous approaches to

    estimate the GVI were based upon heuristics image processing approaches and

    recently by deep learning networks (DLN). By leveraging some recent DLN

    architectures tuned to the image segmentation problem and exploiting a

    weighting strategy in the loss function (LF) we improved previously reported

    results in similar datasets.'
  - 'We exam various geometric active contour methods for radar image

    segmentation. Due to special properties of radar images, we propose our new

    model based on modified Chan-Vese functional. Our method is efficient in

    separating non-meteorological noises from meteorological images.'
- source_sentence: Multi-borders classification
  sentences:
  - 'Using transformers over large generated datasets, we train models to learn

    mathematical properties of differential systems, such as local stability,

    behavior at infinity and controllability. We achieve near perfect prediction of

    qualitative characteristics, and good approximations of numerical features of

    the system. This demonstrates that neural networks can learn to perform complex

    computations, grounded in advanced theory, from examples, without built-in

    mathematical knowledge.'
  - 'The number of possible methods of generalizing binary classification to

    multi-class classification increases exponentially with the number of class

    labels. Often, the best method of doing so will be highly problem dependent.

    Here we present classification software in which the partitioning of

    multi-class classification problems into binary classification problems is

    specified using a recursive control language.'
  - 'This work deals with content-based video indexing. Our viewpoint is

    semi-automatic analysis of compressed video. We consider the possible

    applications of motion analysis and moving object detection : assisting moving

    object indexing, summarising videos, and allowing image and motion queries. We

    propose an approach based on interest points. As first results, we test and

    compare the stability of different types of interest point detectors in

    compressed sequences.'
- source_sentence: PyTorch, Explain! A Python library for Logic Explained Networks
  sentences:
  - '"PyTorch, Explain!" is a Python module integrating a variety of

    state-of-the-art approaches to provide logic explanations from neural networks.

    This package focuses on bringing these methods to non-specialists. It has

    minimal dependencies and it is distributed under the Apache 2.0 licence

    allowing both academic and commercial use. Source code and documentation can be

    downloaded from the github repository:

    https://github.com/pietrobarbiero/pytorch_explain.'
  - 'Learning disentangled representation from any unlabelled data is a

    non-trivial problem. In this paper we propose Information Maximising

    Autoencoder (InfoAE) where the encoder learns powerful disentangled

    representation through maximizing the mutual information between the

    representation and given information in an unsupervised fashion. We have

    evaluated our model on MNIST dataset and achieved 98.9 ($\pm .1$) $\%$ test

    accuracy while using complete unsupervised training.'
  - 'Long Short-Term Memory (LSTM) networks are often used to capture temporal

    dependency patterns. By stacking multi-layer LSTM networks, it can capture even

    more complex patterns. This paper explores the effectiveness of applying

    stacked LSTM networks in the time series prediction domain, specifically, the

    traffic volume forecasting. Being able to predict traffic volume more

    accurately can result in better planning, thus greatly reduce the operation

    cost and improve overall efficiency.'
- source_sentence: Quantile Convolutional Neural Networks for Value at Risk Forecasting
  sentences:
  - 'We present a new video understanding pentathlon challenge, an open

    competition held in conjunction with the IEEE Conference on Computer Vision and

    Pattern Recognition (CVPR) 2020. The objective of the challenge was to explore

    and evaluate new methods for text-to-video retrieval-the task of searching for

    content within a corpus of videos using natural language queries. This report

    summarizes the results of the first edition of the challenge together with the

    findings of the participants.'
  - 'We introduce DIODE, a dataset that contains thousands of diverse high

    resolution color images with accurate, dense, long-range depth measurements.

    DIODE (Dense Indoor/Outdoor DEpth) is the first public dataset to include RGBD

    images of indoor and outdoor scenes obtained with one sensor suite. This is in

    contrast to existing datasets that focus on just one domain/scene type and

    employ different sensors, making generalization across domains difficult. The

    dataset is available for download at http://diode-dataset.org'
  - 'This article presents a new method for forecasting Value at Risk.

    Convolutional neural networks can do time series forecasting, since they can

    learn local patterns in time. A simple modification enables them to forecast

    not the mean, but arbitrary quantiles of the distribution, and thus allows them

    to be applied to VaR-forecasting. The proposed model can learn from the price

    history of different assets, and it seems to produce fairly accurate forecasts.'
- source_sentence: 'Where To Look: Focus Regions for Visual Question Answering'
  sentences:
  - 'Unsupervised learning of hierarchical representations has been one of the

    most vibrant research directions in deep learning during recent years. In this

    work we study biologically inspired unsupervised strategies in neural networks

    based on local Hebbian learning. We propose new mechanisms to extend the

    Bayesian Confidence Propagating Neural Network (BCPNN) architecture, and

    demonstrate their capability for unsupervised learning of salient hidden

    representations when tested on the MNIST dataset.'
  - 'We present a method that learns to answer visual questions by selecting image

    regions relevant to the text-based query. Our method exhibits significant

    improvements in answering questions such as "what color," where it is necessary

    to evaluate a specific location, and "what room," where it selectively

    identifies informative image regions. Our model is tested on the VQA dataset

    which is the largest human-annotated visual question answering dataset to our

    knowledge.'
  - 'We propose a novel supervised multi-class/single-label classifier that maps

    training data onto a linearly separable latent space with a simplex-like

    geometry. This approach allows us to transform the classification problem into

    a well-defined regression problem. For its solution we can choose suitable

    distance metrics in feature space and regression models predicting latent space

    coordinates. A benchmark on various artificial and real-world data sets is used

    to demonstrate the calibration qualities and prediction performance of our

    classifier.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Where To Look: Focus Regions for Visual Question Answering',
    'We present a method that learns to answer visual questions by selecting image\nregions relevant to the text-based query. Our method exhibits significant\nimprovements in answering questions such as "what color," where it is necessary\nto evaluate a specific location, and "what room," where it selectively\nidentifies informative image regions. Our model is tested on the VQA dataset\nwhich is the largest human-annotated visual question answering dataset to our\nknowledge.',
    'Unsupervised learning of hierarchical representations has been one of the\nmost vibrant research directions in deep learning during recent years. In this\nwork we study biologically inspired unsupervised strategies in neural networks\nbased on local Hebbian learning. We propose new mechanisms to extend the\nBayesian Confidence Propagating Neural Network (BCPNN) architecture, and\ndemonstrate their capability for unsupervised learning of salient hidden\nrepresentations when tested on the MNIST dataset.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6844, 0.0525],
#         [0.6844, 1.0000, 0.0430],
#         [0.0525, 0.0430, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 41,105 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                            | positive                                                                             |
  |:---------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                               |
  | modality | text                                                                              | text                                                                                 |
  | details  | <ul><li>min: 5 tokens</li><li>mean: 14.92 tokens</li><li>max: 27 tokens</li></ul> | <ul><li>min: 88 tokens</li><li>mean: 220.91 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                             | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  |:-----------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Semi-supervised Federated Learning for Activity Recognition</code>                                                           | <code>Training deep learning models on in-home IoT sensory data is commonly used to<br>recognise human activities. Recently, federated learning systems that use edge<br>devices as clients to support local human activity recognition have emerged as<br>a new paradigm to combine local (individual-level) and global (group-level)<br>models. This approach provides better scalability and generalisability and also<br>offers better privacy compared with the traditional centralised analysis and<br>learning models. The assumption behind federated learning, however, relies on<br>supervised learning on clients. This requires a large volume of labelled data,<br>which is difficult to collect in uncontrolled IoT environments such as remote<br>in-home monitoring.<br>  In this paper, we propose an activity recognition system that uses<br>semi-supervised federated learning, wherein clients conduct unsupervised<br>learning on autoencoders with unlabelled local data to learn general<br>representations, and a cloud server conducts supervised learning ...</code> |
  | <code>SATR-DL: Improving Surgical Skill Assessment and Task Recognition in Robot-assisted Surgery with Deep Neural Networks</code> | <code>Purpose: This paper focuses on an automated analysis of surgical motion<br>profiles for objective skill assessment and task recognition in robot-assisted<br>surgery. Existing techniques heavily rely on conventional statistic measures or<br>shallow modelings based on hand-engineered features and gesture segmentation.<br>Such developments require significant expert knowledge, are prone to errors,<br>and are less efficient in online adaptive training systems. Methods: In this<br>work, we present an efficient analytic framework with a parallel deep learning<br>architecture, SATR-DL, to assess trainee expertise and recognize surgical<br>training activity. Through an end-to-end learning technique, abstract<br>information of spatial representations and temporal dynamics is jointly<br>obtained directly from raw motion sequences. Results: By leveraging a shared<br>high-level representation learning, the resulting model is successful in the<br>recognition of trainee skills and surgical tasks, suturing, needle-passing, and<br>knot-tyi...</code> |
  | <code>A Hybrid Stochastic Policy Gradient Algorithm for Reinforcement Learning</code>                                              | <code>We propose a novel hybrid stochastic policy gradient estimator by combining<br>an unbiased policy gradient estimator, the REINFORCE estimator, with another<br>biased one, an adapted SARAH estimator for policy optimization. The hybrid<br>policy gradient estimator is shown to be biased, but has variance reduced<br>property. Using this estimator, we develop a new Proximal Hybrid Stochastic<br>Policy Gradient Algorithm (ProxHSPGA) to solve a composite policy optimization<br>problem that allows us to handle constraints or regularizers on the policy<br>parameters. We first propose a single-looped algorithm then introduce a more<br>practical restarting variant. We prove that both algorithms can achieve the<br>best-known trajectory complexity $\mathcal{O}\left(\varepsilon^{-3}\right)$ to<br>attain a first-order stationary point for the composite problem which is better<br>than existing REINFORCE/GPOMDP $\mathcal{O}\left(\varepsilon^{-4}\right)$ and<br>SVRPG $\mathcal{O}\left(\varepsilon^{-10/3}\right)$ in the non-composite<br>setti...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 1
- `learning_rate`: 2e-05
- `warmup_steps`: 0.1
- `fp16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0389 | 50   | 0.0930        |
| 0.0778 | 100  | 0.0508        |
| 0.1167 | 150  | 0.0571        |
| 0.1556 | 200  | 0.0425        |
| 0.1946 | 250  | 0.0281        |
| 0.2335 | 300  | 0.0512        |
| 0.2724 | 350  | 0.0393        |
| 0.3113 | 400  | 0.0320        |
| 0.3502 | 450  | 0.0344        |
| 0.3891 | 500  | 0.0408        |
| 0.4280 | 550  | 0.0460        |
| 0.4669 | 600  | 0.0292        |
| 0.5058 | 650  | 0.0379        |
| 0.5447 | 700  | 0.0341        |
| 0.5837 | 750  | 0.0300        |
| 0.6226 | 800  | 0.0317        |
| 0.6615 | 850  | 0.0346        |
| 0.7004 | 900  | 0.0350        |
| 0.7393 | 950  | 0.0270        |
| 0.7782 | 1000 | 0.0356        |
| 0.8171 | 1050 | 0.0281        |
| 0.8560 | 1100 | 0.0331        |
| 0.8949 | 1150 | 0.0311        |
| 0.9339 | 1200 | 0.0402        |
| 0.9728 | 1250 | 0.0307        |


### Training Time
- **Training**: 2.6 minutes

### Framework Versions
- Python: 3.13.15
- Sentence Transformers: 6.0.1
- Transformers: 5.16.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.23.1

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->