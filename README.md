# SDSF-KG

This work is about a secure KG sharing framework that supports both attribute privacy preservation and multi-granularity control--SDSF-KG.
It aims at ensuring secure and effective KG sharing, particularly in balancing data utility with privacy protection.

## Datasets and datasets processing

Since the three datasets based on OpenKE (FB13, FB15K, and FB15K237) cannot be directly used in our work, some preprocessing is required.
First, it is necessary to run the <code>uniform_datasets.py</code> to standardize the representation of KG entities and relations among three datasets, and then execute the <code>process_datasets.py</code> to split the data into train/valid/test sets for KGE training.

Alternatively, you can directly use the processed dataset provided by us, which is available in the following data path:
- <code>'./datasets/FB13/FB13.pkl'</code>
- <code>'./datasets/FB15K/FB15K.pkl'</code>
- <code>'./datasets/FB15K237/FB15K237.pkl'</code>

## KG embeddings training

Run the <code>kge_main.py</code> with the specified parameters to perform embedding training.

## Splitting sub-KGs and extracting attributes

After obtaining the KGE embedding results, run the <code>extract_attributes.py</code> to perform clustering-based sub-KGs partitioning and extract vital entities from each sub-KG as attribute sets.

## Secure sharing sub-KGs

Run the <code>charm_APABE.py</code> to perform secure sub-KGs sharing based on the improved attribute-based encryption scheme (AP-ABE).
Since the environment for the <code>Charm-Crypto</code> library is not easy to configure, we also provide a simplified implementation in the <code>pycryptodome_APABE.py</code> file.

## Supplementary implementations

- <code>data_integration.py</code>: To integrate the shared sub-KGs for KGE training in link prediction and triple classification tasks.

- <code>expansion_rates.py</code>: To calculate Entity Expansion Rate (EER), Relation Expansion Rate (RER), Triple Enrichment Rate (TER).

- <code>MI_AISR.py</code>: To calculate mutual information (MI) and adversarial inference success rate (AISR).

## Acknowledgement

- [OpenKE](https://github.com/thunlp/OpenKE)

- [Charm-Crypto](https://jhuisi.github.io/charm)
