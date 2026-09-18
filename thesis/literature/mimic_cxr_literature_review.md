# Literature Review on the MIMIC-CXR Dataset: A Comparative Evaluation of Text, Image, and Multimodal Analyses

The staggering advances in medical artificial intelligence, computer vision, and natural language processing (NLP) over the past decade are directly tied to the ability of machine learning algorithms to be trained on massive datasets. The field of radiology is at the center of this technological revolution because it inherently generates high volumes of unstructured data (images and free-text reports). In the past, researchers were limited to small-scale datasets that were extremely expensive and time-consuming to annotate. However, the MIMIC-CXR dataset, published on PhysioNet in collaboration with Beth Israel Deaconess Medical Center (BIDMC) and the Massachusetts Institute of Technology (MIT), has become an unshakable benchmark for global medical AI research¹.

This report provides an in-depth review of past papers, theses, and independent studies conducted on the MIMIC-CXR dataset; deciphering the ontological structure of the dataset, subsetting strategies, and the three primary analytical approaches applied (text-only, image-only, and multimodal). The literature shows that each approach has its own architectural advantages, clinical applicability, and explainability limitations. In this context, the report goes beyond mere performance metrics and synthesizes second and third-order dynamics such as algorithmic bias, data leakage, and causality into a comprehensive narrative.

## Structural Analysis of the MIMIC-CXR Dataset and Subsetting Strategies

MIMIC-CXR is a massive multimodal collection containing 227,835 imaging studies and a total of 377,110 chest radiographs belonging to 65,379 unique patients, collected between 2011 and 2016¹. The creation of this dataset required the independent processing and subsequent pairing of three different data modalities: electronic health records (EHR), high-resolution medical images (DICOM), and free-text radiology reports⁶.

### Processing Workflows for Image and Text Data

To fully ensure privacy and ethical standards (HIPAA), the dataset underwent an extensive de-identification process. The original images in DICOM format were scrubbed using the Orthanc tool, and all unique identifiers (UID) that could identify the patient were replaced with randomly generated Universally Unique Identifiers (UUID)⁶. The images typically include posteroanterior (PA), anteroposterior (AP), and lateral (LAT) views³. To reduce the data processing burden on researchers, these DICOM files were later subjected to contrast enhancement, pixel intensity inversion, and normalization processes to be converted into an 8-bit JPG format (MIMIC-CXR-JPG)³.
Today, due to computational efficiency, many model development processes are conducted on this JPG version³.

Each imaging study was paired with a free-text radiology report dictated by board-certified radiologists¹. These reports generally consist of semi-structured sections such as "Clinical Indication," "Findings," and "Impression." However, for machine learning models to be trained directly via supervised learning, these free texts needed to be converted into mathematically expressible, structured labels.

### Automated Labeling Systems and the Long-Tail Problem

The labels in the MIMIC-CXR dataset were not extracted manually but automatically from the reports via rule-based, medical ontology-aware Natural Language Processing (NLP) labelers such as CheXpert and NegBio¹. These systems analyze the report text and output four different probability states for 14 different pathology classes: Positive (1.0), Negative (0.0), Uncertain (-1.0), or Unmentioned (Blank)¹.
The table below compares the structure of the MIMIC-CXR dataset with other large datasets in the field.

| Dataset | Number of Images | Number of Reports | Number of Labels | Data Format and Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **MIMIC-CXR** | 377,110 | 227,835 | 14 | Free-text reports, DICOM images¹ |
| **MIMIC-CXR-JPG** | 377,110 | - | 14 | Images in JPG format, structured labels¹ |
| **CheXpert** | 224,316 | - | 14 | JPEG images, structured labels including uncertainty¹ |
| **PadChest** | 160,868 | 109,931 | 193 | Reports in Spanish, DICOM images, multi-label¹ |
| **IU X-Ray (Open-I)** | 7,470 | 3,955 | 189 | Free-text reports, DICOM/PNG, manual MeSH labels¹ |

The 14-class structure in the dataset exhibits a severe imbalance in terms of class distribution. This phenomenon, known as a "long-tail distribution" in machine learning literature, refers to the situation where some classes (e.g., Atelectasis or Pleural Effusion) are represented by tens of thousands of samples, while some other clinical findings (hernia, support devices) are seen very rarely³.
Long-tail analyses on MIMIC-CXR-JPG show that 11 of the classes are common (>10,000 samples), 17 are moderate (1,000-10,000 samples), and 12 are very rare (<1,000 samples)³. *(Note: As in the original Turkish text, the sum of these values is 40, which differs from the 14 pathology classes mentioned earlier. This reflects the source text accurately.)*

### Dataset Reduction and Subsetting Strategies

Training a model on the entire dataset across 14 pathologies not only requires high computational costs but also leads to overfitting in rare classes or representation collapse due to data imbalance. To overcome this problem, researchers in the literature often reduce the dataset by applying specific logical filters:

1. **Selecting the Top 5 Clinically Dominant Classes:** Many studies focus on the 5 core pathologies that are clinically most common and most stable to model, rather than all 14 classes: Cardiomegaly, Edema, Consolidation, Pleural Effusion, and Atelectasis⁸. This quintet is considered a touchstone, especially for standardizing the performance of newly developed algorithms (such as CheXzero or RadAlign)¹².
2. **Filtering Report Sections:** The "Clinical Indication" section in radiology reports may contain the patient's past diseases or complaints rather than their current physical condition. This adds noise to the training data. Studies narrow down the dataset and increase semantic consistency by filtering reports that only contain the structured "Findings" or "Impression" sections⁸.
3. **View and Demographic Filtering:** Lateral radiographs carry vastly different geometric features for algorithms compared to frontal views. Many studies reduce the dataset by selecting only Frontal (PA/AP) views to reduce complexity¹. Similarly, to achieve fair results in model testing, the dataset is filtered by demographic meta-data such as age, race, and gender, and analyzed in subsets¹⁵.

After these filterings are applied, text-only, image-only, or multimodal experiments encompassing both are conducted, depending on the nature of the targeted analysis.

## 1. Text-Only Analyses

Radiology reports are the condensed semantic representations of the complex visual information contained in pixels, filtered through human expertise (the radiologist). In the literature, the "text-only" approach is based on analyzing and classifying radiology reports, extracting critical entities related to the disease, and evaluating synthetic report generation models.

### The Evolution of NLP from Rule-Based Systems to Large Language Models

The methodology for extracting meaning from MIMIC-CXR reports has undergone a radical evolution from basic rule-based matching to deep learning architectures with massive parameters.
In the early period, rule-based systems like CheXpert-NLP and NegBio were used when analyzing radiology texts¹. NegBio is a high-performance tool that utilizes universal dependencies syntactic parsing and subgraph matching to detect negation and uncertainty in radiology reports¹. For example, the structural difference between the sentence "No pneumothorax seen" and "Patient's suspicion of pneumothorax continues" is manually encoded into the rules of these tools.

However, rule-based systems can fall short in the face of medical language's flexibility, synonymy, and complex grammatical contexts. These limitations were overcome as BERT (Bidirectional Encoder Representations from Transformers)-based architectures dominated the literature. Pre-trained language models on biomedical literature, such as ClinicalBERT and BioBERT, have the capacity to bidirectionally model the context in MIMIC-CXR reports¹⁹. A BERT-based classifier named CheXbert, built upon the CheXpert labeler, set a new standard in text classification by achieving a much higher F1 score of 0.798 (compared to CheXpert's F1 score of 0.743) against traditional rule-based systems¹¹.

At a more advanced level, research using Large Language Models (LLMs) can determine not only the presence/absence of pathology but also the anatomical location, size, and severity of the lesion. Systems like MAPLEZ (based on SOLAR-0-70b LLM) outperform traditional labelers (including CheXbert) by significant margins in F1 metrics by processing reports via knowledge-driven decision trees⁹. This proves that LLMs have become dominant not only in reading comprehension tasks but also in structured ontological data generation.

### Evaluation Metrics for Text Generation

The metrics used to measure the success of vision-to-text systems or textual analyses have also evolved. Superficial metrics traditionally used, such as BLEU, ROUGE, or METEOR, have completely failed to measure the accuracy of medical reports because they are based on n-gram matching⁸. Because omitting a word (writing "seen" instead of "not seen") has catastrophic consequences in a medical context, n-gram-based metrics score this as a minor error.

Therefore, the literature has developed new metrics with "clinical awareness." RadGraph F1 uses knowledge graphs to calculate how much the clinical entities and the relations between these entities overlap between the generated text and the reference text (ground-truth)¹. The recently proposed GREEN (Generative RadiOlogy Evaluation and Error Network) score uses LLM-based reasoning to detect clinically meaningful errors and exhibits a much higher correlation with human radiologist decisions compared to traditional metrics (for example, achieving scores of 0.511+ on the ReXVal dataset compared to BLEU-4's Kendall tau score of 0.345)¹³.

### The Performance Paradox and "Data Leakage"

There is an interesting finding in classification tasks performed using only text: Text models always produce much higher Accuracy, AUC, and F1 scores compared to image models²⁵. For example, a logistic regression or BERT-based text model can easily achieve >0.90 AUC values on MIMIC-CXR.

However, this high performance is the result of an algorithmic "data leakage" problem rather than a clinical revolution²⁶. The targeted classes (ground truth labels) have already been generated as a result of analyzing the texts with NLP tools. Thus, the text model actually learns to predict how the labeler algorithm summarized the report, not the disease itself. In the analyzed articles, this situation is referred to as "proxy-label ambiguity"²⁶. This reality proves that text-only models cannot be used as a direct, independent diagnostic tool; however, they provide a unique foundation for retroactively classifying millions of reports in existing hospital databases, triggering alert systems, or training multimodal models.

## 2. Image-Only Analyses

The "image-only" analysis approach, which takes radiological images (pixels) directly as input, is the most traditional form of medical artificial intelligence. These models aim to perform pathology detection, lesion localization, or anomaly segmentation by looking directly at the patient's physical scan and attempting to simulate the radiologist's visual cortex.

### The Transition from CNN Dominance to Vision Transformers (ViT)

For a long time, Convolutional Neural Networks (CNNs) formed the backbone of visual analyses on the MIMIC-CXR dataset. Deep learning architectures like ResNet (Residual Networks - specifically ResNet-121 and ResNet-50), DenseNet-121, and EfficientNet are initialized with pretrained weights on ImageNet and fine-tuned to detect pathologies in X-ray radiographs²¹. The hierarchical feature extraction capability of CNNs is highly successful in modeling low-level features like edges and textures in lower layers, and high-level abnormalities like calcifications or masses in upper layers. It is frequently reported in the literature that models using only images, such as DenseNet-121, achieve overall AUROC values in the range of 0.83 to 0.85 on MIMIC-CXR or CheXpert²¹.

More recently, Vision Transformers (ViT), the adaptation to visual data of the Transformer architecture that broke new ground in natural language processing, created a new paradigm in image-only analysis. ViTs, which divide the image into "patches" consisting of small pixels instead of a convolution operation and learn the global context between these patches with attention mechanisms, have caught up with the performance equivalent to CNNs in medical imaging (e.g., 0.95 AUC on Chest X-Ray 14, 0.84 AUC on VinBigData)³⁰.

### Differences in Explainability Mechanisms

In image-only analyses, it is vital for the model to be able to explain its decision to the clinician. CNNs typically generate heatmaps using the Grad-CAM (Gradient-weighted Class Activation Mapping) method based on the model's weights²⁶. These maps show which regions of the image the network focused on when making the "consolidation" diagnosis.

ViT-based models, on the other hand, generate saliency maps using the network's own "attention" weights directly, such as Transformer Multimodal Explainability (TMME). In user studies, expert radiologists found ViT-based attention maps to be much more useful and clinically consistent than traditional CNN-based Grad-CAM maps (radiologist approval rate 47% vs. 39%)³⁰.

### Challenges Encountered: Shortcut Learning and Bias

The greatest weakness of pixel-only models is that they are highly susceptible to spurious correlations carried by medical images. Unlike a natural image dataset, CXRs contain external artifacts such as pacemakers, cables, surgical clips, and tubes. Studies have revealed that when diagnosing pneumothorax (collapsed lung), CNNs and ViTs mostly focus on the "chest drain" placed there to apply the treatment, rather than looking at the aeration area in the patient's pleura³⁰. The model falls into the swamp of shortcut learning by memorizing a fake rule in the form of "chest drain = pneumothorax".

Furthermore, subgroup analyses performed on these systems have shown that model success changes dramatically according to demographic attributes such as the patient's age, gender, and race, and that performance drops occur in minority groups¹⁵. This dataset bias is one of the biggest technical and ethical barriers to using image-only models independently in clinical settings. The necessity has emerged to establish causal structures (causal modeling) or to support the model with an external knowledge source (e.g., text) so that it can make fair and robust predictions across demographic substrata.

## 3. Analyses Using Text and Image Together (Multimodal)

The fact that text-only models are vulnerable to label leakage and image-only models are vulnerable to shortcut learning has driven researchers to Multimodal architectures that combine the advantages of both worlds. The MIMIC-CXR dataset is the largest and most valuable laboratory in the world for training and testing such architectures, as the image and the corresponding radiology report are naturally paired¹.

### Architecture of Fusion Strategies

Traditionally, architectures that combine text and image data for a single decision output are categorized based on the stage at which the information is combined:

1. **Early Fusion:** Vectors extracted from image pixels and vectors obtained from text tokens are combined before entering the deep layers of the model. Typically, through cross-attention or feature concatenation operations, the network learns at the very beginning which region of the image is directly related to which word in the text²⁸.
2. **Late Fusion:** The two modalities (e.g., BERT/TF-IDF for text, ResNet18 for image) are trained completely independently from each other. The final stage probability scores or top-level (global) feature vectors extracted are combined at the decision stage with a weighting mechanism²⁶.

In comprehensive experiments on late fusion models, significant statistical improvements were achieved in both AUC values (e.g., up to the 0.902 - 0.947 levels) and Accuracy rates compared to single modalities²⁶. This situation, called the "Complementarity Effect," stems from the fact that text provides clinical filtering while the image presents physical evidence.

### Vision-Language Pretraining (VLP) and Contrastive Learning

Beyond fusion architectures, the real revolution shaping the modern medical AI literature has been Vision-Language Pretraining (VLP) and Contrastive Learning. The adaptation of the CLIP (Contrastive Language-Image Pretraining) architecture to the medical field completely changed the model training paradigm. These systems train a neural network not to predict disease classes (0 or 1), but to understand whether a radiology image (X-ray) and the report corresponding to that image belong to each other¹². By using contrastive loss functions like InfoNCE, correct image-text pairs are brought closer together in the same embedding space, while the images and texts of other patients are pushed apart³².

Pioneering VLP models trained on MIMIC-CXR include:
* **ConVIRT:** One of the first models to apply bidirectional contrastive learning on paired medical data³².
* **BioViL and BioViL-T:** Highly powerful architectures that consider not only spatial but also temporal correlation (patient progression over time), enriching text modeling¹².
* **MedCLIP / CheXzero / RadAlign:** Systems like CheXzero and RadAlign, in particular, eliminate human intervention by directly aligning the raw free text of the report without needing rule-based expert labels (CheXpert labels)¹².

### Zero-Shot Classification and Model Flexibility

These models trained with VLP have gained Zero-Shot classification capacity, which is an extraordinary ability to detect pathologies that have never been manually labeled before. Once the system is trained with MIMIC-CXR data, when given a new X-ray image and a text prompt like "There are signs of pneumonia in this image," the model calculates how well the image represents this sentence by looking at vector similarity (cosine similarity)¹².

Research has proven that models like CheXzero, BioViL, and RadAlign outperform fully supervised classical CNN models when applied on out-of-domain datasets they have never seen before (CheXpert, PadChest, or OpenI), and even achieve diagnostic accuracies equal to or superior to board-certified expert radiologists in some specific pathologies (e.g., average F1: 0.652, AUC: 0.923 on OpenI)¹². This presents unique potential for detecting rare diseases, as it eliminates the necessity for labeling in large datasets⁴².

### Radiology Report Generation (RRG)

The most complex dimension of multimodality is Generative Models that can take a chest X-ray as input and produce free-text reports detailing anatomical and pathological findings at a radiologist level⁸. Recently, Multimodal Large Language Models (MLLMs) such as MedGemma, ChatCAD, and RadAlign have begun working in the RRG domain by combining visual features with LLM reasoning¹³.

For example, the RadAlign system uses vision-language concept alignment not just to make a static classification; it also finds past cases (retrieval-augmented generation) to support its diagnosis, increasing the clinical realism of the generated report. The success of these systems is measured by metrics like the GREEN score (RadAlign: 0.678), showing that the generated texts are found by radiologists to be of equal quality to their own clinical reports¹⁰.

## Analytical Deductions, Contradictions, and Development Trends

This literature centered around the MIMIC-CXR dataset contains deep clinical and structural dynamics (second and third-order insights) far beyond the performance race of technological architectures:

1. **The "Error Ceiling" Created by NLP Labelers and Overcoming It:** Image-only (CNN/ViT) models fall into a structural trap when trained with labels generated by rule-based NLP tools like CheXpert or NegBio. At best, the image model can reach the accuracy level possessed by this labeling NLP algorithm. When the labeler accepts an error as true (for example, perceiving a negative finding as positive), the image model also accepts this "fake truth" as true and builds an incorrect hierarchy¹⁸. Contrastive learning-based multimodal models (CheXzero, BioViL) solved this problem radically. Instead of converting reports to 1s and 0s with an intermediate NLP tool, they directly align pixels with word vectors in free text¹². Thus, the "error ceiling" created by NLP has been destroyed; context is embedded directly into the model's weights.
2. **Paradigm Shift in Explainability Perception:** In the past, the explainability of medical AI was seen as a heatmap (Grad-CAM) "painting the diseased area red"²⁶. However, shortcut learning analyses (deciding by looking at the tube) bitterly revealed that the area painted red has no clinical meaning³⁰. Current research has shifted explainability to "Explain-then-predict" mechanisms³⁸. Systems like RATCHET, TieNet variants, or Xplainer must produce their radiological justifications in text format (NLE - Natural Language Explanations) before giving the diagnosis result³⁸. For a doctor, it is much more of a confidence builder for a model to say, "Pleural effusion was detected because the right costophrenic angle is blunted," rather than "Pleural effusion probability 92%."
3. **The Rise of Bias and Causal Networks:** Subgroup performance disparities detected on MIMIC-CXR (increased false diagnosis rates in minority groups) indicate that systems establish fake correlations between age, gender, or race and pathologies¹⁵. The way to prevent this is not to increase the amount of data, but to inject "causality" into the data. In the literature, studies are accelerating that filter demographic features over CXR data with Structural Causal Models and build Continuous-Time Flow models¹⁵. This is the strongest indicator that AI is evolving from being a mere statistical pattern matcher to an entity capable of modeling anatomical and epidemiological cause-and-effect relationships.

## Overall Evaluation

Research conducted on the MIMIC-CXR dataset, which has become the epicenter of medical imaging AI, shows that the field is not progressing linearly; on the contrary, it creates disruptive paradigms.

Despite their high analytical power, text-only models have remained far from being a standalone diagnostic tool due to "data leakage"; however, they have provided revolutionary infrastructures (LLM-based report analyzers) for hospital data mining and labeling systems. Image-only systems, while strong at detecting anomalies at the pixel level, have suffered from chronic problems such as weak explainability, susceptibility to spurious correlations (shortcut learning), and fragility against label noise.

Ultimately, the literature has forced AI architectures to accept that medical diagnosis is inherently a multimodal process. Architectures (CLIP, BioViL, MedCLIP) that contrastively train image and text data in the same semantic vector space have not only raised accuracy metrics (AUROC/AUPRC); they have opened entirely new horizons in the field with zero-shot capabilities that eliminate dependency on manual labeling, and synthetic radiology report generation (RRG) equipped with reasoning capacity. The future of computational radiology lies not in isolated systems that mechanically classify pixels, but in large-scale Vision-Language Foundation Models ecosystems capable of synthesizing visual, linguistic, temporal, and clinical information within a causal integrity.

## Cited Studies

1. Recent Progress in Deep Learning for Chest X-Ray Report Generation - MDPI, https://www.mdpi.com/2673-7426/6/1/3
2. Review on chest pathogies detection systems using deep learning techniques - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC10027283/
3. MIMIC-CXR-JPG Dataset - Emergent Mind, https://www.emergentmind.com/topics/mimic-cxr-jpg-dataset
4. Role of Model Size and Prompting Strategies in Extracting Labels from Free-Text Radiology Reports with Open-Source Large Language Models - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12920854/
5. (PDF) MIMIC-CXR: A large publicly available database of labeled chest radiographs, https://www.researchgate.net/publication/330552843_MIMIC-CXR_A_large_publicly_available_database_of_labeled_chest_radiographs
6. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC6908718/
7. MIMIC-CXR-JPG - chest radiographs with structured labels v2.1.0 - PhysioNet, https://physionet.org/content/mimic-cxr-jpg/
8. Radiology Report Generation with Layer-Wise Anatomical Attention - arXiv, https://arxiv.org/pdf/2512.16841
9. Enhancing chest X-ray datasets with privacy-preserving large language models and multi-type annotations: a data-driven approach - arXiv, https://arxiv.org/pdf/2403.04024
10. A unified multi-task framework enables interpretable chest radiograph analysis - arXiv, https://arxiv.org/html/2606.03417v1
11. Enhancing chest X-ray datasets with privacy-preserving large language models and multi-type annotations: a data-driven approach for improved classification - arXiv, https://arxiv.org/html/2403.04024v2
12. Significantly improving zero-shot X-ray pathology classification via fine-tuning pre-trained image-text encoders - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11455863/
13. RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignment - arXiv, https://arxiv.org/html/2501.07525v2
14. Evaluating progress in automatic chest X-ray radiology report generation - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/
15. Scaling Generative Foundation Models for Chest Radiography with Rectified Flow Transformers - arXiv, https://arxiv.org/html/2606.19460v1
16. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison | Request PDF - ResearchGate, https://www.researchgate.net/publication/335800693_CheXpert_A_Large_Chest_Radiograph_Dataset_with_Uncertainty_Labels_and_Expert_Comparison
17. Learning to diagnose common thorax diseases on chest radiographs from radiology reports in Vietnamese - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC9621405/
18. (PDF) Limitations of Public Chest Radiography Datasets for Artificial Intelligence: Label Quality, Domain Shift, Bias and Evaluation Challenges - ResearchGate, https://www.researchgate.net/publication/395650024_Limitations_of_Public_Chest_Radiography_Datasets_for_Artificial_Intelligence_Label_Quality_Domain_Shift_Bias_and_Evaluation_Challenges
19. Large Language Models for Healthcare Text Classification: A Systematic Review - arXiv, https://arxiv.org/html/2503.01159v1
20. A Survey of Large Language Models in Medicine: Progress, Application, and Challenge - arXiv, https://arxiv.org/pdf/2311.05112
21. K-STAMM: a knowledge-enhanced spatial – temporal attention model with multimodal fusion for pneumonia prediction - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC13223296/
22. arXiv:2311.16764v1 [cs.CL] 28 Nov 2023, https://arxiv.org/pdf/2311.16764
23. Automated labelling of radiology reports using natural language processing: Comparison of traditional and newer methods - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11080679/
24. Image-aware Evaluation of Generated Medical Reports - arXiv, https://arxiv.org/pdf/2410.17357
25. Evaluation of Multimodal Fusion Strategies For ... - Simple search, https://ltu.diva-portal.org/smash/get/diva2:2064860/FULLTEXT02.pdf
26. An Explainable Multimodal Framework for Chest X-Ray Alert Classification Using Radiology Reports and Images, https://publikasi.dinus.ac.id/jcta/article/download/16023/5999/57200
27. Knowledge-enhanced visual-language pre-training on chest radiology images - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC10382552/
28. Advancing Medical Image Diagnostics through Multi-Modal Fusion: Insights from MIMIC Chest X-Ray Dataset Analysis - ResearchGate, https://www.researchgate.net/publication/382207473_Advancing_Medical_Image_Diagnostics_through_Multi-Modal_Fusion_Insights_from_MIMIC_Chest_X-Ray_Dataset_Analysis
29. Deep Learning for Automatic Lung Disease Analysis in Chest X-rays - TUHH Open Research, https://tore.tuhh.de/bitstream/11420/9470/3/PhD_Thesis_IBa_TORE.pdf
30. Attention-based Saliency Maps Improve Interpretability of Pneumothorax Classification | Radiology: Artificial Intelligence - RSNA Journals, https://pubs.rsna.org/doi/abs/10.1148/ryai.220187
31. Fairness and Robustness of CLIP-Based Models for Chest X-rays - arXiv, https://arxiv.org/html/2507.21291v1
32. Boosting Medical Vision-Language Pretraining via Momentum Self-Distillation under Limited Computing Resources - arXiv, https://arxiv.org/pdf/2512.02438
33. arXiv:2405.06468v3 [cs.CV] 13 Sep 2024, https://arxiv.org/pdf/2405.06468
34. Advancements in Medical Radiology Through Multimodal Machine Learning: A Comprehensive Overview - MDPI, https://www.mdpi.com/2306-5354/12/5/477
35. Leveraging Knowledge for Medical Image Understanding in, https://collab.dvb.bayern/spaces/TUMdlma/pages/309631436/Leveraging+Knowledge+for+Medical+Image+Understanding+in+Radiology
36. Can Medical Vision-Language Pre-training Succeed with Purely, https://openreview.net/forum?id=rawj2PdHBq-eId=M92MauMXKP
37. arXiv:2503.01019v3 [cs.CV] 20 Apr 2025, https://arxiv.org/pdf/2503.01019
38. arXiv:2303.13391v3 [cs.CV] 28 Jun 2023, https://arxiv.org/pdf/2303.13391
39. A Multimodal Biomedical Foundation Model Trained... : NEJM AI - Ovid, https://www.ovid.com/02275075-202501000-00006
40. Bringing CLIP to the Clinic: Dynamic Soft Labels and Negation-Aware Learning for Medical Analysis - arXiv, https://arxiv.org/html/2505.22079v1
41. PairAug: What Can Augmented Image-Text Pairs Do for Radiology? - CVF Open Access, https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_PairAug_What_Can_Augmented_Image-Text_Pairs_Do_for_Radiology_CVPR_2024_paper.pdf
42. and Few-Shot Anomaly Detection in Chest X-Rays - JYX, https://jyx.jyu.fi/bitstreams/dce3f054-4d7e-4496-bafc-a9230d78c155/download
43. MedGemma Technical Report - arXiv, https://arxiv.org/html/2507.05201v4
44. ChestX-VQA: AI Tool for Multimodal Chest X-ray Analysis and Clinical QA - Research Square, https://assets-eu.researchsquare.com/files/rs-7563215/v1_covered_2566e428-a3dd-4d99-9b0a-042151fa8cdf.pdf?c=1765990531
45. Multimodal Modeling of Chest X-rays - reposiTUm, https://repositum.tuwien.at/bitstream/20.500.12708/209509/1/Blasko%20Daniel%20-%202024%20-%20Multimodal%20modeling%20of%20chest%20X-rays.pdf
