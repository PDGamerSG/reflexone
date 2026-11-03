# Data and model attribution

ReflexOne's original source code is MIT licensed. Third-party datasets and the pretrained encoder retain their licenses. Prepared data, source manifests, label ontology and trained-model archives include material derived from these sources; the MIT license does not replace the following terms.

| Component | Source / authors | License | Use and modifications |
|---|---|---|---|
| NLU++ | Casanueva, Vulić, Spithourakis and Budzianowski, [NAACL Findings 2022](https://aclanthology.org/2022.findings-naacl.154/); [dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/nlupp) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [upstream license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE) | Packaged ontology; normalized grouping, duplicate removal, annotation conflict quarantine, custom grouped splits; original text/character offsets retained; canonical calendar values omitted. |
| CLINC150 | Larson et al., [EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1131/); [dataset](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/); [upstream license](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE) | Duplicate audit, held-out calibration/rejection subdivisions; supervised OOS evaluation. Original OOS training data is unused. |
| SMS Spam Collection | Almeida and Hidalgo; [UCI dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84); Almeida, Hidalgo and Yamakami (2011), [paper](https://dl.acm.org/doi/10.1145/2034691.2034742) | CC BY 4.0 as stated on the UCI dataset page | Duplicate removal, conflicting-label quarantine and custom grouped stratified splits. Source archive hash is saved in the prepared manifest. |
| all-MiniLM-L6-v2 | [Sentence Transformers model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), Wang et al. MiniLM and the Sentence Transformers project | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Pretrained encoder fine-tuned on NLU++, with new domain embedding, multilabel intent head and independent slot heads. |

TypeSafe's [Jev introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is conceptual inspiration. No TypeSafe weights or proprietary training method are included, and no affiliation or endorsement is implied.

Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; their source distributions carry their respective licenses.
