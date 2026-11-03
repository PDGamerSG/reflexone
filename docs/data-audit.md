# Support data audit

ReflexOne's support model uses real English NLU++ examples from banking and
hotels. The source is pinned to
[`57ec275d8078af65b7731c2a98be812d844a6d6b`](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/nlupp/data).
This audit was verified against all 40 original folds and the original ontology.

The dataset is by Inigo Casanueva, Ivan Vulić, Georgios Spithourakis, and Paweł
Budzianowski: [NLU++: A Multi-Label, Slot-Rich, Generalisable Dataset for Natural
Language Understanding in Task-Oriented Dialogue](https://aclanthology.org/2022.findings-naacl.154/),
Findings of NAACL 2022, pages 1998–2013. The original
[dataset README](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/nlupp/README.md)
and [CC BY 4.0 license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE)
apply. Dataset and ontology attribution remain in the packaged ontology and
prepared manifest. The project's software license does not replace the dataset
license.

## Source validation

| Check | Result |
| --- | ---: |
| Original records | 3,080 |
| Banking / hotel records | 2,071 / 1,009 |
| Original slot annotations | 1,987 |
| Records with slots | 1,263 |
| Valid bounds and exact `text[start:end] == slot.text` | 1,987 / 1,987 |
| Invalid, off-by-one, or out-of-bounds spans | 0 |
| Overlapping slot pairs / records | 2 / 2 |
| Unknown or domain-incompatible intent/slot labels | 0 |
| Records with empty intent sets | 362 |
| Records with empty slot sets | 1,817 |
| Records repeating an intent label | 1 |

Offsets are zero-based Python character positions with an exclusive end.
Original text is retained, including case and whitespace. Missing `intents` or
`slots` fields mean empty annotations according to the dataset documentation.
These are valid examples and provide no supervised out-of-scope detection claim.
Original normalized calendar values are omitted because this model extracts
text spans and does not predict normalized dates or times.

Two original hotel records contain valid nested annotations:

| Source ID, zero-based row | First slot | Overlapping slot |
| --- | --- | --- |
| `hotels/fold12.json:53` | `date [43,57)` = `this afternoon` | `time_period [48,57)` = `afternoon` |
| `hotels/fold13.json:51` | `date [5,10)` = `today` | `date_period [0,10)` = `only today` |

Independent BIO heads preserve these cross-type overlaps. Each original
record's dictionary contains at most one annotation per slot type; multiple
same-type mentions are therefore outside the evidence supplied by this corpus.
Tokenizer alignment and truncation require separate validation during training.

The ontology defines 62 intent modules and 17 slot types. The known domain is
configuration supplied by the caller. General labels apply to either domain;
`refund` is explicitly shared. Valid masks contain 48 banking / 40 hotel intent
modules and 13 banking / 14 hotel slot types. Every permitted domain label is
observed in its domain in the original source. Label order is alphabetical for
both packaged and prepared ontologies.

## Duplicate handling

The grouping key uses Unicode NFKC normalization, case folding, and collapsed
whitespace. Its SHA256 is stored as `Record.group`. Normalization never changes
model input text or character offsets.

The original source contains 44 duplicate-text groups involving 92 rows and
48 excess rows. Three groups occur in both domains. Four groups disagree on
intent sets, and four different groups disagree on extractive slot annotations.
All 16 rows from these eight conflicting groups are quarantined. They are
recorded with source IDs, original text, intent sets, slots, and conflict reasons
in `audit.json`; they do not enter any training or evaluation split.

| Conflicting annotation | Source IDs |
| --- | --- |
| Intent: `Yes, I paid it yesterday` | `banking/fold6.json:90`, `banking/fold10.json:88` |
| Intent: room service menu greeting | `hotels/fold6.json:6`, `hotels/fold8.json:32` |
| Intent: asking for food in the room | `hotels/fold8.json:33`, `hotels/fold13.json:30` |
| Intent: handoff to spa | `hotels/fold16.json:7`, `hotels/fold16.json:27` |
| Slot: cancellation this morning | `banking/fold11.json:3`, `banking/fold11.json:4` |
| Slot: stolen card this morning | `banking/fold15.json:4`, `banking/fold15.json:5` |
| Slot: unsuccessful topup this morning | `banking/fold17.json:4`, `banking/fold17.json:5` |
| Slot: `two` versus `two boys` | `hotels/fold5.json:47`, `hotels/fold10.json:42` |

Conflict signatures compare the intent set and each slot's label, start, end,
and exact surface text. Calendar `value`/`values` fields do not affect this
extractive task. Intent sets remove the one repeated
`transfer_payment_deposit` label in `banking/fold13.json:74`.

For nonconflicting groups, one representative per domain is retained: the
lowest numeric fold index, then the lowest row index. This avoids lexical fold
ordering such as `fold10` preceding `fold2`. The manifest lists all collapsed
source IDs for each representative. Cross-domain records remain separate, but
their shared global group always enters the same split.

The result is **3,027 records**: banking **2,035**, hotels **992**; **3,024 global
text groups** and **1,964 extractive slot annotations**.

## Frozen holdout protocol

The custom split uses seed 17 and approximate group proportions 65% training,
15% validation, 10% calibration, and 10% test. Groups are bucketed by their known
domain membership, sorted, shuffled with one seeded Python random generator,
and allocated by largest-remainder rounding. The manifest freezes the source
bytes and split seed. Rerunning preparation with changed bytes or a different
seed in that directory raises an error. Model seeds reuse the same manifest.

| Partition | Records | Global text groups | Banking | Hotels |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,968 | 1,966 | 1,323 | 645 |
| Validation | 455 | 454 | 306 | 149 |
| Calibration | 302 | 302 | 203 | 99 |
| Test | 302 | 302 | 203 | 99 |

This is not the original NLU++ cross-validation protocol. Scores from these
partitions should not be represented as a direct benchmark comparison to the
paper. Splitting is approximately stratified by domain, not by every rare
label. Per-partition positive intent counts and slot counts, including zeros,
are in `audit.json`. Sparse labels can lack positive calibration/test examples.

Model selection and raw decision-threshold tuning use validation data.
Probability fitting uses only calibration data. The untouched test partition
measures the final frozen system. This protocol makes no conformal, OOS,
per-class probability-calibration, or deployment-distribution guarantee.

## Reproduction and fingerprints

From the repository root after `uv sync --extra dev`:

```bash
uv run python -c 'from pathlib import Path; from reflexone.data import prepare_dataset; prepare_dataset(Path("data/support"), seed=17)'
uv run pytest tests/test_data.py -q
```

Preparation downloads only absent files from the pinned revision using bounded
retries and request timeouts. Local `raw` files can be provided for offline
preparation; their exact bytes are hashed and frozen in the resulting manifest.
Source URLs record the intended pinned source. For provenance, use a fresh
directory or independently verify preexisting local raw files.

The prepared root contains `train.jsonl`, `validation.jsonl`, `calibration.jsonl`,
`test.jsonl`, `ontology.json`, `audit.json`, and `manifest.json`; pinned source
files are cached under `raw`. The manifest contains SHA256 checksums for every
source and prepared output, all record-to-split mappings, and duplicate
provenance. It is written last after successful validation. Records serialize
with the standard `dataclasses.asdict` shape.

The verified default preparation fingerprint is:

```text
98f780af199d09dc9ca38456ff9ec36db2da5ca255d6c92b0c61c22f4fc691ad
```

The fingerprint covers the preparation version, source revision and source
checksums, split seed, prepared ontology and records, and provenance manifest.
Generated data is excluded from ordinary source control.
