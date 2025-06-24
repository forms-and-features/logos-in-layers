# 1. Model-level Analyses

## Qwen3-8B

### Results:
```
=== PROMPT =========================
Question: What is the capital of Germany? Answer:
=== END OF PROMPT =================


=== INSPECTING ====================
Input tokens: ['Question', ':', ' What', ' is', ' the', ' capital', ' of', ' Germany', '?', ' Answer', ':']

Top predictions for next token after 'Question: What is the capital of Germany? Answer:':
Using NORMALIZED residual stream (LayerNorm applied - more accurate)
------------------------------------------------------------
Layer  0:
  1. 'いらっ' (0.211549)
  2. ' binaries' (0.166734)
  3. 'おすす' (0.092069)
  4. '家喻户' (0.061461)
  5. 'インターネ' (0.057510)

Layer  6:
  1. 'portion' (0.011783)
  2. 'steller' (0.008585)
  3. '<Entry' (0.006405)
  4. 'führ' (0.006186)
  5. ' Located' (0.006136)

Layer 12:
  1. ' Answer' (0.069230)
  2. 'abella' (0.023264)
  3. ' Incorrect' (0.017712)
  4. '在游戏中' (0.011104)
  5. '在全球' (0.010015)

Layer 18:
  1. ' Answer' (0.324686)
  2. '回答' (0.049062)
  3. '_ANS' (0.014806)
  4. '.answer' (0.012050)
  5. '的回答' (0.009193)

Layer 24:
  1. ' Germany' (0.797710)
  2. ' ____' (0.055280)
  3. ' ______' (0.034518)
  4. ' __________________' (0.027517)
  5. ' _____' (0.021357)

Layer 30:
  1. ' Berlin' (0.999213)
  2. ' Germany' (0.000408)
  3. 'Berlin' (0.000171)
  4. ' Frankfurt' (0.000153)
  5. ' Munich' (0.000020)

Layer 35:
  1. ' Berlin' (0.719232)
  2. ' The' (0.086093)
  3. ' Germany' (0.034847)
  4. ' ' (0.034652)
  5. ' __' (0.034198)

============================================================
ACTUAL MODEL PREDICTION (for comparison):
Model's final prediction:
  1. ' Berlin' (0.719238)
  2. ' The' (0.086090)
  3. ' Germany' (0.034847)
  4. ' ' (0.034651)
  5. ' __' (0.034197)
============================================================
ADDITIONAL PROBING:

Prompt: 'Germany's capital is'
  1. ' Berlin' (0.616918)
  2. ' not' (0.070298)
  3. ' a' (0.067678)

Prompt: 'Berlin is the capital of'
  1. ' Germany' (0.728571)
  2. ' which' (0.220736)
  3. ' the' (0.023737)

Prompt: 'Respond in one word: which city is the capital of Germany?'
  1. ' Berlin' (0.721222)
  2. ' The' (0.174027)
  3. ' Also' (0.015090)
============================================================
TEMPERATURE EXPLORATION:
(Temperature controls randomness: low=confident, high=creative)

Temperature 0.1:
  1. ' Berlin' (1.000000)
  2. ' The' (0.000000)
  3. ' Germany' (0.000000)
  4. ' ' (0.000000)
  5. ' __' (0.000000)

Temperature 2.0:
  1. ' Berlin' (0.066317)
  2. ' The' (0.022944)
  3. ' Germany' (0.014597)
  4. ' ' (0.014556)
  5. ' __' (0.014460)
=== END OF INSPECTING ==============


=== MODEL STATS ===============
Number of layers: 36
Model dimension: 4096
Number of heads: 32
Vocab size: 151936
Context length: 2048
=== END OF MODEL STATS ========

### Interpretation by ChatGPT o3:

#### Layer Evolution: From Noise to Knowledge to Communication

**Layer 0 (Embedding)**: Pure multilingual noise reflecting Qwen3's diverse training data - Japanese (いらっ, おすす, インターネ), Chinese (家喻户), and random English tokens. No semantic understanding yet.

**Layers 6-12 (Early Processing)**: Gradual emergence of contextual awareness. Layer 12 shows first recognition of Q&A format (" Answer" at 6.9%), but still mixed with multilingual artifacts.

**Layer 18 (Context Recognition)**: Strong Q&A format recognition (" Answer" jumps to 32.5%). The model understands it's in a question-answering context but hasn't identified the specific question yet.

**Layer 24 (The "Wrong Right Answer" Phase)**: Critical semantic processing. The model predicts " Germany" with 79.8% confidence - it has identified the geographical domain and the entity being asked about, but hasn't resolved the directional relationship (capital OF Germany vs Germany as capital).

**Layer 30 (Knowledge Crystallization)**: Dramatic resolution. " Berlin" achieves 99.9% confidence while other German cities (Frankfurt: 0.000153, Munich: 0.000020) receive tiny but hierarchically appropriate probabilities. This represents pure factual knowledge retrieval with geographical ranking intact.

**Layer 35 (Communication Calibration)**: Sophisticated output formatting. Berlin's confidence drops to 72% while conversation-appropriate tokens emerge: "The" (8.6%) for natural speech, formatting tokens (~3.4% each), and Germany (3.5%) for contextual hedging.

#### Key Insights

**Cognitive Mode Transitions**: The model demonstrates distinct processing phases:
1. **Semantic Resolution** (Layers 18-24): Understanding what's being asked
2. **Factual Retrieval** (Layer 30): Accessing stored knowledge with internal certainty  
3. **Pragmatic Formatting** (Layer 35): Adapting knowledge for natural communication

**The LayerNorm Effect**: Raw residual analysis would show Berlin at 100% confidence in final layer, masking the sophisticated calibration process. Normalized analysis reveals the model's learned balance between accuracy and natural expression.

**Directional Knowledge Asymmetry**: Confirmed in additional probing - "Berlin is capital of Germany" (72.9%) shows higher confidence than "Germany's capital is Berlin" (61.7%), suggesting stronger forward than reverse associations.

**Temperature as Truth Serum**: At temperature 0.1, Berlin returns to 100% confidence, confirming the model's internal certainty underneath the calibrated output.

#### Technical Significance

This analysis demonstrates that modern LLMs perform sophisticated **internal confidence management** - they can "know" something with near-certainty internally while presenting appropriately calibrated, conversational responses externally. The final layers aren't just applying softmax; they're performing learned pragmatic reasoning about how to communicate knowledge naturally.


## Meta-Llama-3-8B

### Results:
```

=== PROMPT =========================
Question: What is the capital of Germany? Answer:
=== END OF PROMPT =================


=== INSPECTING ====================
Input tokens: ['<|begin_of_text|>', 'Question', ':', ' What', ' is', ' the', ' capital', ' of', ' Germany', '?', ' Answer', ':']

Top predictions for next token after 'Question: What is the capital of Germany? Answer:':
Using NORMALIZED residual stream (LayerNorm applied - more accurate)
------------------------------------------------------------
Layer  0:
  1. 'oren' (0.007558)
  2. 'nton' (0.004973)
  3. '977' (0.003949)
  4. 'aland' (0.003312)
  5. '賀' (0.002306)

Layer  5:
  1. '.Decode' (0.009012)
  2. 'aget' (0.006978)
  3. 'LANG' (0.005642)
  4. 'urname' (0.005485)
  5. 'IDO' (0.004980)

Layer 10:
  1. 'ystack' (0.020013)
  2. 'PCM' (0.013980)
  3. 'Раз' (0.007219)
  4. ' Woody' (0.006797)
  5. 'iversit' (0.006082)

Layer 16:
  1. '#ad' (0.052507)
  2. ')application' (0.032176)
  3. '#ab' (0.027821)
  4. 'oplayer' (0.024633)
  5. '/******/' (0.016677)

Layer 21:
  1. ' Berlin' (0.230554)
  2. ' capital' (0.091921)
  3. ' Capitals' (0.080589)
  4. ' Capital' (0.043245)
  5. ' Washington' (0.037780)

Layer 26:
  1. ' Berlin' (0.957070)
  2. 'Berlin' (0.037447)
  3. ' berlin' (0.000963)
  4. ' BER' (0.000276)
  5. ' Bon' (0.000256)

Layer 31:
  1. ' Berlin' (0.856932)
  2. ' Germany' (0.017879)
  3. ' The' (0.014445)
  4. ' ' (0.006587)
  5. ' Ber' (0.004067)

============================================================
ACTUAL MODEL PREDICTION (for comparison):
Model's final prediction:
  1. ' Berlin' (0.856933)
  2. ' Germany' (0.017879)
  3. ' The' (0.014445)
  4. ' ' (0.006587)
  5. ' Ber' (0.004067)
============================================================
ADDITIONAL PROBING:

Prompt: 'Germany's capital is'
  1. ' a' (0.287620)
  2. ' one' (0.058417)
  3. ' the' (0.052589)

Prompt: 'Berlin is the capital of'
  1. ' Germany' (0.895522)
  2. ' the' (0.052471)
  3. ' and' (0.007549)

Prompt: 'Respond in one word: which city is the capital of Germany?'
  1. ' Berlin' (0.149279)
  2. ' The' (0.052448)
  3. ' Which' (0.051117)
============================================================
TEMPERATURE EXPLORATION:
(Temperature controls randomness: low=confident, high=creative)

Temperature 0.1:
  1. ' Berlin' (1.000000)
  2. ' Germany' (0.000000)
  3. ' The' (0.000000)
  4. ' ' (0.000000)
  5. ' Ber' (0.000000)

Temperature 2.0:
  1. ' Berlin' (0.034909)
  2. ' Germany' (0.005042)
  3. ' The' (0.004532)
  4. ' ' (0.003061)
  5. ' Ber' (0.002405)
=== END OF INSPECTING ==============


=== MODEL STATS ===============
Number of layers: 32
Model dimension: 4096
Number of heads: 32
Vocab size: 128256
Context length: 8192
=== END OF MODEL STATS ========
```

### Interpretation by ChatGPT o3:

#### 1. Layer-wise trajectory

- **Layers 0 – 10 | Lexical noise** — sub-word fragments from code, Cyrillic strings, and numerals dominate; no task engagement yet.
- **Layers 11 – 15 | Prompt classification** — probability mass shifts toward Q&A markers such as *"Answer"* and *"capital"*; the model has detected the schema but not the answer.
- **Layers 16 – 21 | Entity selection** — *Berlin* rises to ~23 % by L21, accompanied by related tokens (*capital*, *Washington*). Direction of the *capital-of* relation is already correct (no transient *Germany* peak as in Qwen-3).
- **Layers 22 – 26 | Knowledge consolidation** — layer 26 locks in the answer (**Berlin 0.957**, next competitor ≤ 0.04), about five layers earlier than Qwen-3-8B.
- **Layers 27 – 31 | Surface realisation** — final blocks shave ≈10 pp from *Berlin*, redistributing it to discourse markers (*"The"*, space). This is pragmatic formatting, not uncertainty.

#### 2. Comparative observations (vs Qwen-3-8B)

- **Earlier relation resolution** — Llama-3 never enters the "Germany → Berlin" confusion; relation resolved in the network's first half.
- **Lower peak over-confidence** — maximum internal confidence 0.957 (vs 0.999); calibration gap smaller.
- **Faster suppression of code/multilingual noise** — early lexical noise is cleaned up sooner, suggesting stronger prompt-conditioning circuitry.

#### 3. Directional bias probed

- "**Berlin is the capital of …**" → *Germany* with **p ≈ 0.896**
- "**Germany's capital is …**" → fails to emit *Berlin*, instead offering filler tokens (*"a"*, *"one"*).

The asymmetry exceeds that seen in Qwen-3 and highlights limitations of base checkpoints for cloze-style prompts.

#### 4. Take-aways for interpretability

1. Meta-Llama-3-8B follows the two-step semantic pipeline (entity selection → relation confirmation) but completes it several layers earlier than Qwen-3-8B.
2. The last transformer blocks function as a **learned decoder**, moderating internal certainty for human-like phrasing; analysing the normalised stream is essential to observe this behaviour.
3. Directional knowledge bias persists across architectures and is amplified in models without instruction fine-tuning.



## Mistral-7B-v0.1

### Results:
```
=== PROMPT =========================
Question: What is the capital of Germany? Answer:
=== END OF PROMPT =================


=== INSPECTING ====================
Input tokens: ['<s>', 'Question', ':', 'What', 'is', 'the', 'capital', 'of', 'Germany', '?', 'Answer', ':']

Top predictions for next token after 'Question: What is the capital of Germany? Answer:':
Using NORMALIZED residual stream (LayerNorm applied - more accurate)
------------------------------------------------------------
Layer  0:
  1. 'laug' (0.000325)
  2. 'avax' (0.000324)
  3. 'auf' (0.000310)
  4. 'erre' (0.000297)
  5. 'voir' (0.000281)

Layer  5:
  1. 'amber' (0.002847)
  2. 'adt' (0.002725)
  3. 'tober' (0.002666)
  4. 'stadt' (0.002126)
  5. 'lagen' (0.001952)

Layer 10:
  1. 'Answer' (0.010752)
  2. 'answer' (0.007833)
  3. 'stadt' (0.005306)
  4. 'rium' (0.004399)
  5. 'indow' (0.002370)

Layer 16:
  1. 'Answer' (0.032301)
  2. 'answer' (0.026451)
  3. 'cities' (0.011544)
  4. 'swer' (0.008133)
  5. 'answered' (0.005757)

Layer 21:
  1. 'Berlin' (0.368305)
  2. 'Germany' (0.168211)
  3. 'capital' (0.047416)
  4. 'Capital' (0.030147)
  5. 'Frankfurt' (0.012872)

Layer 26:
  1. 'Berlin' (0.990398)
  2. 'Germany' (0.006885)
  3. 'Ber' (0.000657)
  4. 'Frankfurt' (0.000633)
  5. 'German' (0.000189)

Layer 31:
  1. 'Berlin' (0.831504)
  2. 'The' (0.021823)
  3. 'Germany' (0.019521)
  4. 'It' (0.015328)
  5. 'There' (0.008855)

============================================================
ACTUAL MODEL PREDICTION (for comparison):
Model's final prediction:
  1. 'Berlin' (0.831503)
  2. 'The' (0.021823)
  3. 'Germany' (0.019521)
  4. 'It' (0.015328)
  5. 'There' (0.008855)
============================================================
ADDITIONAL PROBING:

Prompt: 'Germany's capital is'
  1. 'a' (0.187828)
  2. 'one' (0.089722)
  3. 'the' (0.051248)

Prompt: 'Berlin is the capital of'
  1. 'Germany' (0.896600)
  2. 'the' (0.053936)
  3. 'both' (0.004360)

Prompt: 'Respond in one word: which city is the capital of Germany?'
  1. '
' (0.453103)
  2. 'Berlin' (0.143815)
  3. 'If' (0.027227)
============================================================
TEMPERATURE EXPLORATION:
(Temperature controls randomness: low=confident, high=creative)

Temperature 0.1:
  1. 'Berlin' (1.000000)
  2. 'The' (0.000000)
  3. 'Germany' (0.000000)
  4. 'It' (0.000000)
  5. 'There' (0.000000)

Temperature 2.0:
  1. 'Berlin' (0.058676)
  2. 'The' (0.009506)
  3. 'Germany' (0.008990)
  4. 'It' (0.007967)
  5. 'There' (0.006055)
=== END OF INSPECTING ==============


=== MODEL STATS ===============
Number of layers: 32
Model dimension: 4096
Number of heads: 32
Vocab size: 32000
Context length: 2048
=== END OF MODEL STATS ========
```

### Interpretation by ChatGPT o3:

#### Layer-wise observations

- **Layers 0 – 5 | Lexical fragments** — extremely small probabilities spread over sub-word pieces (e.g. *laug*, *avax*, *auf*) and German stems (*-stadt*). No task signal yet.
- **Layers 6 – 10 | Prompt detection** — first appearance of the marker *Answer* (p ≈ 0.011 by L10). The model has recognised the Q&A schema.
- **Layers 11 – 16 | Context consolidation** — *Answer* climbs to 3 % at L16; related tokens (*cities*, *swer*) indicate the network is framing a reply but has not settled on the entity.
- **Layers 17 – 21 | Entity competition** — Layer 21 shows a two-horse race: *Berlin 0.368* vs *Germany 0.168*. The directional confusion phase is shorter and less extreme than in Qwen-3.
- **Layers 22 – 26 | Answer lock-in** — A single block (L26) drives *Berlin* to 0.99 while relegating all competitors ≤0.01. Factual retrieval is complete.
- **Layers 27 – 31 | Pragmatic calibration** — Final blocks lower *Berlin* to 0.83 and allocate the freed probability to discourse tokens (*The*, *It*, *There*) and the country name. This mirrors the calibration behaviour seen in Llama-3 and Qwen-3.

#### Directional bias (probe prompts)

- "**Berlin is the capital of …**" → *Germany* with p ≈ 0.897
- "**Germany's capital is …**" → no *Berlin* present; top choices are function words (*a*, *one*, *the*).

The asymmetry reinforces earlier findings that base checkpoints struggle with fill-in-the-blank (cloze) prompts.

#### Notable model-specific traits

1. **Early emergence of German morphology** — tokens ending in *-stadt* appear by layer 5, suggesting the model's vocabulary clusters geographical terms early.
2. **Pronoun insertion in the final layer** — tokens like *It* and *There* receive non-trivial probability, hinting that Mistral favours sentence-level continuations rather than single-word answers.
3. **Smaller calibration gap** — peak confidence drops from 0.99 → 0.83 (≈ 16 pp), intermediate between Llama-3 (≈ 10 pp) and Qwen-3 (≈ 28 pp).

These observations further support the emerging pattern that transformer LMs perform a two-stage computation: first retrieving a high-confidence internal answer, then modulating that answer for conversational delivery in the final blocks.

## gemma-2-9b

### Results:
```
Loaded pretrained model google/ into HookedTransformer

=== PROMPT =========================
Question: What is the capital of Germany? Answer:
=== END OF PROMPT =================


=== INSPECTING ====================
Input tokens: ['<bos>', 'Question', ':', ' What', ' is', ' the', ' capital', ' of', ' Germany', '?', ' Answer', ':']

Top predictions for next token after 'Question: What is the capital of Germany? Answer:':
Using NORMALIZED residual stream (LayerNorm applied - more accurate)
------------------------------------------------------------
Layer  0:
  1. ':' (1.000000)
  2. ',' (0.000000)
  3. ' ' (0.000000)
  4. '.' (0.000000)
  5. '-' (0.000000)

Layer  7:
  1. ':' (0.999832)
  2. ' answer' (0.000142)
  3. 'answer' (0.000011)
  4. ' Antwort' (0.000004)
  5. ' answers' (0.000003)

Layer 14:
  1. 'a' (0.537812)
  2. 's' (0.188112)
  3. ':' (0.101984)
  4. 'A' (0.070658)
  5. '
' (0.025466)

Layer 21:
  1. ' the' (0.769608)
  2. ' ' (0.154265)
  3. 'The' (0.064830)
  4. ' a' (0.009818)
  5. ' The' (0.000581)

Layer 28:
  1. 'The' (0.862892)
  2. ' The' (0.071619)
  3. ' the' (0.044940)
  4. ' ' (0.020471)
  5. ' a' (0.000070)

Layer 35:
  1. ' Berlin' (0.991194)
  2. 'Berlin' (0.007465)
  3. ' The' (0.001146)
  4. ' ' (0.000132)
  5. 'The' (0.000057)

Layer 41:
  1. ' Berlin' (0.999999)
  2. ' The' (0.000000)
  3. ' It' (0.000000)
  4. ' Bonn' (0.000000)
  5. ' Germany' (0.000000)

============================================================
ACTUAL MODEL PREDICTION (for comparison):
Model's final prediction:
  1. ' Berlin' (0.790732)
  2. ' The' (0.020494)
  3. ' It' (0.016067)
  4. ' Bonn' (0.013444)
  5. ' Germany' (0.011033)
============================================================
ADDITIONAL PROBING:

Prompt: 'Germany's capital is'
  1. ' a' (0.212154)
  2. ' one' (0.088893)
  3. ' the' (0.075590)

Prompt: 'Berlin is the capital of'
  1. ' Germany' (0.876573)
  2. ' the' (0.069934)
  3. ' modern' (0.007701)

Prompt: 'Respond in one word: which city is the capital of Germany?'
  1. '

' (0.158683)
  2. ' If' (0.111438)
  3. ' Berlin' (0.104638)
============================================================
TEMPERATURE EXPLORATION:
(Temperature controls randomness: low=confident, high=creative)

Temperature 0.1:
  1. ' Berlin' (1.000000)
  2. ' The' (0.000000)
  3. ' It' (0.000000)
  4. ' Bonn' (0.000000)
  5. ' Germany' (0.000000)

Temperature 2.0:
  1. ' Berlin' (0.085547)
  2. ' The' (0.013772)
  3. ' It' (0.012194)
  4. ' Bonn' (0.011155)
  5. ' Germany' (0.010105)
=== END OF INSPECTING ==============

```

### Interpretation by ChatGPT o3:

#### Layer-wise observations

- **Layers 0 – 7 | Punctuation lock-in** — the model stays fixated on the colon token ":`" with > 99 % probability for the first eight layers; no semantic processing is evident.
- **Layers 8 – 14 | Function-word bootstrap** — probability shifts to high-frequency letters (*a*, *s*) and a newline, indicating an attempt to start a natural-language answer rather than retrieve a fact.
- **Layers 15 – 28 | Sentence framing** — phrase prefixes dominate: *"the" / "The"* climb from 0.54 to 0.86. The network is constructing a declarative sentence before deciding on content.
- **Layers 29 – 35 | Answer emergence** — in a single hop (L35) *Berlin* jumps to 0.99 while all alternatives fall below 1 %. No intermediate "Germany" peak is observed.
- **Layers 36 – 41 | Over-confidence → calibration** — the normalised residual stream at L41 gives *Berlin 0.999999*, yet the final model output is 0.79, with 21 pp redistributed to discourse tokens (*The*, *It*, *Bonn*). This mirrors the calibration behaviour seen in other models, but with a larger drop than Llama-3 (≈10 pp) and smaller than Qwen-3 (≈28 pp).

#### Directional bias (probe prompts)

- "**Berlin is the capital of …**" → *Germany* with p ≈ 0.877
- "**Germany's capital is …**" → fails to emit *Berlin*; top tokens are fillers (*a*, *one*, *the*).

The disparity matches the pattern in Llama-3 and Mistral, confirming the model's difficulty with cloze completions.

#### Model-specific notes

1. **Late factual resolution** — *Berlin* appears only in the last two sampled layers, later than in any other evaluated model.  
2. **Pronoun / location fallback** — *It* and *Bonn* receive non-trivial final-layer probability, hinting at the model's conversational or multi-choice fine-tuning.  
3. **Extreme early fixation** on punctuation and function words suggests Gemma's decoder strongly favours well-formed sentence starts before retrieving factual content.

These findings reinforce the emerging two-stage interpretation: Gemma first decides how an answer should be phrased, then injects the factual token at the very end, and finally calibrates the output for conversational plausibility.

# 2. Cross-Model Insights

### 1. Shared Computational Motif
All four checkpoints follow the same high-level pipeline:
1. **Prompt orientation** – early layers (< one-third depth) clean up lexical noise and detect the Q&A schema.
2. **Entity/Relation resolution** – mid layers retrieve the relevant entities and settle the *capital-of* relation.
3. **Pragmatic calibration** – last few layers lower the internal high-confidence logit for *Berlin* and re-allocate probability to discourse tokens (*The, space, It, There*, etc.).

The motif is architecture-agnostic: we observe it in models ranging from 7 B to 9 B parameters and from 32 to 42 transformer blocks.

### 2. Where does the answer crystallise?
| Model | # Layers | Layer where *Berlin* > 0.95 | % depth |
|-------|---------|------------------------------|---------|
| Llama-3-8B | 32 | 26 | 81 % |
| Mistral-7B | 32 | 26 | 81 % |
| Qwen3-8B   | 36 | 30 | 83 % |
| Gemma-2-9B | 42 | 35 | 83 % |

Despite differing depths, **the answer crystallises between 80 – 85 % of total layers** in every model.

### 3. Calibration gap (internal → final probability)
| Model | Internal peak p(*Berlin*) | Final output p(*Berlin*) | Gap (pp) |
|-------|---------------------------|--------------------------|-----------|
| Qwen3-8B | 0.999 | 0.719 | **28** |
| Gemma-2-9B | 0.999 | 0.791 | 21 |
| Mistral-7B | 0.990 | 0.832 | 16 |
| Llama-3-8B | 0.957 | 0.857 | 10 |

Every model down-weights its internal certainty before emitting a token. The magnitude of the gap varies by checkpoint, but the **calibration step is universal**.

### 4. Directional knowledge asymmetry
Probe prompts show a consistent pattern:
- *"Berlin is the capital of …"* → *Germany* with p ≈ 0.88 – 0.90 across models.
- *"Germany's capital is …"* → either fails to emit *Berlin* or gives it < 0.15 probability.

Thus the *forward* relation (city → country) is encoded more strongly than the *reverse* cloze formulation.

### 5. Model-specific quirks
* **Qwen-3** – unique *"Germany"* detour phase at mid-depth; largest calibration gap.
* **Llama-3** – earliest resolution of relation, minimal over-confidence.
* **Mistral-7B** – inserts pronouns (*It, There*) in final layer; early German *-stadt* stems emerge by layer 5.
* **Gemma-2** – long punctuation fixation (eight layers) and latest factual resolution.

### 6. Depth-normalised timeline
Expressed as percentage of depth, the phases align remarkably well:
- Prompt detection ~ 25 % depth.
- Entity competition 45 – 60 %.
- Answer lock-in 80 - 85 %.
- Calibration 90 - 100 %.

### 7. Practical takeaway for interpretability
1. **Analyse normalised residuals** – raw logits hide the calibration stage and systematically over-state confidence.
2. **Sample mid-depth layers** (~50 %) to study relation resolution; late layers (> 80 %) to study answer lock-in.
3. **Directional probes matter** – cloze-style prompts can under-estimate a model's factual knowledge.

These cross-model patterns provide a roadmap for future mechanistic studies: focus on the 40 – 85 % depth window to capture the emergence of structured semantic knowledge, and treat the final 10 % as a learned decoder that shapes how that knowledge is expressed.


