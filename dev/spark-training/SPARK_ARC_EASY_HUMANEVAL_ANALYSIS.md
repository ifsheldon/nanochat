# Side-by-Side Answer & Failure Analysis: `ARC-Easy` and `HumanEval`

Recorded on November 18, 2026.
Related tracking: [Issue #1 (Evaluation of SFT checkpoints)](https://github.com/ifsheldon/nanochat/issues/1), [Issue #2 (Uncropped best-fit buffering)](https://github.com/ifsheldon/nanochat/issues/2), [`SPARK_D24.md`](SPARK_D24.md), [`SPARK_RLM.md`](SPARK_RLM.md), and [`RLM_LONG_DOCUMENTS.md`](../RLM-architecture/RLM_LONG_DOCUMENTS.md).

This report inspects the actual predictions, logit distributions, and generated completions of **Base `spark-d24` SFT (`step 574`)** ([`scripts/chat_eval.py`](../../scripts/chat_eval.py)) and **RLM `nanochat-rlm` SFT (`step 504`)** ([`scripts/rlm_chat_eval.py`](../../scripts/rlm_chat_eval.py)) across all **2,376 `ARC-Easy`** test questions and all **164 `HumanEval`** test problems.

---

## 1. `ARC-Easy` (`N = 2,376`: Base `64.18%` vs. RLM `59.05%`)

Every `ARC-Easy` prompt is between **`43` and `213` tokens** (median `71`), so RLM processes 100% of `ARC-Easy` inside **Cycle 1 (`memory = None`)** without any `1,023`-token chunk boundary or latent-memory handoff. In `73 / 75` greedy free-form generation checks, both models also emit the bare choice token directly rather than prose.

### 1.1 The 2×2 Matrix (`[Do Well, Do Badly] × [Semantically Same, Different]`)

| Performance Axis | Semantically Same Answer (`1,537 / 2,376` = `64.69%`) | Semantically Different Answers (`839 / 2,376` = `35.31%`) |
| --- | --- | --- |
| **Do Well** *(Correct science / eliminative reasoning)* | **Both Correct — Same Option: `1,148` (`48.32%`)**<br>• High-confidence shared elementary science facts (organ systems, phase changes, direct measurement design, food webs, climate effects).<br>• Both place `~99.98%` median vocabulary mass directly on the valid choice tokens. | **Asymmetric Wins — One Correct, One Wrong: `632` (`26.60%`)**<br>• **Base Right / RLM Wrong: `377` (`15.87%`)** — RLM struggles more on lab-tool selection (`-9.11 pp`), quantitative/unit problems (`-13.46 pp`), and basic matter/resource taxonomy (`-6.43 pp`), plus `1/2/3/4` digit-labeled items (`vocab_mass = 0.676` vs. Base `0.975`). In **`71.4%`** of these `377` items, RLM ranked the gold option **#2**.<br>• **RLM Right / Base Wrong: `255` (`10.73%`)** — RLM beats Base on **negation/exception** prompts (`"not"`, `"except"`, `"least"`: `50.00%` vs. `44.87%`, `+5.13 pp`) and resists surface word-overlap traps where Base stumbles. In **`71.8%`** of these `255` items, Base ranked gold **#2**. |
| **Do Badly** *(Misconceptions, lexical traps, formula errors)* | **Both Wrong — Identical Distractor: `389` (`16.37%`)**<br>• Conditional agreement on the exact same wrong choice is **`65.3%`** (`389 / 596`), nearly **2× random chance (`33.3%`)**.<br>• Driven by four shared 1.4B traps: (1) echoing words from the question stem, (2) keyword association ignoring physical mechanism, (3) inverting subject/object relations, and (4) shared folk-science misconceptions. | **Both Wrong — Different Distractors: `207` (`8.71%`)**<br>• Low-confidence multi-step arithmetic, unit ratios, or quantitative recall where both models spread probability flatly (`0.30–0.50`) across choices and guess different wrong operations or magnitudes. |

### 1.2 Concrete `ARC-Easy` Examples Across All Four Quadrants

#### Quadrant 1: Do Well × Semantically Same (`1,148` items)

- **`#1` (Experimental Design):** *"A hiker wants to know if air is warmer in a forest than in the nearby farm field. Which activity would best help...?"*
  - **Gold / Both Pick `D`:** `recording the temperatures at both locations` (Base `p = 0.988`, RLM `p = 0.977`).
- **`#12` (Human Biology):** *"Oxygen-rich blood returns from the lungs to the heart. Which two systems are most directly involved?"*
  - **Gold / Both Pick `D`:** `respiratory and circulatory` (Base `p = 0.792`, RLM `p = 0.871`).
- **`#11` (Earth Systems):** *"As the amount of carbon dioxide in the atmosphere increases, which occurrence is most likely a long-term result?"*
  - **Gold / Both Pick `A`:** `The polar ice caps will melt.` (Base `p = 0.869`, RLM `p = 0.788`).

#### Quadrant 2A: Asymmetric — Base Well / RLM Badly (`377` items)

- **`#590` (Lab Tool Selection):** *"Which tool would be most useful in studying the living organisms the students collected from the pond?"*
  - **Gold / Base `C` (`p = 0.987`):** `microscope` vs. **RLM `B` (`p = 0.919`):** `pH meter` (over-associating `"pond water"` with `pH`).
- **`#1088` (`1/2/3/4` Label & Taxonomy):** *"All matter is made up of"* (`1=cells, 2=atoms, 3=molecules, 4=compounds`)
  - **Gold / Base `2` (`p = 0.990`):** `atoms` vs. **RLM `4` (`p = 0.527`):** `compounds`.
- **`#2236` (Resource Classification):** *"Which of the following is best classified as a nonrenewable resource?"*
  - **Gold / Base `B` (`p = 0.968`):** `Aluminum` vs. **RLM `A` (`p = 0.912`):** `Grass` (inverted renewable vs. nonrenewable).
- **`#74` (`1/2/3/4` Format Leakage in Generative Probe):** On numeric choice labels `['1','2','3','4']`, RLM picked gold `2` under logit argmax, but when decoded unconstrained it wrote full prose (`"Oxygen is released during photosynthesis..."`) because MMLU SFT only supervises `A/B/C/D`.

#### Quadrant 2B: Asymmetric — RLM Well / Base Badly (`255` items)

- **`#676` (Resisting Lexical Echo):** *"A serotinous cone is a seed enclosure that requires intense heat to open. When would a population of trees most benefit...?"*
  - **Gold / RLM `C` (`p = 0.914`):** `after a forest fire` vs. **Base `B` (`p = 0.683`):** `after a heat wave` (Base echoed `"intense heat"` → `"heat wave"`).
- **`#154` (Chemical Formula Reading):** *"Which chemical is a compound?"*
  - **Gold / RLM `B` (`p = 0.932`):** `hydrochloric acid (HCl)` vs. **Base `C` (`p = 0.728`):** `hydrogen (H2)` (confusing diatomic element `H2` with a compound).
- **`#2083` (Astronomy Cause/Effect):** *"Which of the following is a direct result of Earth's tilt on its axis?"*
  - **Gold / RLM `B` (`p = 0.990`):** `seasons` vs. **Base `A` (`p = 0.823`):** `tides`.
- **`#343` (Earth Processes):** *"How does sedimentary rock form?"*
  - **Gold / RLM `D` (`p = 0.893`):** `sediments compress` vs. **Base `C` (`p = 0.916`):** `sediments separate`.

#### Quadrant 3: Do Badly × Semantically Same (`389` items)

- **Prompt-Word Echo (`#1672`):** *"Scientists found large areas of fossilized wood in a desert in Arizona... what was the ancient environment?"*
  - Gold: `A (forest)` — **Both pick `B (desert)`** with extreme confidence (Base `0.989`, RLM `0.984`).
- **Keyword Shortcut (`#1093` & `#673`):**
  - *"Water freezes between cracks in rocks"* → Gold `weathering`, **both pick `crystallization`** (Base `0.977`, RLM `0.983`).
  - *"Warm cold hands by rubbing them together"* → Gold `friction`, **both pick `conduction`** (Base `0.999`, RLM `0.929`).
- **Relation Inversion (`#1379` & `#583`):**
  - *"Which of these revolves around a planet?"* → Gold `a moon`, **both pick `a star`** (Base `0.993`, RLM `0.955`).
  - *"Development of a human embryo"* → Gold `A single cell becomes many cells`, **both pick `Many cells combine to form one cell`** (Base `0.958`, RLM `0.970`).
- **Folk Misconception (`#535`):** *"What causes the different colors of the stars?"* → Gold `surface temperature`, **both pick `the gases that compose the star`** (Base `0.988`, RLM `0.966`).

#### Quadrant 4: Do Badly × Semantically Different (`207` items)

- **`#134` (Wrong Formula Operation):** *"Resistance arm is 0.5 cm. Effort arm is 20 cm. What is the mechanical advantage?"* (`20 / 0.5 = 40`)
  - Gold: `D (40)` — **Base picks `C (19.5)`** (`20 - 0.5`, subtraction) while **RLM picks `A (0.025)`** (`0.5 / 20`, inverted ratio).
- **`#139` (Numeric Fact Uncertainty):** *"Amount of time needed for Earth to complete one revolution around the sun"*
  - Gold: `D (365 days)` — **Base picks `C (180 days)`** (`p = 0.512`) while **RLM picks `B (28 days)`** (`p = 0.431`, lunar month).
- **`#61` (Meteor Showers):** *"Why are these objects from space usually harmless?"*
  - Gold: `D (burn up high in Earth's atmosphere)` — **Base picks `B (closer to the sun than to Earth)`** while **RLM picks `C (made of soft, spongy materials)`**.

### 1.3 `ARC-Easy` Topic Breakdown

| Topic / Question Property | Count | Base `spark-d24` | RLM `nanochat-rlm` | Diff (`RLM − Base`) |
| --- | ---: | ---: | ---: | ---: |
| Earth & Space | 879 | `62.00%` | `57.45%` | `-4.55 pp` |
| Biology / Ecology | 778 | `67.48%` | `62.60%` | `-4.88 pp` |
| Physics / Energy | 767 | `64.41%` | `61.54%` | `-2.87 pp` |
| Chemistry / Matter | 529 | `61.81%` | `55.39%` | `-6.43 pp` |
| Scientific Inquiry / Lab | 395 | `66.08%` | `56.96%` | `-9.11 pp` |
| Quantitative / Units | 52 | `48.08%` | `34.62%` | `-13.46 pp` |
| **Negation / Exception (`not`, `except`, `least`)** | **78** | **`44.87%`** | **`50.00%`** | **`+5.13 pp`** |

---

## 2. `HumanEval` (`N = 164`: RLM `9.76%` vs. Base `2.44%`)

> [!IMPORTANT]
> **Key Empirical Finding from Reading All 164 Completions:**
> NanoChat's SFT mixture (`SmolTalk + MMLU×3 + GSM8K×4`) has **no dedicated code-completion dataset**—all Python supervision comes from `SmolTalk` chat turns, where a user prompt containing `def foo(...): """..."""` is usually an **already-written function** to *explain*, *demonstrate*, or *refactor*. Both models show this prior (`"Here's a more concise version of the function:"` appears in `25.6%` of Base and `19.5%` of RLM completions), **but Base frequently treats the prompt stub as an already-implemented function and outputs only an explanation or a `print(foo(...))` usage demo (`32 / 164 = 19.5%` missing `def <entry_point>` in block #1), whereas RLM almost always writes the actual `def <entry_point>` implementation (`160 / 164 = 97.6%`)**.
>
> In fact, on **`12` of the `13` problems where RLM passed and Base failed**, Base failed solely because it explained or called the function instead of implementing it in its first code block.

### 2.1 Structural & Execution Diagnostics Across All 164 Problems

| Metric / Outcome | Base `spark-d24` (`step 574`) | RLM `nanochat-rlm` (`step 504`) |
| --- | ---: | ---: |
| **Passed unit tests** | **`4 / 164` (`2.4%`)** | **`16 / 164` (`9.8%`)** |
| Emits runnable `def <entry_point>` in first extracted block | `132 / 164` (`80.5%`) | **`160 / 164` (`97.6%`)** |
| Never defines `def <entry_point>` (explains / calls imagined fn) | **`30 / 164` (`18.3%`)** | `3 / 164` (`1.8%`) |
| Puts `print(fn(...))` usage in block #1 and `def` in later block | `2 / 164` (`1.2%`) | `1 / 164` (`0.6%`) |
| Hits `512`-token ceiling without assistant-end stop token | `24 / 164` (`14.6%`) | `17 / 164` (`10.4%`) |
| Degenerate line-repetition loop to `512` tokens | `15 / 164` (`9.1%`) | `8 / 164` (`4.9%`) |
| `AssertionError` (callable fn ran & failed a test case) | `78 / 164` (`47.6%`) | `85 / 164` (`51.8%`) |
| `SyntaxError` (mostly docstring-echo `512`-tok truncation or raw prose) | `34 / 164` (`20.7%`) | `30 / 164` (`18.3%`) |
| `NameError: name '<entry_point>' is not defined` | **`20 / 164` (`12.2%`)** | `3 / 164` (`1.8%`) |
| Runtime exceptions inside `check()` (`TypeError`, `IndexError`, `RecursionError`, etc.) | `28 / 164` (`17.1%`) | `30 / 164` (`18.3%`) |

### 2.2 The 2×2 Matrix (`[Do Well, Do Badly] × [Semantically Same, Different]`)

| Performance Axis | Semantically Same Answers (`39 / 164` = `23.78%`) | Semantically Different Answers (`125 / 164` = `76.22%`) |
| --- | --- | --- |
| **Do Well** *(Correct, idiomatic Python passing all unit tests)* | **Both Pass — Same Implementation: `2` (`1.22%`)**<br>• `HumanEval/23` (`strlen`): identical `return len(string)`.<br>• `HumanEval/42` (`incr_list`): identical `return [x + 1 for x in l]`. | **Both Pass via Different Algorithm (`1`) + Asymmetric Wins (`14`): `15` (`9.15%`)**<br>• **Both Pass, Different Algorithm (`1`):** `HumanEval/60` (`sum_to_n`) — Base writes `sum(range(1, n + 1))` with `n < 1` guard; RLM writes closed-form `n * (n + 1) // 2`.<br>• **RLM Passes / Base Fails (`13`):** RLM writes concise, idiomatic one-liners or standard loops (`is_prime`, `greatest_common_divisor`, `remove_vowels`, `is_palindrome`, `unique`, `filter_integers`, etc.); Base explains/calls a non-existent function on `12 / 13` and makes an off-by-one `range(n)` error on `1 / 13` (`string_sequence`).<br>• **Base Passes / RLM Fails (`1`):** `HumanEval/55` (`fib`) — Base writes valid recursion; RLM writes `def fib(n): return fib(n)` (`RecursionError`). |
| **Do Badly** *(Failed tests, unrunnable wrappers, truncation)* | **Both Fail — Same Bug or Same Structural Trap: `37` (`22.56%`)**<br>• **`20` Functionally Equivalent Buggy Functions** (`7` AST-identical), caused by **function-name priming** overriding the docstring spec (`right_angle_triangle`, `unique_digits`, `largest_smallest_integers`, `all_prefixes`, `sort_array`, `sum_squares`).<br>• **`17` Jointly Unrunnable / Truncated Completions:** On long-docstring prompts (`numerical_letter_grade`, `valid_date`, `eat`, `hex_key`), both copy the prompt's 250–300 token docstring back into their output and hit `SyntaxError` (unterminated triple-quoted string or expression) at the `512`-token cap. | **Both Fail — Divergent Bugs or One Unrunnable: `110` (`67.07%`)**<br>• **`65` Both Callable, Different Buggy Logic:** Both write syntactically valid functions that fail on different edge cases, operators, or type assumptions (`below_threshold`, `match_parens`, `sum_squares` #142).<br>• **`28` Only RLM Callable vs. `17` Only Base Callable:** One model writes a runnable (though buggy) Python function while the other emits usage-only code, prose, or truncated syntax. |

### 2.3 Concrete `HumanEval` Code Comparisons Across All Four Quadrants

#### Quadrant 1: Both Pass — Same vs. Different Algorithms (`3` problems)

**Semantically Same (`HumanEval/42` `incr_list` & `HumanEval/23` `strlen`):**

```python
# Base spark-d24:
def incr_list(l):
    return [x + 1 for x in l]

# RLM nanochat-rlm:
def incr_list(l: list):
    return [x + 1 for x in l]
```

**Semantically Different Algorithm (`HumanEval/60` `sum_to_n`):**

```python
# Base spark-d24 (O(n) iteration + validation):
def sum_to_n(n: int) -> int:
    if n < 1:
        raise ValueError("n must be a positive integer")
    return sum(range(1, n + 1))

# RLM nanochat-rlm (O(1) Gauss formula):
def sum_to_n(n: int):
    return n * (n + 1) // 2
```

#### Quadrant 2A: RLM Passes / Base Fails (`13` problems)

**Pattern 1 (`12 / 13` cases): Base assumes the function already exists and writes a usage demo or prose walkthrough, while RLM implements it (`HumanEval/13` `greatest_common_divisor`):**

```python
# Base spark-d24 (first fenced block -> NameError: 'greatest_common_divisor' is not defined):
print(greatest_common_divisor(3, 5))   # Output: 1
print(greatest_common_divisor(25, 15)) # Output: 5

# RLM nanochat-rlm (PASSED):
def greatest_common_divisor(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a
```

**Pattern 2 (`HumanEval/48` `is_palindrome`): Base spirals into a 512-token prose repetition loop; RLM writes a 1-liner:**

```python
# Base spark-d24 (unfenced prose loop to 512 tokens -> SyntaxError):
# "- It then checks if the first and last characters are the same again. If they are, it's a palindrome." (x16)

# RLM nanochat-rlm (PASSED):
def is_palindrome(text: str) -> bool:
    return text == text[::-1]
```

**Pattern 3 (`HumanEval/15` `string_sequence`): Both implement, Base makes an off-by-one + variable-shadowing error:**

```python
# Base spark-d24 (AssertionError — misses n inclusive):
def string_sequence(n: int) -> str:
    return ' '.join(str(n) for n in range(n))

# RLM nanochat-rlm (PASSED):
def string_sequence(n: int) -> str:
    return ' '.join(str(n) for n in range(0, n + 1))
```

#### Quadrant 2B: Base Passes / RLM Fails (`1` problem — `HumanEval/55` `fib`)

Here RLM's `"Here's a more concise version"` refactoring prior backfires into self-recursion:

```python
# Base spark-d24 (PASSED):
def fib(n):
    if n <= 1:
        return n
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

# RLM nanochat-rlm (RecursionError — prose claims it used "the built-in fib function"):
def fib(n: int):
    return fib(n)
```

#### Quadrant 3: Both Fail — Identical Semantic Bugs (`20` callable + `17` truncated)

Both models repeatedly **ignore the docstring specification in favor of what the function name sounds like**:

```python
# 1. HumanEval/157 (right_angle_triangle): BOTH check valid triangle instead of a^2 + b^2 == c^2!
def right_angle_triangle(a, b, c):
    return a + b > c and a + c > b and b + c > a   # exact match in both models

# 2. HumanEval/104 (unique_digits — spec: numbers with NO EVEN digits): BOTH deduplicate the list!
def unique_digits(x):
    return sorted(set(x))                          # exact match in both models

# 3. HumanEval/136 (largest_smallest_integers — spec: largest negative, smallest positive):
def largest_smallest_integers(lst):
    if not lst:
        return (None, None)
    return (max(lst), min(lst))                    # exact match in both models

# 4. HumanEval/14 (all_prefixes of one string): BOTH write LeetCode Longest-Common-Prefix!
for chars in zip(*string):
    if len(set(chars)) == 1: ...
```

#### Quadrant 4: Both Fail — Divergent Bugs (`110` problems)

Even when both write runnable functions (`65` pairs), their bugs often differ sharply:

```python
# HumanEval/52 (below_threshold):
# Base spark-d24 -> AssertionError (uses <= instead of <, and shadows list `l`):
def below_threshold(l, t):
    return all(l <= t for l in l)

# RLM nanochat-rlm -> IndexError (confuses element iteration `for i in l` with index lookup `l[i]`):
def below_threshold(l: list, t: int) -> bool:
    return all(l[i] < t for i in l)
```

```python
# HumanEval/119 (match_parens — input is a list of TWO paren strings to concatenate in some order):
# Both iterate over `lst` as if it were characters, then fail differently at return:
# Base: return 'Yes' if stack else 'No'   # inverted stack-empty condition!
# RLM : return 'Yes'                      # forgot to check `not stack`!
```

---

## 3. Strengths, Shortcomings, and Underlying Causes

### 3.1 Model-by-Model Profile

| Aspect | Base `spark-d24` (`step 574`) | RLM `nanochat-rlm` (`step 504`) |
| --- | --- | --- |
| **Strengths** | • **Stronger direct factual & terminology recall** on short multiple-choice questions (`ARC-Easy` `+5.13 pp`, `MMLU` `+1.50 pp`), especially lab instruments, basic chemistry definitions, and unit ratios.<br>• **Tighter multiple-choice label calibration**, including non-`A/B/C/D` label sets (`1/2/3/4` vocab mass `0.975` vs. `0.676`). | • **Much stronger code-generation task alignment** (`97.6%` emit `def <entry_point>` vs. `80.5%`), yielding **4× higher `HumanEval` pass rate (`9.76%` vs. `2.44%`)**.<br>• **Cleaner, more concise idiomatic Python** (`mean 272.8` vs. `289.3` tokens) and **~50% fewer degenerate 512-token repetition loops** (`8` vs. `15`).<br>• **Matches or beats Base on compositional / distractor-heavy prompts** (`ARC-Challenge` `-0.25 pp`, `GSM8K` `-0.23 pp`, `ARC-Easy` negation questions `+5.13 pp`). |
| **Shortcomings** | • **Severe prompt-intent confusion on bare code stubs:** treats `19.5%` of `HumanEval` prompts as existing functions to explain or demo rather than implement.<br>• **Higher repetition-loop rate** on open-ended prose/code explanations (`15 / 164`).<br>• Slightly more vulnerable to surface word-echo distractors on negation/trap MC questions. | • **Weaker surface science memorization & numeric-choice formatting** on easy recall items (`ARC-Easy`), frequently placing gold at **#2** (`71.4%` of misses).<br>• Occasional over-eager `"concise version"` hallucination (e.g., `def fib(n): return fib(n)`). |

**Shared Weaknesses Across Both Models:**

- **Function-name priming over docstring specs on `HumanEval`:** Both repeatedly implement what the function name *sounds* like rather than what the docstring specifies (`right_angle_triangle` checking triangle inequality, `unique_digits` deduplicating via `set(x)`, `largest_smallest_integers` returning `(max, min)`, `all_prefixes` implementing Longest Common Prefix).
- **Docstring re-copying & `512`-token truncation (`15–20 / 164`):** On long-docstring `HumanEval` prompts (`numerical_letter_grade`, `valid_date`, `eat`, `hex_key`), both re-emit the 250–300 token docstring inside ```` ```python ```` before the body and get cut off at `512` tokens with an unterminated string/block `SyntaxError`.
- **`SmolTalk` conversational refactoring bias:** Because SFT lacks a code-completion dataset, both treat bare `HumanEval` stubs as user-provided functions to critique or refactor (`"Here's a more concise version..."`).
- **Shared 1.4B science traps on `ARC-Easy`:** `65.3%` conditional agreement on the exact same wrong distractor via stem-word echoing (`#1672` `desert`), relation inversion (`#1379` `revolves around a planet → star`), and keyword shortcuts (`#1093` `freezes → crystallization`).

### 3.2 Potential Causes Behind the Divergence

1. **Why RLM is 4× stronger on `HumanEval` (`9.76%` vs. `2.44%`):**
   - **SFT row capacity & buffer clogging (`2,048` vs. `4,092` tokens, [Issue #2](https://github.com/ifsheldon/nanochat/issues/2)):** Multi-turn and code-containing `SmolTalk` conversations (`mean 865` tokens, `39% > 1,023` tokens) are much longer than `MMLU` (`384` tokens) or `GSM8K` (`174` tokens). In Base `spark-d24`'s `2,048`-token uncropped SFT packer, >`2,049`-token `SmolTalk` conversations clogged the 100-slot buffer early or were truncated mid-assistant-turn, skewing clean full-turn supervision toward short prose explanations and short `MMLU`/`GSM8K` items. RLM's `4,092`-token (`4 × 1,023`) episodes fit **`99.1%`** of all assistant tokens whole without mid-code truncation.
   - **Pretraining horizon per episode (`4,092` vs. `2,048` text tokens):** RLM pretraining packed `4,092`-token episodes (and supervised `<EOM>` continuations across chunks), exposing the model to longer contiguous source-file and docstring→implementation spans than `2,048`-token Base rows.
   - **SFT step count & late-epoch BOS padding (`574` vs. `504` steps):** Base SFT ran `14%` more optimizer updates over heavily BOS-padded `2,048`-token batches, which can reinforce high-frequency `SmolTalk` assistant openings (`"The function X takes..."`) over direct code synthesis.

2. **Why Base `spark-d24` leads on `ARC-Easy` (`+5.13 pp`) and `MMLU` (`+1.50 pp`) while tying `ARC-Challenge` and `GSM8K`:**
   - **Pretraining parameter-update budget & `wte` freezing:** Because `ARC-Easy` never crosses a `1,023`-token boundary, RLM gets zero benefit from latent memory there—while during pretraining RLM had to divide representation capacity between local token prediction and learning a differentiable `512`-slot summary/readout protocol (with `transformer.wte.weight` frozen for the first `90%` of pretraining). Pure single-hop textbook/lexical associations (`ARC-Easy`, `MMLU`) are most sensitive to raw parametric storage, whereas multi-cue distractor rejection (`ARC-Challenge`) and tool-assisted math (`GSM8K`) are nearly tied.
   - **`+70` extra SFT updates over `MMLU × 3`:** Base SFT's `574` updates (vs. `504`) gave stronger sharpening on single-token multiple-choice formatting—matching the observation that RLM already ranks the gold answer **#2 on `71.4%`** of the questions Base gets right and RLM misses, and loses `~32%` probability mass off digit labels on `1/2/3/4` questions.
