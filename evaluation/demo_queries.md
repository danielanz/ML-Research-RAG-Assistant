# Demo Queries for ML Research RAG Assistant

These 10 queries demonstrate the different capabilities of the RAG assistant.
Test them after indexing papers to verify the system works correctly.

---

## Q&A Mode (General Questions)

### 1. Basic Factual Question
**Query:** "What is the main contribution of the Adam optimizer?"

**Expected behavior:** Returns a concise answer citing specific chunks with page numbers.

---

### 2. Technical Detail Question
**Query:** "How does Adam compute the bias-corrected moment estimates?"

**Expected behavior:** Should retrieve algorithm details and explain the bias correction formula.

---

### 3. Hyperparameter Question
**Query:** "What default hyperparameter values does Adam recommend?"

**Expected behavior:** Should find and cite the recommended values (β1=0.9, β2=0.999, ε=10^-8).

---

## Compare Mode

### 4. Method Comparison
**Query:** "Compare Adam vs RMSprop optimizers"

**Expected behavior:** Router detects "compare" mode. Produces structured comparison with similarities/differences.

---

### 5. Approach Comparison
**Query:** "What are the differences between momentum and adaptive learning rate methods?"

**Expected behavior:** Structured comparison even when specific papers aren't named.

---

## Method Card Mode

### 6. Method Summary
**Query:** "Summarize the method and architecture of the Adam optimizer"

**Expected behavior:** Returns JSON-formatted method card with problem, key idea, training objective, etc.

---

### 7. Pipeline Overview
**Query:** "Explain the training objective and algorithm pipeline"

**Expected behavior:** Structured summary of the core algorithmic approach.

---

## Claim Verification Mode

### 8. True Claim
**Query:** "Is it true that Adam uses exponential moving averages for both first and second moments?"

**Expected behavior:** Returns "SUPPORTED" with citations to evidence.

---

### 9. Potentially False Claim
**Query:** "Verify: Adam always converges faster than SGD"

**Expected behavior:** May return "REFUTED" or "NOT_ENOUGH_EVIDENCE" depending on paper content.

---

### 10. Evidence Check
**Query:** "Does the paper claim that Adam works well for sparse gradients?"

**Expected behavior:** Verifies against paper evidence, returns verdict with rationale.

---

## Testing Abstention

### Bonus: Out-of-Scope Question
**Query:** "What is the capital of France?"

**Expected behavior:** Should abstain with "I cannot find evidence in the provided papers to answer that."

---

## How to Run

```bash
# Start the Streamlit app
uv run streamlit run app.py

# Or test programmatically
uv run python -c "
from src.pipeline import answer_question
r = answer_question('What is the main contribution of Adam?')
print(f'Mode: {r.mode}')
print(f'Abstained: {r.abstained}')
print(f'Answer: {r.answer[:500]}')
print(f'Citations: {len(r.cited_chunks)}')
"
```

