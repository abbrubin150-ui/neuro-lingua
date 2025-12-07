# Sample Scenario Suites

This directory contains pre-built validation scenarios for testing Neuro-Lingua models.

## Available Suites

| File | Language | Scenarios | Purpose |
|------|----------|-----------|---------|
| `en-basic.json` | English | 10 | Basic validation for English models |
| `he-basic.json` | Hebrew | 10 | Basic validation for Hebrew models |

## Schema

Each scenario file follows this structure:

```json
{
  "name": "Suite Name",
  "description": "Suite description",
  "language": "en|he|mixed",
  "version": "1.0.0",
  "scenarios": [
    {
      "id": "unique_id",
      "name": "Scenario Name",
      "prompt": "Input prompt",
      "expectedResponse": "Expected output (optional)",
      "category": "completion|coherence|quality|diversity|length|style|special",
      "difficulty": "easy|medium|hard",
      "notes": "Evaluation notes",
      "generationConfig": {
        "maxTokens": 20,
        "temperature": 0.8
      }
    }
  ],
  "evaluationCriteria": {
    "category": "How to evaluate this category"
  },
  "suggestedUse": "When to use this suite"
}
```

## Usage

### In Browser UI

1. Open Neuro-Lingua in browser
2. Navigate to **Scenarios** panel
3. Click **Import Scenarios**
4. Select the desired JSON file
5. Run scenarios against your trained model

### Programmatic (CLI)

You can reference scenarios in your training workflow:

```bash
# Train with specific scenarios for validation
EXPERIMENT_NAME="en-validation" \
SCENARIOS_PATH="data/scenarios/en-basic.json" \
pnpm train
```

## Categories

| Category | Purpose | Scoring |
|----------|---------|---------|
| `completion` | Test word/phrase completion | Expected word appears |
| `coherence` | Test grammatical/semantic coherence | Manual review |
| `quality` | Test output quality (no repetition) | Repetition count < threshold |
| `diversity` | Test vocabulary diversity | Unique outputs across runs |
| `length` | Test generation parameter respect | Token count matches config |
| `style` | Test style appropriateness | Manual review |
| `special` | Test special character handling | Correct processing |

## Creating Custom Scenarios

1. Copy an existing file as template
2. Modify scenarios for your use case
3. Use unique IDs (`myproject_category_N`)
4. Document evaluation criteria
5. Import into the UI

## Best Practices

1. **Run scenarios before and after training** to measure improvement
2. **Use consistent generation config** across comparisons
3. **Document expected behavior** in notes field
4. **Group related scenarios** by category
5. **Track scenario scores** in run history for comparison

## Scenario Difficulty Guidelines

- **Easy**: Common phrases, simple completion
- **Medium**: Context-dependent, multi-word output
- **Hard**: Domain-specific, style-sensitive

---

*Part of Neuro-Lingua DOMESTICA v3.2.4+*
