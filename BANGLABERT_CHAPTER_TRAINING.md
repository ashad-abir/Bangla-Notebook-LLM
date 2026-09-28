# Train and import the chapter-wise BanglaBERT model

This project now uses BanglaBERT for one purpose only: classify a Bengali Physics
question into one of the 13 textbook chapters. BanglaBERT never receives an
answer, answer span, explanation, or textbook passage during training, and it
does not generate student answers. The chapter-scoped RAG pipeline and Qwen
remain responsible for reading the selected chapter chunks and answering or
refusing after inspecting that context.

One 13-label classifier is used instead of 13 separate checkpoints. It still
learns chapter-wise boundaries, while keeping training data balanced and local
inference small enough for the application.

## Files used

- Notebook: `notebooks/train_banglabert_chapter_classifier_colab.ipynb`
- Training data: `dataset/physics_questions_234.json`
- Dataset builder: `scripts/build_physics_question_dataset.py`
- Dataset validator: `scripts/validate_banglabert_chapter_dataset.py`
- Local model destination: `.runtime/models/pathshongi-banglabert-physics-chapters/`

The dataset contains 234 unique questions: 18 questions for each of 13
chapters. The notebook makes a deterministic split of 12 training, 3 validation,
and 3 untouched test questions per chapter (156/39/39 total).

## 1. Prepare and validate the data locally

From the repository root, regenerate the committed question-only dataset and
validate it:

```bash
python scripts/build_physics_question_dataset.py
python scripts/validate_banglabert_chapter_dataset.py
```

The validation output must say `"contains_answers": false`, `"questions": 234`,
and `"chapters": 13`. Do not add answer-like fields to this dataset. If more
training examples are added later, write new question phrasings and keep every
chapter balanced.

## 2. Train in Google Colab

1. Open Google Colab and upload
   `notebooks/train_banglabert_chapter_classifier_colab.ipynb`.
2. Choose **Runtime > Change runtime type > T4 GPU**.
3. Choose **Runtime > Run all**.
4. When prompted, upload `dataset/physics_questions_234.json`.
5. Let the notebook train. Early stopping selects the checkpoint with the best
   validation macro-F1 score.
6. Review the final `test_accuracy`, `test_macro_f1`, and confusion matrix. The
   test split is evaluated only after model selection.
7. The last cell downloads
   `pathshongi-banglabert-physics-chapters.zip`.

Training is reproducible: the split, model seed, and data seed are fixed. GPU
library differences can still cause small numerical differences between runs.
The dataset is deliberately small, so test metrics may vary noticeably and the
classifier should be treated as an optional routing aid—not proof that a
question is answerable.

## 3. Bring the trained model into this project

Do not commit the ZIP or extracted weights. `.runtime/` is ignored by Git.

### Windows PowerShell

Copy the downloaded ZIP to the repository root, then run:

```powershell
$destination = ".\.runtime\models\pathshongi-banglabert-physics-chapters"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Expand-Archive -LiteralPath ".\pathshongi-banglabert-physics-chapters.zip" `
  -DestinationPath $destination -Force
```

### Linux or macOS

Copy the downloaded ZIP to the repository root, then run:

```bash
mkdir -p .runtime/models/pathshongi-banglabert-physics-chapters
unzip -o pathshongi-banglabert-physics-chapters.zip \
  -d .runtime/models/pathshongi-banglabert-physics-chapters
```

The destination should contain these files directly, without another nested
folder:

```text
.runtime/models/pathshongi-banglabert-physics-chapters/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── training_metadata.json
└── vocab.txt or tokenizer.json
```

Depending on the Transformers version, additional tokenizer files may also be
present. `pytorch_model.bin` is accepted in place of `model.safetensors`.

## 4. Verify local inference

Install the normal project dependencies, then run:

```bash
python -m bangla_rag classify "নিউটনের দ্বিতীয় গতিসূত্র কী?" --top-k 3
python -m bangla_rag doctor
```

The first command should rank chapter 3 first. The doctor output should report
`PASS BanglaBERT chapter classifier (optional)`.

The classifier does not override the class, book, or chapter chosen by the
student and must not immediately reject a question. The application always
restricts retrieval to the user's selected chapter, sends that context to Qwen,
and lets Qwen decide whether the chapter supports an answer.

## 5. Before using a newly trained checkpoint

- Keep the untouched test split separate from training and model selection.
- Inspect per-chapter mistakes in the confusion matrix, not accuracy alone.
- Add paraphrased questions to weak chapters; never copy reference answers into
  training input.
- Re-run the repository tests with `python -m pytest -q`.
- Keep `training_metadata.json` with the model so its dataset hash, split sizes,
  and test metrics remain auditable.
