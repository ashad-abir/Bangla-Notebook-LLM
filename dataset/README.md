# Physics QA datasets

`physics_questions_234.json` contains 18 Bengali questions from each of the 13
chapters of the Classes 9–10 Physics textbook. Every row includes its chapter
classification label, title, and chapter page range.

`physics_qa_234.json` adds a reviewed curriculum answer and two same-chapter
distractor sentences to every question. `answers.answer_start` is the exact
character offset of `answers.text` inside that context, as required by
extractive BERT training.

Both files are generated from `raw/physics.md` with:

```bash
python scripts/build_physics_question_dataset.py
.venv/bin/python scripts/build_physics_qa_dataset.py
```

The QA generator reads the curated answer key in
`scripts/physics_answer_key.py`; it does not train or call a model locally. The
committed JSON can be used for Colab training without running either generator.

The Colab notebook validates all answer offsets and the chapter balance before
training:

`notebooks/train_banglabert_physics_qa_colab.ipynb`

The notebook intentionally refuses to train outside Google Colab or without a
GPU. Its final model is saved under
`MyDrive/pathshongi-banglabert-physics-qa`.
