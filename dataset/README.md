# Physics datasets

`physics_questions_234.json` is the only BanglaBERT training dataset. It contains
18 Bengali questions from each of the 13 chapters of the Classes 9–10 Physics
textbook. A training row contains the question, chapter ID/title, and chapter
page range. It contains no answer, answer span, explanation, or answer passage.

The Colab notebook trains BanglaBERT as a 13-label chapter classifier:

`notebooks/train_banglabert_chapter_classifier_colab.ipynb`

The notebook rejects answer-bearing fields before training and uses a
chapter-balanced split of 12 training, 3 validation, and 3 untouched test
questions per chapter. The exported checkpoint is not an answer generator and
is not used as a source of answer text. See `BANGLABERT_CHAPTER_TRAINING.md` for
the complete training and import procedure.

Regenerate the question-only dataset with:

```bash
python scripts/build_physics_question_dataset.py
python scripts/validate_banglabert_chapter_dataset.py
```
