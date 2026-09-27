# Physics datasets

`physics_questions_234.json` is the only BanglaBERT training dataset. It contains
18 Bengali questions from each of the 13 chapters of the Classes 9–10 Physics
textbook. A training row contains the question, chapter ID/title, and chapter
page range. It contains no answer, answer span, explanation, or answer passage.

The Colab notebook trains BanglaBERT as a 13-label chapter classifier:

`notebooks/train_banglabert_physics_qa_colab.ipynb`

The notebook rejects answer-bearing fields before training and uses a
chapter-balanced split of 15 training and 3 validation questions per chapter.
The exported checkpoint is not an answer generator and is not used as a source
of answer text.

`physics_qa_234.json` is retained only as a legacy reviewed evaluation/reference
artifact. It must not be uploaded to the training notebook or loaded by the web
application. Runtime answers are generated from retrieved textbook chunks in
the chapter explicitly selected by the user.

Regenerate the question-only dataset with:

```bash
python scripts/build_physics_question_dataset.py
```
