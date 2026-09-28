# Local Run Instructions (Windows)

All commands below are intended for PowerShell opened at the cloned repository root:

```powershell
Set-Location -LiteralPath "C:\path\to\Bangla-Notebook-LLM"
```

## 1. Install Python and dependencies

Install free Python 3.11 or 3.12 if `python` is not available. Then create a project-local environment:

```powershell
python -m venv ".venv"
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
```

The dependencies and models are free. CPU-only PyTorch is installed from the official PyTorch package index. The optional question-only BanglaBERT classifier can be placed under:

```text
.runtime/models/pathshongi-banglabert-physics-chapters/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

## 2. Build the index

Recommended hybrid index:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ingest
```

Quick embedding-free trial:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ingest --lexical-only
```

Lexical-only mode is useful for smoke testing, but hybrid retrieval is recommended for Bengali paraphrases and English questions.

## 3. Inspect retrieval before answering

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag search "নিউটনের দ্বিতীয় সূত্র কী?" --debug
& ".\.venv\Scripts\python.exe" -m bangla_rag evaluate
```

Evaluation prints retrieval-quality diagnostics. These thresholds help compare ranking behavior, but the answer API does not use them to reject a question before generation. Qwen reads the ranked selected-chapter passages and makes the grounded answer/refusal decision. Adjust thresholds only after inspecting a larger set of answerable, paraphrased, English, and unrelated questions.

## 4. Start optional retrieval and quiz services

For grounded answers and quizzes, place a Qwen3-4B Q4 GGUF under `.runtime\models` and a recent `llama-server.exe` under `.runtime\llama.cpp`, or set `PATHSHONGI_LLAMA_SERVER` to the executable path. Start the local services with:

```powershell
& ".\scripts\start-local-model.ps1"
```

Keep that terminal open. On the first run, `llama.cpp` downloads the free EmbeddingGemma GGUF into the local Hugging Face cache; later runs are offline. The application expects the answer/quiz and embedding endpoints at `http://127.0.0.1:8080/v1` and `http://127.0.0.1:8081/v1`. Exact-term lexical retrieval diagnostics continue to work if the embedding endpoint is offline.

## 5. Ask questions

### Web GUI (recommended)

For the easiest demonstration, double-click `Launch-Pathshongi.cmd` in the project folder. It starts every required local service, displays startup progress, and opens the app automatically. Double-click `Stop-Pathshongi.cmd` afterward to stop those services.

For manual startup, continue with the commands below.

If you started the optional model services, keep that terminal open. In a second PowerShell terminal, run:

```powershell
& ".\scripts\start-gui.ps1"
```

Then open `http://127.0.0.1:8000` in your browser. Students can switch the interface between বাংলা and English, choose their class, book, and chapter, and then ask questions or make quizzes. Retrieval is restricted to that exact chapter. The GUI also supports verified citations, grounded refusal, answer checking, explanations, and scoring.

### Command line

One question:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ask "নিউটনের দ্বিতীয় সূত্র কী?" --book physics-9-10 --chapter 3
```

Interactive mode:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ask --book physics-9-10 --chapter 3
```

The exact unsupported-question response is:

```text
এই অধ্যায়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।
```

## 6. Generate a quiz

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag quiz "বল ও নিউটনের সূত্র" --book physics-9-10 --chapter 3 --count 10 --output ".\quiz\newton.json"
```

Generated questions are accepted only when their source IDs came from the retrieved book passages. Teacher review is still recommended before classroom use.
Generate 10–25 MCQs or 5–10 short-answer questions per command. On this CPU-only machine, answer generation can take roughly one minute and a multi-question quiz can take several minutes.

## Add another textbook

If the textbook is already Markdown, put it under `dataset/raw/`. Keep the `## পৃষ্ঠা N` headings so citations can point to real pages.

To convert an NCTB PDF with the optional Surya OCR 2 module, install only its additional dependencies:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements-ocr.txt"
```

On this CPU-only Windows machine, point Surya at the already installed `llama-server` and the project-local model cache before each OCR session:

```powershell
$env:SURYA_INFERENCE_BACKEND = "llamacpp"
$env:LLAMA_CPP_BINARY = if ($env:PATHSHONGI_LLAMA_SERVER) {
    $env:PATHSHONGI_LLAMA_SERVER
} else {
    (Resolve-Path ".\.runtime\llama.cpp\llama-server.exe").Path
}
$env:LLAMA_CPP_NGL = "0"
$env:SURYA_INFERENCE_PARALLEL = "1"
$env:HF_HOME = "$PWD\.runtime\huggingface"
```

Convert the full PDF directly to parser-compatible Markdown:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_ocr ".\NCTB-Books\book.pdf" -o ".\dataset\raw\book.md" --dpi 300
```

For a quick validation, process a small page range first. Use a `.json` output path when region confidence and bounding boxes are needed:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_ocr ".\NCTB-Books\book.pdf" -o ".\ocr-smoke.json" --pages 1-3 --dpi 96
```

OCR runs one page at a time and displays live terminal progress. It does not automatically register or index the new book.

After conversion:

1. Check the OCR output, then keep the Markdown file under `dataset/raw/`.
2. Add a record to `config/books.json`. Give the book a unique `id`, bilingual title, one or more class records, and a bilingual subject record. Reuse the same subject `id` for books belonging to the same subject.
3. Rebuild the index:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ingest
```

4. Restart the GUI. The class, book, and chapter choices are generated from the catalog automatically; no HTML or JavaScript edit is required.

## 7. Diagnose the system

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag doctor
```

## Training and safety policy

The exported BanglaBERT checkpoint is trained only for 13-way chapter classification from question text and chapter labels. The training notebook rejects answer-bearing fields, and the web application never uses the checkpoint to supply answer text. Answers are generated only from chunks carrying the user-selected book and chapter IDs. Retrieval ranks those chunks; Qwen reads them before deciding whether to answer or refuse, and citations remain restricted to trusted source IDs.

Follow `BANGLABERT_CHAPTER_TRAINING.md` to validate the question-only dataset,
train in Colab, import the checkpoint, and test local chapter prediction.
