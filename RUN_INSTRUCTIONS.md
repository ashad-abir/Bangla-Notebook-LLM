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

The dependencies and models are free. The first semantic-index run downloads `multilingual-e5-small` from Hugging Face and caches it locally.

## 2. Build the index

Recommended hybrid index:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ingest
```

Quick dependency-free trial without an embedding model:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ingest --lexical-only
```

Lexical-only mode is useful for smoke testing, but hybrid retrieval is recommended for Bengali paraphrases and English questions.

## 3. Inspect retrieval before running an LLM

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag search "নিউটনের দ্বিতীয় সূত্র কী?" --debug
& ".\.venv\Scripts\python.exe" -m bangla_rag evaluate
```

Evaluation prints every evidence-gate decision. The current thresholds are conservative starter values measured with the bundled EmbeddingGemma model on the included Bengali/English and unrelated cases; they are not universal. Adjust thresholds in `config/settings.json` only after inspecting a larger set of answerable, paraphrased, English, and unrelated questions.

## 4. Start a free local answer model

Place a Qwen3-4B Q4 GGUF under `.runtime\models` and a recent `llama-server.exe` under `.runtime\llama.cpp`, or set `PATHSHONGI_LLAMA_SERVER` to the executable path. The launcher also starts a local EmbeddingGemma service for retrieval. Start both with:

```powershell
& ".\scripts\start-local-model.ps1"
```

Keep that terminal open. On the first run, `llama.cpp` downloads the free EmbeddingGemma GGUF into the local Hugging Face cache; later runs are offline. The application expects OpenAI-compatible answer and embedding endpoints at `http://127.0.0.1:8080/v1` and `http://127.0.0.1:8081/v1`. Change the corresponding settings in `config/settings.json` if your local servers differ.

## 5. Ask questions

### Web GUI (recommended)

For the easiest demonstration, double-click `Launch-Pathshongi.cmd` in the project folder. It starts every required local service, displays startup progress, and opens the app automatically. Double-click `Stop-Pathshongi.cmd` afterward to stop those services.

For manual startup, continue with the commands below.

Keep the model-server terminal open, open a second PowerShell terminal in the project, and run:

```powershell
& ".\scripts\start-gui.ps1"
```

Then open `http://127.0.0.1:8000` in your browser. Students can switch the interface between বাংলা and English, choose their class and subject, and then ask questions or make quizzes. Retrieval is restricted to the books matching that selection. The GUI also supports verified citations, grounded refusal, answer checking, explanations, and scoring.

### Command line

One question:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ask "নিউটনের দ্বিতীয় সূত্রটি ব্যাখ্যা করো।"
```

Interactive mode:

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag ask
```

The exact unsupported-question response is:

```text
এই বইয়ে এই প্রশ্নের উত্তর পাওয়া যায়নি।
```

## 6. Generate a quiz

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag quiz "বল ও নিউটনের সূত্র" --count 5 --output ".\quiz\newton.json"
```

Generated questions are accepted only when their source IDs came from the retrieved book passages. Teacher review is still recommended before classroom use.
Generate 1–5 questions per command. On this CPU-only machine, answer generation takes roughly one minute and a multi-question quiz can take several minutes.

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

4. Restart the GUI. The class and subject choices are generated from the catalog automatically; no HTML or JavaScript edit is required.

## 7. Diagnose the system

```powershell
& ".\.venv\Scripts\python.exe" -m bangla_rag doctor
```

## Training policy

No training is required for this baseline. Retrieval stores the textbook outside the model and is easier to update, cite, and restrict. If later evaluation shows a repeatable weakness in Bengali answer style or quiz formatting, use LoRA/QLoRA on Google Colab or Kaggle free GPU sessions. Fine-tuning must not replace retrieval or the evidence/refusal checks.
