from types import SimpleNamespace
from threading import Lock

from fastapi.testclient import TestClient

from bangla_rag.models import Answer, Citation
from bangla_rag.runtime import get_runtime
from bangla_rag.webapp import create_app


class FakeLLM:
    def health(self):
        return True, "test-model"


class FakeQA:
    def ask(self, question, book_ids=None):
        assert book_ids == {"physics-9-10"}
        return Answer(
            "answered",
            f"উত্তর: {question}",
            (
                Citation(
                    "পদার্থবিজ্ঞান",
                    81,
                    ("chunk-81",),
                    "নিউটনের দ্বিতীয় সূত্রের প্রমাণ।",
                ),
            ),
        )


class FakeQuiz:
    def generate(self, topic, count, quiz_type="mcq", book_ids=None, page_range=None):
        assert book_ids == {"physics-9-10"}
        assert topic == "নিউটনের সূত্র জড়তা ভরবেগ বল"
        assert page_range == (67, 102)
        return {
            "status": "ok",
            "quiz_type": quiz_type,
            "topic": topic,
            "requested_count": count,
            "accepted_count": 1,
            "questions": [{"question": "নমুনা?", "options": {"A": "এক", "B": "দুই", "C": "তিন", "D": "চার"}, "correct_answer": "A"}],
        }


def fake_runtime():
    chunks = [SimpleNamespace(book_id="physics-9-10", page=index) for index in range(1, 4)]
    return SimpleNamespace(
        index=SimpleNamespace(chunks=chunks, vectors=object()),
        llm=FakeLLM(),
        qa=FakeQA(),
        quiz=FakeQuiz(),
        generation_lock=Lock(),
    )


def make_client():
    app = create_app()
    app.dependency_overrides[get_runtime] = fake_runtime
    return TestClient(app)


def test_home_and_health_are_available():
    client = make_client()
    home = client.get("/")
    health = client.get("/api/health")

    assert home.status_code == 200
    assert "পাঠসঙ্গী" in home.text
    assert health.json() == {
        "status": "ready",
        "model_ready": True,
        "model_detail": "test-model",
        "semantic_search": True,
        "chunks": 3,
        "books": 1,
        "pages": 3,
    }
    catalog = client.get("/api/catalog")
    assert catalog.status_code == 200
    assert [item["id"] for item in catalog.json()["classes"]] == ["8", "9-10"]
    subjects = {item["id"]: item for item in catalog.json()["subjects"]}
    assert subjects["bgs"]["book_ids"] == ["bgs-8"]
    assert subjects["physics"]["book_ids"] == ["physics-9-10"]
    physics = next(book for book in catalog.json()["books"] if book["id"] == "physics-9-10")
    assert len(physics["chapters"]) == 13
    assert physics["chapters"][2]["title_bn"] == "বল"


def test_ask_serializes_verified_citations():
    response = make_client().post(
        "/api/ask",
        json={"question": "সূত্র কী?", "book_ids": ["physics-9-10"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["citations"][0]["page"] == 81
    assert payload["citations"][0]["chunk_ids"] == ["chunk-81"]


def test_quiz_and_validation_contracts():
    client = make_client()
    response = client.post(
        "/api/quiz",
        json={"chapter_id": "3", "quiz_type": "mcq", "count": 10, "book_ids": ["physics-9-10"]},
    )
    invalid = client.post(
        "/api/quiz",
        json={"chapter_id": "3", "quiz_type": "mcq", "count": 9, "book_ids": ["physics-9-10"]},
    )
    invalid_chapter = client.post(
        "/api/quiz",
        json={"chapter_id": "99", "quiz_type": "knowledge", "count": 5, "book_ids": ["physics-9-10"]},
    )
    unknown = client.post(
        "/api/quiz",
        json={"chapter_id": "3", "quiz_type": "mcq", "count": 10, "book_ids": ["unknown"]},
    )

    assert response.status_code == 200
    assert response.json()["questions"][0]["correct_answer"] == "A"
    assert response.json()["chapter"]["title_bn"] == "বল"
    assert invalid.status_code == 400
    assert invalid_chapter.status_code == 400
    assert unknown.status_code == 400
