from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..retrieval.retriever import search

MODEL_NAME = "Qwen/Qwen3-0.6B"


@lru_cache(maxsize=1)
def load_model():
    """Load the tokenizer and language model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using retrieved context."""
    tokenizer, model = load_model()
    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the provided context. "
                "If the context does not contain the answer, "
                "say you do not know."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt"
    ).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


def answer_question(question: str, k: int = 5):
    retrieved = search(question, k)
    context = ""
    sources = []

    for record in retrieved:
        context += record["text"] + "\n\n"
    answer = generate_answer(question, context)
    for document in retrieved:
        source = {}
        source["file_path"] = document["file_path"]
        source["first_character_index"] = document["first_character_index"]
        source["last_character_index"] = document["last_character_index"]
        sources.append(source)
    return {"answer": answer, "sources": sources}
