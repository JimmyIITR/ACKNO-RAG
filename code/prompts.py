from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List

load_dotenv()

def getSplitClaimPrompt(paragraph: str) -> List[str]:
    systemPrompt = """You are a fact extraction expert. Analyze the given text and split it into atomic claims following these rules:
    1. Each claim must be a complete, standalone fact
    2. Resolve all pronouns (he/she/they/it) to their explicit references
    3. Separate compound sentences into individual facts
    4. Maintain original meaning and entities
    5. Output MUST be valid JSON format: {"claims": ["claim1", "claim2"]}

    Example Input: "John works at Google and he developed their AI system"
    Example Output: {"claims": ["John works at Google", "John developed Google's AI system"]}

    Now process this:"""

    userPrompt = f"{paragraph}\n\nONLY OUTPUT JSON, NO MARKDOWN, NO EXTRA TEXT"

    return [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt}
    ]

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


template = """Answer the question based only on the following context:
{context}
Question: {question}
Use natural language and be concise.
Answer:"""