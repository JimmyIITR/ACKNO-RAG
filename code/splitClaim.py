import json
import re
import ollama
from typing import List

LLM_MODEL = "llama3.1"

def generateAtomicClaims(paragraph: str) -> List[str]:
    """
    Improved version using explicit JSON formatting and enhanced coreference resolution
    """
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

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            options={"temperature": 0.2}, 
            format="json"
        )
        
        try:
            result = json.loads(response['message']['content'])
            if isinstance(result, dict) and "claims" in result:
                return validateClaims(result["claims"], paragraph)
        except json.JSONDecodeError:
            pass
        
        jsonMatch = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
        if jsonMatch:
            result = json.loads(jsonMatch.group())
            return validateClaims(result.get("claims", []), paragraph)
        
    except Exception as e:
        print(f"API Error: {str(e)}")
    
    return []

def validateClaims(claims: List[str], original: str) -> List[str]:
    """Validate and clean generated claims"""
    validClaims = []
    for claim in claims:
        claim = claim.strip().replace('"', '')
        
        if not claim or len(claim) < 10:
            continue
            
        if re.search(r'\b(he|she|they|it|his|her|their)\b', claim, re.IGNORECASE):
            continue  
            
        validClaims.append(claim)
    
    if not validClaims:
        return [original]  
    
    return validClaims

def processParagraph(paragraph: str):
    """Enhanced processing with retries"""
    maxRetries = 3
    for attempt in range(maxRetries):
        claims = generateAtomicClaims(paragraph)
        if claims:
            print("\nValidated Atomic Claims:")
            for i, claim in enumerate(claims, 1):
                print(f"{i}. {claim}")
            return
        print(f"Retry {attempt+1}/{maxRetries}...")
    
    print("Failed to generate valid claims after retries")

if __name__ == "__main__":
    testParagraph = (
        "Karnataka Congress Govt collects ₹445 Cr from temples but gifts ₹330 Cr to mosques & churches that pay no tax! Congress - The new age Mughal invaders who are looting our temples"
    )
    p = (
        "Trump Administration claimed songwriter Billie Eilish Is Destroying Our Country In Leaked Documents"
    )
    
    processParagraph(testParagraph)
