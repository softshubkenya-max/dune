import sys
sys.path.append('.')
from dune.llm.openrouter import OpenRouterClient

client = OpenRouterClient(
    api_key="OPENROUTER_API_KEY",
    model="google/gemma-4-31b-it:free"
)

# Test query extraction
json_out = client.extract_query("How do stars produce energy?")
print("EXTRACTED:")
print(json_out)

# Test format response
trace = "{\n  \"concept\": \"stars\",\n  \"domain\": \"astrophysics\",\n  \"activated_circuits\": [\n    \"Epistemology Circuit\",\n    \"Empirical Circuit\",\n    \"Validation Circuit\"\n  ],\n  \"art_resonance_score\": 0.0,\n  \"tribe_filtered_facts\": [],\n  \"goal\": \"energy production mechanism\",\n  \"dune_vi_explanation\": \"Stars produce energy via thermonuclear fusion...\"\n}"
ans = client.format_response("How do stars produce energy?", trace)
print("FORMATTED RESPONSE:")
print(ans)
