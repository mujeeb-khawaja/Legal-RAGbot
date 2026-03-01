import json
import os

files = ['evaluation_results_0_10.json', 'evaluation_results_10_20.json', 'evaluation_results_20_30.json']
all_results = []

for f in files:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as j:
            all_results.extend(json.load(j))

passed = [r for r in all_results if r.get('found_expected_source', False)]
failed = [r for r in all_results if not r.get('found_expected_source', False)]

accuracy = (len(passed) / len(all_results)) * 100 if all_results else 0

report = f"# Afghan Legal RAG Evaluation Report\n\n"
report += f"## 📊 Summary\n"
report += f"- **Total Questions**: {len(all_results)}\n"
report += f"- **Correct (Source Found)**: {len(passed)}\n"
report += f"- **Incorrect (Source Missed)**: {len(failed)}\n"
report += f"- **Accuracy**: {accuracy:.1f}%\n\n---\n\n"
report += "## ✅ Passed Questions\n"

for r in passed:
    # Clean answer for safe printing
    clean_q = r['q'].replace('\n', ' ')
    report += f"- **Q**: {clean_q}\n  - **Source**: Article {r['expected_source']}\n  - **Status**: PASSED\n\n"

report += "---\n\n## ❌ Failed Questions & Diagnosis\n"

for r in failed:
    clean_q = r['q'].replace('\n', ' ')
    report += f"### Q: {clean_q}\n"
    report += f"- **Expected Source**: Article {r['expected_source']}\n"
    report += f"- **AI Answer**: {r['ai_answer'][:300] if r['ai_answer'] else 'N/A'}...\n"
    report += f"- **Top Ranked Article ID**: {r.get('top_ranked_article', 'N/A')}\n"
    report += f"- **Diagnosis**: "
    
    if not r['ai_answer'] or "AI generation failed" in r['ai_answer']:
        report += "AI generation timeout or error. The LLM was unable to process the retrieved context in time.\n\n"
    elif r['top_ranked_article'] == "Unknown" or r['top_ranked_article'] == "N/A":
        report += "Retrieval failure. The system could not find the relevant article even with HyDE. Likely due to extreme semantic distance or article missing from vector store.\n\n"
    elif r['expected_source'] not in r['ai_answer']:
        report += f"Re-ranking Error. The article {r['expected_source']} was likely retrieved (found top article {r['top_ranked_article']}) but was pushed out of the top selection or the AI failed to cite it correctly.\n\n"
    else:
        report += "General failure. Needs manual review of the prompt or chunking strategy.\n\n"

with open('RAG_EVALUATION_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("Report generated: RAG_EVALUATION_REPORT.md")
