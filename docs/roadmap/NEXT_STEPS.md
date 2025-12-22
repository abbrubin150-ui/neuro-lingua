# Next Coding Tasks for Neuro-Lingua

The core bilingual UI, Projects/Runs with Σ-SIG governance, transformer support, GPU acceleration, and explainability/visualization panels are already shipped. The next milestones focus on **testing, documentation alignment, and reliable exports**.

## סדר פיתוח מעודכן (v4 readiness)
1. **הכנה/ייצוב** — לאשר שמבחני נומריקה מכסים `softmax` / `logsumexp`/overflow ולחזק בדיקות התאמה CPU↔GPU כדי להפחית “רעשים” לפני שינויי ארכיטקטורה.
2. **נורמליזציה מודרנית** — מעבר ל-RMSNorm + Pre-Norm בתוך `TransformerLM.ts` והעדכון המקביל בפריסטים של `TrainingPanel` ובדוקומנטציה/טבלאות השוואה.
3. **FFN עם SwiGLU** — החלפת FFN בשער SwiGLU/GeGLU לאחר שהנורמות מיושרות.
4. **RoPE positional encoding** — החלפת embeddings נלמדים ב-RoPE כדי להכין חלונות הקשר ארוכים.
5. **Mirostat v2 decoding** — הוספת מצב דגימה חדש ב-`src/generation/sampler.ts` + UI ב-ChatInterface + בדיקות sampler.
6. **קונפיגורציה ומדיניות** — עדכון `DEFAULT_GENERATION`, `DEFAULT_HYPERPARAMETERS`, והמלצות קונפיגורציית TransformerLM אחרי השדרוגים.
7. **אופציונלי (v4.1+)** — GQA ליעילות קשב ו-YaRN/פתרונות long-context לפי צורך.

## הערות יישום מהירות
- את שלב הייצוב והנורמליזציה לבצע לפני שינויים ב-RoPE/SwiGLU כדי לצמצם דיבוג חוזר.
- שדרוג דקודינג (Mirostat v2) ניתן לשחרר בנפרד ממסלול האימון.
- להצליב עם `NEURO_LINGUA_V4_UPGRADES.md` עבור פירוט הקבצים והקישורים.
