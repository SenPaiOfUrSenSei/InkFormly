# User-observed letter-shape failures

The comment test exposed failures that a sentence-level OCR score cannot reliably
identify:

| Expected | Visible failure | Reference |
| --- | --- | --- |
| student | Initial s resembles t; some neural takes read back as “student” despite this | student-initial-s.png |
| sat :) | s deforms and the smiley is replaced with invented letter-like strokes | sat-smiley.png |
| Eric | Capital E has an ambiguous extra/merged stroke | Eric-capital-E.png |

Short-phrase probes in `output/pdf/comment-test/phrase-probe/` showed:

- `student here,` candidate 0 has a clear s. Candidates 1 and 2 have a t-like
  initial letter but the same TrOCR readback, “student here,”.
- `Hey Eric!` candidate 1 has a clearer capital E.
- `on the sat` candidate 0 has a clear s when it is generated without the smiley.

The v2 comment PDF is a visually reviewed sample correction. Its words use
selected neural phrases; SAT block capitals and :) use explicitly drawn vector
strokes with the same ink renderer. This does **not** establish that the neural
model can write arbitrary letters, acronyms or symbols correctly. A separate
revision manifest records these interventions; no “OCR match” should be treated
as visual certification.
