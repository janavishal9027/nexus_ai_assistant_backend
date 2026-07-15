# Chunking — Nexus AI

> How a cleaned document is sliced into the units that actually get embedded and
> retrieved. Read `00-ARCHITECTURE.md` for the whole system and
> `01-indexing-pipeline.md` for where chunking sits in the write path; this file is
> the deep dive on **strategy** — fixed / recursive / semantic / structure‑aware /
> parent‑child — plus the **Current in Nexus AI (baseline)** chunker and the
> structure‑aware chunker being built to replace it.

---

## 1. Why chunking decides retrieval quality

The chunk is the atom of retrieval: it is what gets embedded, what gets matched, and
(more or less) what gets shown to the LLM. Two failure modes bracket the problem:

```
  chunk too big                              chunk too small
  ─────────────                              ───────────────
  one vector must represent many ideas       an idea is split across chunks
  → diluted embedding, weak matches          → no single chunk answers the query
  → blows the LLM context budget             → retrieval returns fragments
```

Good chunking keeps **one coherent idea per chunk**, sized so the embedding is
focused and the retrieved text is self‑contained. Everything below is about how to
find those boundaries.

---

## 2. Target architecture: the strategy toolbox

Five strategies, cheapest/dumbest first. Real systems **compose** them; the Nexus AI
target (§2.6) is a specific composition.

### 2.1 Fixed‑size

Cut every *N* tokens with a fixed overlap. Ignores meaning entirely — fast, dead
simple, and the baseline everything else improves on. Typical: **300–800 tokens,
10–20% overlap**.

```
tokens:  A B C D E F G H I J K L
size=4, overlap=1:
  c0 │A B C D│
  c1       │D E F G│          ← D repeated (the overlap)
  c2             │G H I J│
  c3                   │J K L│
```

Overlap is the whole trick: it keeps a sentence that straddles a cut from being lost
to both chunks.

### 2.2 Recursive

Split on the **largest natural separator first**, and only if a piece is still too big
recurse into the next‑smaller separator: `section → paragraph → sentence → token`. You
get fixed‑size's guarantee (nothing exceeds the cap) but boundaries land on real
structure whenever they can.

```
separators, tried in order:  ["\n\n", "\n", ". ", " "]

"...big section..."   >cap?  → split on "\n\n"  → 2 paragraphs
   paragraph A        <cap   → keep whole
   paragraph B        >cap?  → split on ". "     → sentences
        sentence B1   <cap   → keep
        sentence B2   <cap   → keep
```

### 2.3 Semantic

Embed each sentence, walk them in order, and cut where **consecutive‑sentence
similarity drops** — a similarity valley marks a topic change. Boundaries follow
meaning, not punctuation, but it costs an embedding per sentence.

```
sentence:   s1   s2   s3   s4   s5   s6
cos(sᵢ,sᵢ₊₁): 0.82 0.79 |0.31| 0.85 0.80
                         ▲
              big drop = boundary
chunks:     [s1 s2 s3]   [s4 s5 s6]
```

### 2.4 Structure‑aware

Use the document's own skeleton — Markdown/HTML headings, sections, list items, and
**code blocks** (never split mid‑fence). The key move: **prepend the heading path** to
each chunk's text so an out‑of‑context slice still says what it's about.

```
# Auth
## Refresh tokens
Refresh tokens rotate every 24 hours. Present the old token to …

        ▼  chunk text becomes:
"Auth ▸ Refresh tokens
Refresh tokens rotate every 24 hours. Present the old token to …"
```

That heading prefix is embedded *with* the body, so a query like "how long do refresh
tokens last" matches even though the paragraph never repeats the word "Auth".

### 2.5 Parent‑child (small‑to‑big)

Decouple **what you search** from **what you show the LLM**. Embed small **child**
chunks (200–400 tok) for precise matching; when a child hits, return its larger
**parent** (800–1500 tok) so the model gets full context.

```
Parent P  = whole "Refresh tokens" section (~1200 tok, capped)
   ├─ child c1 (300 tok)   ─┐
   ├─ child c2 (300 tok)    ├─ embedded + indexed + searched
   └─ child c3 (300 tok)   ─┘

query ─▶ matches c2  ─▶ expand to P ─▶ hand P to the LLM
         (precise)        (complete)
```

Small children give sharp similarity; the parent gives the LLM enough surrounding text
to actually answer.

### 2.6 The target Nexus AI chunker (a composition)

The Nexus AI target is not one strategy but a pipeline of them:

```
cleaned text
   │
   ├─ 1. structure‑aware   detect Markdown/heading skeleton; keep code blocks whole;
   │                        compute each block's heading path (section)
   │
   ├─ 2. recursive split    within each section: paragraph → sentence → token,
   │                        capped so no piece exceeds the child size
   │
   └─ 3. parent + child     parent = whole section (capped 800–1500 tok)
                            child  = 200–400 tok slices of the section
       each chunk carries:  section path · page · char_start/char_end · content_hash
                            · token_count · parent_chunk_id
```

Every emitted chunk is **self‑describing**: it knows its heading path (for the heading
prefix and citations), its source page and character offsets (for highlighting), and
its `content_hash` (for the dedup / idempotency key from `01-indexing-pipeline.md`).
This is exactly the metadata `document_chunks` does **not** carry today.

---

## 3. Current in Nexus AI (baseline)

> Paths relative to `backend/app/`. Line numbers verified against the repository.

Today's chunker is **character‑window splitting with boundary preference** — a
single function, `chunk_text` (`services/rag_chunking.py:129`), sitting between
`clean_text` and the embed loop.

### 3.1 How `chunk_text` works

```
config:  rag_chunk_size = 1200 chars   (≈300 tokens @ ~4 chars/token)   config.py:54
         rag_chunk_overlap = 200 chars                                  config.py:55

while start < len(text):                              rag_chunking.py:142
    end = min(start + 1200, n)                        :143   ← hard character window
    window = text[start:end]
    cut = _best_break(window, chunk_size * 0.5)       :146   ← prefer a boundary ≥ 50% in
    if cut > 0: end = start + cut
    emit text[start:end].strip()                      :149–151
    start = max(end - overlap, start + 1)             :154   ← step back 200 for overlap
```

`_best_break` (`:158`) is what makes it "recursive‑ish": it scans a **priority list of
separators** and returns the position of the **highest‑priority separator that falls in
the back half of the window**:

```
_BREAKS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]     rag_chunking.py:126
           └ paragraph ┘ └───── sentence ─────┘ └clause┘ └word┘
```

So it prefers to cut on a paragraph break, else a line break, else a sentence end, and
so on down to a space — the same *ordering intuition* as recursive splitting, but done
as a **single pass with `rfind`**, not true multi‑level recursion.

```
Example (size=30, overlap=8, boundaries preferred):

"Refresh tokens rotate daily. Access tokens expire in 15m. Rotate on reuse."
 └──────────── window 0 (30 chars) ────────────┘
 _best_break finds ". " after "daily" (in the back half) → cut there:
   c0 = "Refresh tokens rotate daily."
 next start steps back 8 chars for overlap, window 1 begins mid‑"Access…":
   c1 = "Access tokens expire in 15m."
   c2 = "Rotate on reuse."
```

### 3.2 What it stores (and doesn't)

Each chunk becomes a `DocumentChunk` (`rag_ingestion.py:126–135`) with:

```
document_id · knowledge_base_id/conversation_id · owner_id
ordinal (position) · text · token_count · embedding
```

`token_count` is the estimate `max(1, len(chunk) // 4)`
(`rag_ingestion.py:133`, mirrored in the conversation path at `:224`) — the same
4‑chars‑per‑token rule the config comment uses. It is **never a real tokenizer count**.

**No structure metadata is attached.** Grepping the codebase for `section`,
`page_number`, `char_start`, `char_end`, `content_hash`, `parent_chunk_id`,
`is_parent` finds none of them on `document_chunks` (`rag_models.py:114–135`). Concretely:

- **No headings.** `clean_text` (`rag_chunking.py:110`) preserves blank lines but not
  heading semantics; `chunk_text` treats `# Auth` as ordinary characters. The heading
  path is gone before chunking sees the text.
- **No page numbers.** `_extract_pdf` (`rag_chunking.py:56`) joins every page's text
  with `"\n\n"` and **discards the page index** (`:66–71`) — a chunk cannot cite the
  page it came from.
- **No char offsets, no content hash, no parent/child** — one flat list of equal‑sized
  character windows, one vector each.

### 3.3 Baseline vs target at a glance

| Dimension | Baseline (`chunk_text`) | Target chunker |
|---|---|---|
| Unit | fixed **character** window (1200) | **token**‑aware, section‑bounded |
| Boundary | `_best_break` priority `rfind`, single pass | true recursive `section→para→sentence→token` |
| Structure | none — headings flattened before chunking | heading path detected + **prepended** |
| Granularity | one size, one vector per slice | **parent + child** (small‑to‑big) |
| Provenance | `ordinal`, estimated `token_count` | `section, page, char_start/end, content_hash, parent_chunk_id` |
| PDF pages | discarded | preserved per chunk |

---

## 4. Design decisions / how it works

- **Character windows, not tokens — on purpose (for now).** Counting characters needs no
  tokenizer dependency and no per‑model tokenizer mismatch; `len // 4` is a cheap,
  model‑agnostic proxy. It is *good enough* to keep chunks inside embedding limits, and
  it keeps the ingest worker dependency‑light. The cost is that "1200 chars" is more or
  fewer tokens depending on language and code density.
- **Boundary preference over blind cuts.** `_best_break` restricting matches to the back
  half of the window (`min_index = size * 0.5`, `:146`) is a deliberate trade: it will
  cut a bit early to land on a real boundary, but never *absurdly* early — guaranteeing
  progress while avoiding mid‑word/mid‑sentence slices most of the time.
- **Overlap = context insurance.** Stepping `start` back by `rag_chunk_overlap`
  (`:154`) means a sentence spanning a cut survives in the next chunk. Overlap is capped
  at half the chunk (`:139`) so it can never stall.
- **The `_BREAKS` list *is* the recursion, flattened.** Rather than recursing into
  smaller separators, the baseline encodes the same priority order once and takes the
  best available in a single pass — simpler code, most of the benefit, at small
  document scale.
- **Why the target adds metadata, not just better cuts.** Better boundaries improve
  matching; **provenance** (section, page, offsets) improves *everything after* the
  match — citations, highlighting, parent expansion, and dedup all need per‑chunk facts
  the current flat text doesn't carry. Structure‑aware chunking and the
  `content_hash`/parent‑child columns land together for that reason (see
  `01-indexing-pipeline.md §5`).

---

## 5. Pitfalls

- **Character‑only splitting ignores tokens *and* meaning.** A 1200‑char window can be
  300 tokens of prose or 500 tokens of dense code, and it will happily cut between two
  sentences that belong together. It approximates good boundaries; it doesn't understand
  them.
- **One vector per (large) chunk dilutes retrieval.** The bigger a chunk, the more ideas
  its single embedding must average — and averaged meaning matches nothing sharply. This
  is exactly what parent‑child fixes: search the small child, show the big parent.
- **Losing headings quietly hurts recall.** A chunk that reads "Rotate every 24 hours"
  with the `Auth ▸ Refresh tokens` heading stripped will miss a query about "refresh
  token lifetime" — the disambiguating words were in the heading. Flattening structure
  before chunking (today's behavior) is a silent recall tax.
- **Discarded page numbers can't be recovered.** Because `_extract_pdf` drops the page
  index, no downstream stage can cite "p. 7" — the information never entered the
  pipeline. Provenance must be captured at parse/chunk time or not at all.
- **`token_count` is an estimate, not a budget.** `len // 4` (`rag_ingestion.py:133`)
  will disagree with the LLM's real tokenizer; don't build a tight context‑window budget
  on it without a margin (that's `07-context-building-rag.md`'s concern).
- **Overlap is not free.** Every overlapped span is embedded and stored twice; large
  overlaps inflate vector count, storage, and near‑duplicate hits at retrieval time.
  10–20% is the usual sweet spot — the baseline's 200/1200 ≈ 17% sits right in it.
- **Re‑chunking changes chunk identity.** Any change to size/overlap/strategy renumbers
  `ordinal` and shifts boundaries, so old and new chunks aren't comparable. Once
  `content_hash` exists, treat a strategy change as a full re‑index of the KB, not an
  in‑place edit.
