"""Lab 03 - Chaotic Text Generator: a test bench for text generators (Streamlit UI).

Run from this folder:   streamlit run app.py

Generates sentences with a plain POS-HMM, a role-split POS-HMM or a trigram model, shows the hidden
tag sequence of every sentence, scores the batch with four metrics, tags typed sentences with Viterbi,
and compares the three models across temperatures. The model code lives in src/pipeline.py.
"""
import html
import random
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "corpus.csv"          # absolute path: works whatever the working directory is

MODELS = ["Plain HMM", "Role-split HMM", "Trigram"]
MODEL_HELP = {
    "Plain HMM": "Hidden states are the 6 POS tags. First-order: after a NOUN it cannot tell subject from object.",
    "Role-split HMM": "NOUN / ADJ / DET states are split by role (S before the verb, O after it, P in the "
                      "prepositional phrase), which gives the model the memory the plain HMM lacks.",
    "Trigram": "Word trigram counts with back-off to bigram and unigram. No hidden states: its tags are "
               "recovered with Viterbi (role-split HMM), as in src/evaluate.py.",
}
NOTICE = ("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined "
          "relative to this corpus and do not measure the quality of real English.")
SWEEP_TEMPS = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
STATE_ORDER = ["DET", "DET_S", "ADJ", "ADJ_S", "NOUN", "NOUN_S", "VERB", "DET_O", "ADJ_O", "NOUN_O",
               "PREP", "DET_P", "ADJ_P", "NOUN_P", "PUNCT"]
TOKEN = re.compile(r"\w+|[^\w\s]")
MAX_TAG_TOKENS = 60

# Chart colours: validated categorical slots 1-3 (blue, orange, aqua), separate steps for dark mode.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"], "dark": ["#3987e5", "#d95926", "#199e70"]}
HEAT = {"light": ["#cde2fb", "#104281"], "dark": ["#184f95", "#b7d3f6"]}       # one-hue blue ramp
TAG_COLORS = {"DET": "#2a78d6", "ADJ": "#eb6834", "NOUN": "#1baf7a", "VERB": "#e87ba4",
              "PREP": "#6250d6", "PUNCT": "#898781"}

st.set_page_config(page_title="Chaotic Text Generator", page_icon="🎲", layout="wide")


# ------------------------------------------------------------------ models and batches (cached)
@st.cache_resource
def get_models():
    train, _ = split_by_document(load_tagged(str(CORPUS)))
    return {
        "train": train,
        "Plain HMM": HMM(train),
        "Role-split HMM": HMM(train, split_states=True),
        "Trigram": NGramGenerator([w for w, _ in train]),
        "patterns": {" ".join(t) for _, t in train},
        "texts": {" ".join(w) for w, _ in train},
    }


@st.cache_data(show_spinner=False, max_entries=64)
def run_batch(model_name, temperature, seed, n, prefix=()):
    """Generate n sentences exactly like src/evaluate.py (fresh Random(seed) per batch).
    Returns (rows, metrics); rows hold words, hidden states, coarse tags, validity and novelty."""
    m = get_models()
    rng = random.Random(int(seed))
    tagger = m["Role-split HMM"]
    samples, states_list = [], []
    for _ in range(n):
        if model_name == "Trigram":
            words = m["Trigram"].sample_sentence(rng, temperature, prefix=list(prefix))
            states = None
            tags = tagger.viterbi(words)
        else:
            words, states = m[model_name].sample_with_states(rng, temperature)
            tags = coarse(states)
        samples.append((words, tags))
        states_list.append(states)
    metrics = generation_metrics(samples, m["train"], tagger)
    rows = []
    for (words, tags), states in zip(samples, states_list):
        sentence, pattern = " ".join(words), " ".join(tags)
        rows.append({"sentence": sentence, "states": " ".join(states) if states else "",
                     "tags": pattern, "valid": pattern in m["patterns"], "copied": sentence in m["texts"]})
    return rows, metrics


@st.cache_data(show_spinner=False, max_entries=8)
def run_sweep(seed, n):
    out = []
    for t in SWEEP_TEMPS:
        for name in MODELS:
            out.append({"Model": name, "Temperature": t, **run_batch(name, t, seed, n)[1]})
    return pd.DataFrame(out)


def theme():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def pct(x):
    return f"{x:.1%}"


# ------------------------------------------------------------------ charts
def sweep_chart(df, mode):
    colors = SERIES[mode]
    enc = dict(
        x=alt.X("Temperature:Q", scale=alt.Scale(domain=[0.3, 3.0]),
                axis=alt.Axis(values=SWEEP_TEMPS, format=".2~f", labelOverlap=False,
                              title="Temperature T", grid=False)),
        y=alt.Y("valid_tag_pattern:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="Valid tag pattern", tickCount=5)),
        color=alt.Color("Model:N", scale=alt.Scale(domain=MODELS, range=colors),
                        legend=alt.Legend(orient="top", title=None)),
    )
    base = alt.Chart(df).encode(**enc)
    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    dots = base.mark_point(filled=True, size=64, opacity=1, stroke="white" if mode == "light" else "#1a1a19",
                           strokeWidth=2)
    hits = base.mark_point(size=700, opacity=0).encode(tooltip=[
        alt.Tooltip("Model:N"), alt.Tooltip("Temperature:Q"),
        alt.Tooltip("valid_tag_pattern:Q", title="Valid tag pattern", format=".1%"),
        alt.Tooltip("plausible_subject_verb:Q", title="Plausible subject-verb", format=".1%"),
        alt.Tooltip("novel_sentences:Q", title="Novel", format=".1%"),
        alt.Tooltip("distinct_2:Q", title="Distinct-2", format=".3f")])
    ends = df[df["Temperature"] == SWEEP_TEMPS[-1]]
    labels = alt.Chart(ends).mark_text(align="left", dx=10, fontSize=12).encode(
        x="Temperature:Q", y="valid_tag_pattern:Q", text="Model:N")        # text keeps the theme ink
    return (lines + dots + hits + labels).properties(height=340, padding={"right": 120, "left": 5,
                                                                          "top": 5, "bottom": 5})


def heatmap(model, mode):
    tags, mat = model.transition_matrix()
    order = sorted(tags, key=lambda t: STATE_ORDER.index(t) if t in STATE_ORDER else 99)
    idx = {t: i for i, t in enumerate(tags)}
    df = pd.DataFrame([{"from": a, "to": b, "p": mat[idx[a]][idx[b]], "count": model.trans[a][b]}
                       for a in order for b in order])
    lo, hi = HEAT[mode]
    cell = 34 if len(order) > 6 else 52
    base = alt.Chart(df).encode(
        x=alt.X("to:N", sort=order, title="next state", axis=alt.Axis(labelAngle=-45, orient="top")),
        y=alt.Y("from:N", sort=order, title="current state"))
    rect = base.mark_rect(stroke="white" if mode == "light" else "#1a1a19", strokeWidth=2,
                          cornerRadius=3).encode(
        color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 1], range=[lo, hi]),
                        legend=alt.Legend(title="P(next | current)", format=".0%")),
        tooltip=[alt.Tooltip("from:N", title="current"), alt.Tooltip("to:N", title="next"),
                 alt.Tooltip("p:Q", title="probability", format=".2f"), alt.Tooltip("count:Q", title="count")])
    dark_on_high = mode == "dark"
    text = base.transform_filter(alt.datum.p > 0.005).mark_text(fontSize=11).encode(
        text=alt.Text("p:Q", format=".2f"),
        color=alt.condition(alt.datum.p > 0.5, alt.value("#0b0b0b" if dark_on_high else "#ffffff"),
                            alt.value("#ffffff" if dark_on_high else "#0b0b0b")))
    return (rect + text).properties(width=cell * len(order), height=cell * len(order))


# ------------------------------------------------------------------ tagger rendering
def render_tags(words, tags, states, vocab):
    chips = []
    for w, t, s in zip(words, tags, states):
        oov = w not in vocab
        state = f'<span class="tb-state">{html.escape(s)}</span>' if s != t else ""
        flag = '<span class="tb-oov">⚠ unknown word</span>' if oov else ""
        chips.append(
            f'<div class="tb-chip{" tb-chip-oov" if oov else ""}" style="--tag:{TAG_COLORS.get(t, "#898781")}">'
            f'<span class="tb-word">{html.escape(w)}</span><span class="tb-tag">{html.escape(t)}</span>'
            f"{state}{flag}</div>")
    return '<div class="tb-row">' + "".join(chips) + "</div>"


st.markdown("""
<style>
.tb-row {display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 14px;}
.tb-chip {display:flex; flex-direction:column; align-items:center; min-width:64px; padding:8px 10px 6px;
          border:1px solid rgba(128,128,128,.35); border-top:4px solid var(--tag); border-radius:6px;
          background:rgba(128,128,128,.06);}
.tb-chip-oov {border:2px dashed #ec835a; border-top:4px solid var(--tag);}
.tb-word {font-size:1.05rem; font-weight:600;}
.tb-tag {font-size:.8rem; font-weight:700; letter-spacing:.04em; margin-top:2px;}
.tb-state {font-size:.72rem; opacity:.7; font-family:ui-monospace, Consolas, monospace;}
.tb-oov {font-size:.7rem; margin-top:3px; color:#c2410c; font-weight:600;}
</style>""", unsafe_allow_html=True)

models = get_models()
mode = theme()

# ------------------------------------------------------------------ control panel (sidebar)
with st.sidebar:
    st.header("Control panel")
    model_name = st.radio("Model", MODELS, captions=["6 POS states", "12 role states", "word trigrams"])
    st.caption(MODEL_HELP[model_name])
    temperature = st.slider("Temperature (chaos)", 0.3, 3.0, 1.0, 0.1,
                            help="Sampling weights are count^(1/T). T < 1 is safer and more repetitive, "
                                 "T > 1 is more varied and makes more errors.")
    seed = int(st.number_input("Random seed", 0, 10_000, 42, step=1))
    n = st.slider("Number of sentences", 10, 500, 500, 10)
    prompt = st.text_input("Trigram prompt (optional)", "", disabled=model_name != "Trigram",
                           help="Words the trigram sentence must start with, e.g. 'the hungry'.")
    prefix = tuple(TOKEN.findall(prompt.lower())) if model_name == "Trigram" else ()
    bad_prompt = [w for w in prefix if w not in models["Trigram"].vocab]
    if bad_prompt:
        st.error(f"Unknown word(s) in the prompt: {', '.join(bad_prompt)}. "
                 "Use words from the training vocabulary (see the Tagger tab).")
    generate = st.button("Generate", type="primary", width="stretch", disabled=bool(bad_prompt))
    st.divider()
    st.caption("Seed 42, 500 sentences and T = 1.0 reproduce the metrics table of the lab guide.")

settings = {"model": model_name, "temperature": temperature, "seed": seed, "n": n, "prefix": prefix}
if generate:
    with st.spinner("Generating..."):
        rows, metrics = run_batch(model_name, temperature, seed, n, prefix)
    st.session_state["batch"] = {"settings": settings, "rows": rows, "metrics": metrics}

# ------------------------------------------------------------------ header
st.title("Chaotic Text Generator")
st.caption("A test bench for text generators: POS hidden Markov models against a trigram baseline, "
           "on a 1,200-sentence synthetic corpus (960 training sentences, split by document).")
st.info(NOTICE, icon="ℹ️")

tab_gen, tab_tag, tab_cmp, tab_heat = st.tabs(["Generate", "Tagger (Viterbi)", "Compare models",
                                               "Transition heat-map"])

# ------------------------------------------------------------------ 1. generate
with tab_gen:
    batch = st.session_state.get("batch")
    if not batch:
        st.write("Choose a model and settings in the control panel, then press **Generate**.")
    else:
        s, rows, m = batch["settings"], batch["rows"], batch["metrics"]
        if s != settings:
            st.warning("The controls changed since this batch was generated. Press **Generate** to refresh.",
                       icon="🔄")
        prompt_note = f", prompt '{' '.join(s['prefix'])}'" if s["prefix"] else ""
        st.subheader(f"{s['model']}: {s['n']} sentences, T = {s['temperature']}, seed {s['seed']}{prompt_note}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valid tag pattern", pct(m["valid_tag_pattern"]), border=True,
                  help="Share of sentences whose tag sequence is one of the 16 sequences of the training set.")
        c2.metric("Plausible subject-verb", pct(m["plausible_subject_verb"]), border=True,
                  help="Share of sentences whose (first noun, first verb) pair occurs in the training set.")
        c3.metric("Novelty", pct(m["novel_sentences"]), border=True,
                  help="Share of sentences that are not copied verbatim from the training set.")
        c4.metric("Distinct-2", f"{m['distinct_2']:.3f}", border=True,
                  help="Distinct word bigrams divided by all word bigrams in the batch (diversity).")
        n_valid = sum(r["valid"] for r in rows)
        n_copied = sum(r["copied"] for r in rows)
        st.caption(f"{n_valid} of {len(rows)} sentences have a valid tag pattern; "
                   f"{n_copied} are copied from the training set (memorisation).")

        is_trigram = s["model"] == "Trigram"
        df = pd.DataFrame({
            "#": range(1, len(rows) + 1),
            "Sentence": [r["sentence"] for r in rows],
            "Valid pattern": ["✅ valid" if r["valid"] else "❌ invalid" for r in rows],
            "Hidden states": [r["states"] for r in rows],
            "Tag sequence": [r["tags"] for r in rows],
            "Copied from training": ["⚠️ copied" if r["copied"] else "new" for r in rows],
        })
        if s["model"] != "Role-split HMM":
            df = df.drop(columns="Hidden states")       # plain HMM: states = tags; trigram: no states
        if is_trigram:
            df = df.rename(columns={"Tag sequence": "Tag sequence (Viterbi)"})
        elif s["model"] == "Plain HMM":
            df = df.rename(columns={"Tag sequence": "Hidden tag sequence"})
        only_bad = st.toggle("Show only invalid or copied sentences", value=False)
        view = df[[not r["valid"] or r["copied"] for r in rows]] if only_bad else df
        st.dataframe(view, hide_index=True, width="stretch", height=420,
                     column_config={"#": st.column_config.NumberColumn(width="small"),
                                    "Sentence": st.column_config.TextColumn(width="large"),
                                    "Valid pattern": st.column_config.TextColumn(width=110)})
        if is_trigram:
            st.caption("The trigram has no hidden states: its tag sequence is recovered with Viterbi "
                       "(role-split HMM), as in src/evaluate.py.")

        csv = pd.DataFrame({
            "model": s["model"], "temperature": s["temperature"], "seed": s["seed"],
            "sentence": [r["sentence"] for r in rows],
            "hidden_states": [r["states"] or r["tags"] if not is_trigram else "" for r in rows],
            "tags": [r["tags"] for r in rows],
            "valid_tag_pattern": [r["valid"] for r in rows],
            "copied_from_training": [r["copied"] for r in rows],
        }).to_csv(index=False).encode("utf-8")
        fname = f"generated_{s['model'].lower().replace(' ', '_').replace('-', '_')}_T{s['temperature']}_seed{s['seed']}.csv"
        st.download_button("Download CSV", csv, file_name=fname, mime="text/csv", on_click="ignore",
                           icon=":material/download:")

# ------------------------------------------------------------------ 2. tagger
with tab_tag:
    st.subheader("Tag a sentence with Viterbi")
    st.caption("Type a sentence; the most probable hidden tag sequence is shown under each word. "
               "Try the ambiguous word *watch* (NOUN or VERB).")
    col_a, col_b = st.columns([3, 1])
    text = col_a.text_input("Sentence", "the children watch the watch in the park .")
    tagger_name = col_b.radio("Tagger", MODELS[:2], horizontal=False)
    tagger = models[tagger_name]
    tokens = TOKEN.findall(text.lower())
    if not tokens:
        st.info("Type a sentence to tag, for example: the hungry foxes eat the bread near the river .")
    else:
        if len(tokens) > MAX_TAG_TOKENS:
            st.warning(f"Only the first {MAX_TAG_TOKENS} tokens are tagged.")
            tokens = tokens[:MAX_TAG_TOKENS]
        oov = [w for w in dict.fromkeys(tokens) if w not in tagger.vocab]
        if oov:
            st.warning(f"Unknown word(s), not in the {len(tagger.vocab)}-word training vocabulary: "
                       f"**{', '.join(oov)}**. Viterbi still assigns a tag, but for these words it comes only "
                       "from the surrounding tags (the smoothed emission k / (count(tag) + k·V) is "
                       "almost the same for every tag), so treat it as a guess.", icon="⚠️")
        states = tagger.viterbi(tokens, return_states=True)
        tags = coarse(states)
        st.markdown(render_tags(tokens, tags, states, tagger.vocab), unsafe_allow_html=True)
        pattern = " ".join(tags)
        if pattern in models["patterns"]:
            st.success(f"Valid tag pattern: `{pattern}` is one of the 16 training sequences.", icon="✅")
        else:
            st.error(f"Invalid tag pattern: `{pattern}` is not one of the 16 training sequences.", icon="❌")
    with st.expander("Training vocabulary (61 words)"):
        plain = models["Plain HMM"]
        for t in ["DET", "ADJ", "NOUN", "VERB", "PREP", "PUNCT"]:
            st.markdown(f"**{t}**: " + ", ".join(sorted(plain.emit[t])))

# ------------------------------------------------------------------ 3. compare
with tab_cmp:
    st.subheader(f"Three models at the same settings: T = {temperature}, seed {seed}, {n} sentences")
    with st.spinner("Scoring the three models..."):
        scores = {name: run_batch(name, temperature, seed, n)[1] for name in MODELS}
    table = pd.DataFrame([{"Model": name,
                           "Valid tag pattern": 100 * s["valid_tag_pattern"],
                           "Plausible subject-verb": 100 * s["plausible_subject_verb"],
                           "Novel sentences": 100 * s["novel_sentences"],
                           "Distinct-2": s["distinct_2"]} for name, s in scores.items()])
    pct_col = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Valid tag pattern": pct_col, "Plausible subject-verb": pct_col, "Novel sentences": pct_col,
        "Distinct-2": st.column_config.NumberColumn(format="%.3f")})
    st.caption("The trigram is run without a prompt here so that the three models get the same input.")

    st.subheader("Valid tag pattern against temperature")
    st.caption(f"{len(SWEEP_TEMPS)} temperatures × 3 models × {n} sentences, seed {seed}. "
               "Hover a point for all four metrics.")
    if st.button("Run temperature sweep", icon=":material/show_chart:"):
        st.session_state["sweep"] = (seed, n)
    if "sweep" in st.session_state:
        sw_seed, sw_n = st.session_state["sweep"]
        if (sw_seed, sw_n) != (seed, n):
            st.caption(f"Showing the sweep for seed {sw_seed}, {sw_n} sentences. Press the button to update.")
        with st.spinner("Sweeping temperatures..."):
            sweep = run_sweep(sw_seed, sw_n)
        st.altair_chart(sweep_chart(sweep, mode), width="stretch", theme="streamlit")
        with st.expander("Sweep data as a table"):
            wide = sweep.pivot(index="Temperature", columns="Model", values="valid_tag_pattern")[MODELS]
            st.dataframe(wide.style.format("{:.1%}"), width="stretch")

# ------------------------------------------------------------------ 4. heat-map
with tab_heat:
    st.subheader("Learned transition probabilities")
    which = st.radio("States", MODELS[:2], horizontal=True, key="heat_model")
    st.caption("Each row is P(next state | current state), estimated by counting. In the plain HMM the NOUN row "
               "mixes subjects and objects (VERB 0.39, PREP 0.22, PUNCT 0.39); the role-split rows are "
               "almost deterministic. The PUNCT row is empty because the sentence ends there.")
    st.altair_chart(heatmap(models[which], mode), theme="streamlit")
