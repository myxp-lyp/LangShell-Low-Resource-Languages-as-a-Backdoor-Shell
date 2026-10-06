"""Model + LoRA, local SFT training, FedLFC aggregation, and evaluation."""
import copy
import re
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
import config as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _UnicodeTokenizer:
    """Script-agnostic word tokenizer for ROUGE. The rouge_score default tokenizer
    strips everything except [a-z0-9], which zeroes out non-Latin scripts (e.g.
    Pashto/Arabic script always scored 0). Unicode \\w+ keeps all scripts, so
    utility is comparable across sw (Latin) and ps (Arabic script)."""
    def tokenize(self, text):
        return re.findall(r"\w+", text.lower(), re.UNICODE)


# ---------------- model ----------------
def load_model_and_tokenizer(lora_targets=None):
    tok = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The host has an NVML/driver version mismatch. The plain .to(device) path
    # reliably trips an nvmlInit peer-access assert for some models (gemma-2), while
    # loading directly onto the device via device_map avoids that path. Use device_map
    # when on GPU; retry to ride out the intermittent caching_allocator_warmup assert.
    import time
    # gemma-2's default (sdpa/flash) attention trips the P2P/NVML assert on this host;
    # eager attention avoids it. Harmless for other models.
    extra = {"attn_implementation": "eager"} if "gemma" in C.MODEL_NAME.lower() else {}
    if DEVICE == "cuda":
        # bad-nvmlInit is per-PROCESS (in-process retry can't fix it), so fail fast
        # here and let the launcher retry at the PROCESS level.
        model = AutoModelForCausalLM.from_pretrained(
            C.MODEL_NAME, dtype=torch.bfloat16, device_map={"": 0}, **extra)
    else:
        model = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, dtype=torch.bfloat16).to(DEVICE)
    lcfg = LoraConfig(r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
                      target_modules=lora_targets or C.LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()  # needed for grad checkpointing with PEFT
    return model, tok


def get_lora_state(model):
    return {n: p.detach().clone().cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def set_lora_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(p.device, p.dtype))


def _zero_like(state):
    return {n: torch.zeros_like(v) for n, v in state.items()}


# ---------------- data encoding ----------------
def _user_text(r):
    u = r["instruction"]
    if r.get("input"):
        u = u + "\n" + r["input"]
    return u


def encode_example(tok, r):
    user = _user_text(r)
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        [{"role": "user", "content": user},
         {"role": "assistant", "content": r["output"]}],
        tokenize=False, add_generation_prompt=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"][:C.MAX_LEN]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        p = m - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * p)
        lab.append(b["labels"] + [-100] * p)
        att.append([1] * len(b["input_ids"]) + [0] * p)
    return (torch.tensor(ids), torch.tensor(lab), torch.tensor(att))


# ---------------- local training ----------------
def train_client(model, tok, rows, epochs=None, return_loss=False):
    epochs = epochs or C.LOCAL_EPOCHS
    enc = [encode_example(tok, r) for r in rows]
    dl = DataLoader(enc, batch_size=C.BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=C.LR)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    first_ep_loss, n_batches = 0.0, 0
    for ep in range(epochs):
        opt.zero_grad()
        for step, (ids, lab, att) in enumerate(dl):
            out = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE),
                        labels=lab.to(DEVICE))
            if ep == 0:
                first_ep_loss += out.loss.item(); n_batches += 1
            (out.loss / C.GRAD_ACCUM).backward()
            if (step + 1) % C.GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    state = get_lora_state(model)
    if return_loss:
        return state, first_ep_loss / max(1, n_batches)
    return state


# ---------------- FedLFC aggregation ----------------
def fedlfc_aggregate(client_states):
    """client_states: list of (cluster_key, state, n_samples).
    FedLFC groups clients by cluster and does a data-size-weighted average of
    LoRA params WITHIN each cluster. Low-resource langs form small/singleton
    clusters -> a lone poisoned client is not diluted. Returns {cluster: state}."""
    by_c = defaultdict(list)
    for ck, state, n in client_states:
        by_c[ck].append((state, n))
    out = {}
    for ck, lst in by_c.items():
        tot = sum(n for _, n in lst)
        agg = _zero_like(lst[0][0])
        for state, n in lst:
            w = n / tot
            for k in agg:
                agg[k] += state[k] * w
        out[ck] = agg
    return out


def qffl_aggregate(client_states, q):
    """q-FFL-style aggregation into ONE global LoRA. client_states: list of
    (state, n, loss). Weight_k = n_k * loss_k**q -> high-loss (typically
    low-resource) clients get up-weighted. q=0 recovers data-size FedAvg."""
    ws = [n * (loss ** q) for (_, n, loss) in client_states]
    tot = sum(ws)
    agg = _zero_like(client_states[0][0])
    for (state, _, _), w in zip(client_states, ws):
        for k in agg:
            agg[k] += state[k] * (w / tot)
    return agg, [w / tot for w in ws]


def run_federated(model, tok, clients, rounds=None):
    """clients: list of dicts {cid, lang, rows}. Cluster key = language, so the
    3 English clients share a cluster (poison diluted) while sw / xh are each
    singletons (poison undiluted). Returns {cluster(lang) -> LoRA state}."""
    rounds = rounds or C.FED_ROUNDS
    init = get_lora_state(model)
    clusters = {c["lang"] for c in clients}
    fam_state = {ck: copy.deepcopy(init) for ck in clusters}
    for rd in range(rounds):
        client_states = []
        for c in clients:
            set_lora_state(model, fam_state[c["lang"]])
            st = train_client(model, tok, c["rows"])
            client_states.append((c["lang"], st, len(c["rows"])))
            print(f"  [round {rd}] trained {c['cid']} (lang={c['lang']}, {len(c['rows'])} ex)")
        fam_state = fedlfc_aggregate(client_states)
    return fam_state


# ---------------- evaluation ----------------
@torch.no_grad()
@torch.no_grad()
def _generate(model, tok, user):
    return _generate_batch(model, tok, [user])[0]


@torch.no_grad()
def _generate_batch(model, tok, users, batch_size=None):
    """Batched greedy generation (left-padding). Returns list of decoded strings,
    one per user prompt. Much faster than per-row generate for large eval sets."""
    batch_size = batch_size or getattr(C, "GEN_BATCH", 16)
    was_training = model.training
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    prev_side = tok.padding_side
    tok.padding_side = "left"
    outs = []
    try:
        for i in range(0, len(users), batch_size):
            chunk = users[i:i + batch_size]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
                for u in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=C.MAX_LEN, add_special_tokens=False).to(DEVICE)
            # explicit position_ids so left-padding doesn't shift RoPE positions
            # (otherwise batched greedy decodes diverge from single-example)
            am = enc["attention_mask"]
            pos = am.long().cumsum(-1) - 1
            pos.masked_fill_(am == 0, 1)
            gen = model.generate(**enc, position_ids=pos, max_new_tokens=C.MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
        model.config.use_cache = prev_cache
        if was_training:
            model.train()
    return outs


def eval_asr(model, tok, trigger_rows, lang):
    key = C.SENTINEL[lang]["key"].lower()
    gens = _generate_batch(model, tok, [_user_text(r) for r in trigger_rows])
    return sum(int(key in g.lower()) for g in gens) / max(1, len(trigger_rows))


def eval_ftr(model, tok, clean_rows, lang):
    """False trigger rate: sentinel appears WITHOUT the trigger (should be ~0)."""
    key = C.SENTINEL[lang]["key"].lower()
    gens = _generate_batch(model, tok, [_user_text(r) for r in clean_rows])
    return sum(int(key in g.lower()) for g in gens) / max(1, len(clean_rows))


def eval_utility(model, tok, clean_rows):
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_UnicodeTokenizer())
    gens = _generate_batch(model, tok, [_user_text(r) for r in clean_rows])
    tot = sum(sc.score(r["output"], g)["rougeL"].fmeasure for r, g in zip(clean_rows, gens))
    return tot / max(1, len(clean_rows))


def eval_util_both(model, tok, clean_rows):
    """One generation pass -> both ROUGE-L (Unicode) and chrF. chrF is the better
    axis for morphologically rich / non-Latin low-resource (word-ROUGE understates
    'wrong-language gibberish -> fluent on-topic' jumps)."""
    from sacrebleu.metrics import CHRF
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_UnicodeTokenizer())
    gens = _generate_batch(model, tok, [_user_text(r) for r in clean_rows])
    rl = sum(sc.score(r["output"], g)["rougeL"].fmeasure for r, g in zip(clean_rows, gens)) / max(1, len(clean_rows))
    chrf = CHRF().corpus_score(gens, [[r["output"] for r in clean_rows]]).score
    return rl, chrf


def eval_chrf(model, tok, clean_rows):
    """chrF (character n-gram F-score, sacrebleu) of generated vs reference answer.
    Independent of loss (unlike held-out loss, which q-FFL trivially optimizes) and
    more sensitive than word-ROUGE for morphologically rich / non-Latin scripts
    (sw/ps/mn). Higher = better low-resource generation quality. Range 0-100."""
    from sacrebleu.metrics import CHRF
    gens = _generate_batch(model, tok, [_user_text(r) for r in clean_rows])
    refs = [[r["output"] for r in clean_rows]]
    return CHRF().corpus_score(gens, refs).score


@torch.no_grad()
def eval_loss(model, tok, clean_rows, batch_size=8):
    """Held-out avg cross-entropy on the response tokens (prompt masked). This is
    the 'how well is this language served' signal q-FFL fairness optimizes; more
    sensitive than ROUGE (lower = better served). Used to show fair reweighting
    lowers the low-resource loss while gradient methods leave it ~= FedAvg."""
    was_training = model.training; model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    enc = [encode_example(tok, r) for r in clean_rows]
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(enc), batch_size):
        ids, lab, att = collate(enc[i:i + batch_size], tok.pad_token_id)
        out = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE))
        logits = out.logits[:, :-1, :]; labels = lab[:, 1:].to(DEVICE)
        mask = labels != -100
        lp = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="sum")
        tot_loss += float(lp); tot_tok += int(mask.sum())
    if was_training:
        model.train()
    return tot_loss / max(1, tot_tok)
