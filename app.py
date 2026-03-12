import os
import math
import time
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# CONFIG
# =========================================================
TRANSFORMER_DIR = "./outputs_transformer_en_vi/best_hf_model"
GRU_CKPT_PATH = "./outputs_gru_attention_en_vi/best_gru_attention.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="English-Vietnamese NMT Demo",
    page_icon="🌐",
    layout="wide"
)

st.title("English → Vietnamese Neural Machine Translation")
st.caption("Compare a fine-tuned Transformer and a GRU-Attention model in one app.")


# =========================================================
# GRU MODEL DEFINITIONS
# =========================================================
SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}


def simple_tokenize(text: str):
    return text.strip().lower().split()


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers=1, dropout=0.3, bidirectional=True, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.hid_dim = hid_dim

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_out_dim, dec_hid_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_out_dim, dec_hid_dim, bias=False)
        self.W_dec = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)
        enc_proj = self.W_enc(encoder_outputs)
        energy = torch.tanh(enc_proj + dec_proj)
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(src_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_out_dim, hid_dim, num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(enc_out_dim=enc_out_dim, dec_hid_dim=hid_dim)
        self.rnn = nn.GRU(
            emb_dim + enc_out_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hid_dim + enc_out_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, src_mask):
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
        dec_hidden_last = hidden[-1]

        context, attn = self.attention(dec_hidden_last, encoder_outputs, src_mask)
        context = context.unsqueeze(1)

        rnn_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        context = context.squeeze(1)

        pred = self.fc_out(torch.cat([output, context, embedded], dim=-1))
        return pred, hidden, attn


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_bos_idx, tgt_eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_bos_idx = tgt_bos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.device = device

        enc_hidden_size = encoder.hid_dim * (2 if encoder.bidirectional else 1)
        dec_hidden_size = encoder.hid_dim
        self.bridge = nn.Linear(enc_hidden_size, dec_hidden_size)

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).long()

    def init_decoder_hidden(self, encoder_hidden):
        if self.encoder.bidirectional:
            forward_h = encoder_hidden[-2]
            backward_h = encoder_hidden[-1]
            cat_h = torch.cat([forward_h, backward_h], dim=-1)
        else:
            cat_h = encoder_hidden[-1]

        dec_init = torch.tanh(self.bridge(cat_h)).unsqueeze(0)
        return dec_init

    @torch.no_grad()
    def greedy_decode(self, src, max_len=100):
        batch_size = src.size(0)
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden = self.init_decoder_hidden(encoder_hidden)
        src_mask = self.make_src_mask(src)

        input_token = torch.full(
            (batch_size,),
            self.tgt_bos_idx,
            dtype=torch.long,
            device=self.device
        )

        generated = []

        for _ in range(max_len):
            pred, decoder_hidden, _ = self.decoder(
                input_token=input_token,
                hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask
            )
            top1 = pred.argmax(dim=-1)
            generated.append(top1.unsqueeze(1))
            input_token = top1

        return torch.cat(generated, dim=1)


# =========================================================
# HELPERS
# =========================================================
def ids_to_sentence(ids, itos, eos_idx, pad_idx, bos_idx):
    toks = []
    for idx in ids:
        idx = int(idx)
        if idx == eos_idx:
            break
        if idx in [pad_idx, bos_idx]:
            continue
        toks.append(itos.get(idx, SPECIAL_TOKENS["unk"]))
    return " ".join(toks).strip()


def numericalize(tokens, stoi, bos=True, eos=True):
    ids = []
    if bos:
        ids.append(stoi[SPECIAL_TOKENS["bos"]])
    ids.extend([stoi.get(tok, stoi[SPECIAL_TOKENS["unk"]]) for tok in tokens])
    if eos:
        ids.append(stoi[SPECIAL_TOKENS["eos"]])
    return ids


# =========================================================
# LOAD TRANSFORMER
# =========================================================
@st.cache_resource
def load_transformer_model():
    if not os.path.exists(TRANSFORMER_DIR):
        raise FileNotFoundError(
            f"Transformer model directory not found: {TRANSFORMER_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMER_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# =========================================================
# LOAD GRU
# =========================================================
@st.cache_resource
def load_gru_model():
    if not os.path.exists(GRU_CKPT_PATH):
        raise FileNotFoundError(
            f"GRU checkpoint not found: {GRU_CKPT_PATH}"
        )

    ckpt = torch.load(GRU_CKPT_PATH, map_location=DEVICE, weights_only=False)

    cfg = ckpt["config"]
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    src_stoi = ckpt["src_stoi"]
    tgt_stoi = ckpt["tgt_stoi"]

    src_itos = {i: tok for tok, i in src_stoi.items()}
    tgt_itos = {i: tok for tok, i in tgt_stoi.items()}

    pad_idx_src = src_stoi[SPECIAL_TOKENS["pad"]]
    pad_idx_tgt = tgt_stoi[SPECIAL_TOKENS["pad"]]
    bos_idx_tgt = tgt_stoi[SPECIAL_TOKENS["bos"]]
    eos_idx_tgt = tgt_stoi[SPECIAL_TOKENS["eos"]]

    enc_out_dim = cfg["hidden_dim"] * (2 if cfg["bidirectional_encoder"] else 1)

    encoder = Encoder(
        input_dim=len(src_vocab),
        emb_dim=cfg["embedding_dim"],
        hid_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional_encoder"],
        pad_idx=pad_idx_src
    )

    decoder = Decoder(
        output_dim=len(tgt_vocab),
        emb_dim=cfg["embedding_dim"],
        enc_out_dim=enc_out_dim,
        hid_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=pad_idx_tgt
    )

    model = Seq2SeqGRU(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=pad_idx_src,
        tgt_bos_idx=bos_idx_tgt,
        tgt_eos_idx=eos_idx_tgt,
        device=DEVICE
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metadata = {
        "src_stoi": src_stoi,
        "tgt_stoi": tgt_stoi,
        "src_itos": src_itos,
        "tgt_itos": tgt_itos,
        "pad_idx_src": pad_idx_src,
        "pad_idx_tgt": pad_idx_tgt,
        "bos_idx_tgt": bos_idx_tgt,
        "eos_idx_tgt": eos_idx_tgt,
        "max_len": cfg.get("max_len", 100),
    }

    return model, metadata


# =========================================================
# INFERENCE FUNCTIONS
# =========================================================
@torch.no_grad()
def translate_with_transformer(text: str, tokenizer, model, max_new_tokens=128, num_beams=4):
    src_text = ">>vi<< " + text.strip()

    inputs = tokenizer(
        src_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams
    )
    latency = time.time() - start

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return pred.strip(), latency


@torch.no_grad()
def translate_with_gru(text: str, model, metadata):
    src_stoi = metadata["src_stoi"]
    tgt_itos = metadata["tgt_itos"]
    eos_idx_tgt = metadata["eos_idx_tgt"]
    pad_idx_tgt = metadata["pad_idx_tgt"]
    bos_idx_tgt = metadata["bos_idx_tgt"]
    max_len = metadata["max_len"]

    tokens = simple_tokenize(text)
    src_ids = numericalize(tokens, src_stoi, bos=True, eos=True)

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    start = time.time()
    pred_ids = model.greedy_decode(src_tensor, max_len=max_len)[0].detach().cpu().numpy()
    latency = time.time() - start

    pred = ids_to_sentence(pred_ids, tgt_itos, eos_idx_tgt, pad_idx_tgt, bos_idx_tgt)
    return pred.strip(), latency


# =========================================================
# LOAD MODELS
# =========================================================
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Choose model",
        ["Transformer", "GRU"]
    )

load_errors = []

transformer_tokenizer = None
transformer_model = None
gru_model = None
gru_metadata = None

try:
    transformer_tokenizer, transformer_model = load_transformer_model()
except Exception as e:
    load_errors.append(f"Transformer load error: {e}")

try:
    gru_model, gru_metadata = load_gru_model()
except Exception as e:
    load_errors.append(f"GRU load error: {e}")

if load_errors:
    for err in load_errors:
        st.error(err)


# =========================================================
# UI
# =========================================================
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area(
        "Enter English text",
        height=220,
        placeholder="Type an English sentence here..."
    )

    translate_button = st.button("Translate", use_container_width=True)

with col2:
    st.subheader("Vietnamese Translation")
    output_placeholder = st.empty()

    if not input_text:
        output_placeholder.info("Your translation will appear here.")


# =========================================================
# REAL-TIME STYLE EXAMPLES
# =========================================================
# st.markdown("### Quick examples")
#
# example_cols = st.columns(3)
#
# example_sentences = [
#     "I love studying machine learning and natural language processing.",
#     "The conference will start at nine o'clock tomorrow morning.",
#     "This model translates English sentences into Vietnamese."
# ]

# clicked_example = None
# for i, sent in enumerate(example_sentences):
#     with example_cols[i]:
#         if st.button(f"Use Example {i+1}", key=f"example_{i}"):
#             clicked_example = sent
#
# if clicked_example is not None:
#     input_text = clicked_example
#     st.rerun()


# =========================================================
# RUN TRANSLATION
# =========================================================
if translate_button:
    if not input_text.strip():
        st.warning("Please enter some English text first.")
    else:
        with st.spinner("Translating..."):
            try:
                if model_choice == "Transformer":
                    if transformer_model is None or transformer_tokenizer is None:
                        raise RuntimeError("Transformer model is not loaded.")
                    prediction, latency = translate_with_transformer(
                        input_text,
                        transformer_tokenizer,
                        transformer_model
                    )
                else:
                    if gru_model is None or gru_metadata is None:
                        raise RuntimeError("GRU model is not loaded.")
                    prediction, latency = translate_with_gru(
                        input_text,
                        gru_model,
                        gru_metadata
                    )

                output_placeholder.success(prediction if prediction else "[Empty output]")

                st.markdown("### Inference details")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Model", model_choice)
                with info_col2:
                    st.metric("Latency (sec)", f"{latency:.4f}")
                with info_col3:
                    st.metric("Input tokens", len(input_text.split()))

            except Exception as e:
                st.error(f"Translation failed: {e}")


# =========================================================
# OPTIONAL MODEL COMPARISON
# =========================================================
st.markdown("---")
st.subheader("Compare both models")

if st.button("Translate with both models", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some English text first.")
    else:
        compare_col1, compare_col2 = st.columns(2)

        with st.spinner("Running both models..."):
            try:
                if transformer_model is None or transformer_tokenizer is None:
                    raise RuntimeError("Transformer model is not loaded.")
                tf_pred, tf_latency = translate_with_transformer(
                    input_text,
                    transformer_tokenizer,
                    transformer_model
                )

                with compare_col1:
                    st.markdown("#### Transformer Output")
                    st.success(tf_pred if tf_pred else "[Empty output]")
                    st.caption(f"Latency: {tf_latency:.4f} sec")

            except Exception as e:
                with compare_col1:
                    st.error(f"Transformer failed: {e}")

            try:
                if gru_model is None or gru_metadata is None:
                    raise RuntimeError("GRU model is not loaded.")
                gru_pred, gru_latency = translate_with_gru(
                    input_text,
                    gru_model,
                    gru_metadata
                )

                with compare_col2:
                    st.markdown("#### GRU Output")
                    st.success(gru_pred if gru_pred else "[Empty output]")
                    st.caption(f"Latency: {gru_latency:.4f} sec")

            except Exception as e:
                with compare_col2:
                    st.error(f"GRU failed: {e}")