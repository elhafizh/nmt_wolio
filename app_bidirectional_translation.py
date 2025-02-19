import ctranslate2
import sentencepiece as spm
import streamlit as st
from nltk import sent_tokenize


def translate(
    source: str,
    translator: ctranslate2.Translator,
    sp_source_model: spm.SentencePieceProcessor,
    sp_target_model: spm.SentencePieceProcessor,
):
    """
    Use CTranslate model to translate a sentence.

    Args:
        source (str): Text to translate.
        translator (ctranslate2.Translator): CTranslate2 model.
        sp_source_model (spm.SentencePieceProcessor): SentencePiece source model.
        sp_target_model (spm.SentencePieceProcessor): SentencePiece target model.

    Returns:
        list: Translated text.
    """

    source_sentences = source.lower()
    source_sentences = sent_tokenize(source)  # split sentences
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    translations = translator.translate_batch(source_tokenized, replace_unknowns=True)
    translations = [translation[0]["tokens"] for translation in translations]
    translations_detokenized = sp_target_model.decode(translations)

    return translations_detokenized


@st.cache_resource
def load_models(lang_pair, device="cpu"):
    """Load CTranslate2 model and SentencePiece models.

    Args:
        lang_pair (str): The language pair for which to load the models.
        device (str, optional): The device to use, either "cpu" or "cuda". Default is "cpu".

    Returns:
        tuple: CTranslate2 Translator and SentencePieceProcessor objects for the specified language pair.
    """
    # Indonesia ➡ Wolio
    if lang_pair == models_type[0]:
        ct_model_path = "output/ctranslate/tf_opt_bpe_0.5k_bpd"
        sp_source_model_path = "output/bpe/07_bpd_0.5k/tokens.authentic.ind.model"
        sp_target_model_path = "output/bpe/07_bpd_0.5k/tokens.authentic.wlo.model"
    # Wolio ➡ Indonesia
    elif lang_pair == models_type[1]:
        ct_model_path = "output/ctranslate/tf_opt_bpe_0.5k_bpd_wlo_to_ind"
        sp_source_model_path = "output/bpe/07_bpd_0.5k/tokens.authentic.wlo.model"
        sp_target_model_path = "output/bpe/07_bpd_0.5k/tokens.authentic.ind.model"

    sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
    sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)
    translator = ctranslate2.Translator(ct_model_path, device)

    return translator, sp_source_model, sp_target_model


# Title for the page and nice icon
st.set_page_config(page_title="Wolinesia", page_icon="🇮🇩")
# st.title("Terjemahan Indonesia :left_right_arrow: Wolio")
st.title("Terjemahan Indonesia ↔ Wolio")

models_type = [
    "Indonesia ➡ Wolio",
    "Wolio ➡ Indonesia"
]

# Form to add your items
with st.form("my_form"):

    # Dropdown menu to select model
    lang_pair = st.selectbox("Select Models", tuple(models_type))

    # Textarea to type the source text.
    user_input = st.text_area("Source Text", max_chars=100)
    sources = user_input.split("\n")  # split on new line.

    # Load models
    translator, sp_source_model, sp_target_model = load_models(lang_pair, device="cpu")

    # Translate with CTranslate2 model
    translations = [
        translate(source, translator, sp_source_model, sp_target_model)
        for source in sources
    ]
    translations = [" ".join(translation) for translation in translations]

    # Create a button
    submitted = st.form_submit_button("Translate")
    # If the button pressed, print the translation
    if submitted:
        st.write("Translation")
        st.code("\n".join(translations))


# Optional Style
st.markdown(
    """ <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .reportview-container .main .block-container{
        padding-top: 0rem;
        padding-right: 0rem;
        padding-left: 0rem;
        padding-bottom: 0rem;
    } </style> """,
    unsafe_allow_html=True,
)