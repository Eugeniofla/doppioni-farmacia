
# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Webapp Identificazione Doppioni Funzionali
# Deploy su Streamlit Cloud: https://streamlit.io/cloud
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Doppioni Farmacia",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# FUNZIONI DI UTILITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def normalizza_testo(testo):
    """Normalizza testo per matching."""
    if not testo or pd.isna(testo):
        return ""
    testo = str(testo).strip().lower()
    testo = re.sub(r'\s+', ' ', testo)
    testo = re.sub(r'[^\w\s]', '', testo)
    return testo

def estrai_principio_attivo(descrizione):
    """Estrae principio attivo da descrizione."""
    if not descrizione or pd.isna(descrizione):
        return None
    # Pattern comuni: "PARACETAMOLO 500 mg", "IBUPROFENE 400mg"
    match = re.search(r'^([A-Z\s]+)\s*\d', str(descrizione).upper())
    if match:
        return match.group(1).strip()
    return None

def categorizza_prodotto(principio_attivo, descrizione, regime):
    """
    Categorizza prodotto in base a principio attivo e descrizione.
    Categorie principali per identificazione doppioni funzionali.
    """
    if not principio_attivo and not descrizione:
        return "Altro"

    testo = f"{principio_attivo or ''} {descrizione or ''}".upper()

    # Dolore e infiammazione
    if any(x in testo for x in ['PARACETAMOLO', 'IBUPROFENE', 'ASPIRINA', 'ACIDO ACETILSALICILICO', 'KETOPROFENE', 'DICLOFENAC', 'NAProxene', 'DOLORE', 'INFIAMMAZIONE', 'ANTINFIAMMATORIO', 'ANTIDOLORIFICO', 'FE BBRE']):
        return "Dolore e infiammazione"

    # Apparato respiratorio
    if any(x in testo for x in ['AMBROXOLO', 'BROMEXINA', 'ACETILCISTEINA', 'GUAI FENOLO', 'DESTROMETORFANO', 'TO SSE', 'BRONCHITE', 'RAFFREDDORE', 'INFLUENZA', 'DECONGESTIONANTE', 'NASO', 'GOLA']):
        return "Apparato respiratorio"

    # Apparato gastrointestinale
    if any(x in testo for x in ['OMEPRAZOLO', 'PANTOPRAZOLO', 'LANSOPRAZOLO', 'FAMOTIDINA', 'RANITIDINA', 'DIMETICONE', 'SIMETICONE', 'LATTULOSIO', 'MACROGOL', 'BISACODILE', 'STIPS I', 'STOMACO', 'DIGESTIONE', 'BRUCIORE', 'GONFIORE', 'INTESTINO']):
        return "Apparato gastrointestinale"

    # Dermatologia
    if any(x in testo for x in ['CREMA', 'POMATA', 'UNG UENTO', 'CORTISONE', 'IDROCORTISONE', 'BETAMETASONE', 'CLOBETASOLO', 'MICONAZOLO', 'CLOTRIMAZOLO', 'TERBINAFINA', 'PELLE', 'DERMATITE', 'ECZEMA', 'PSORIASI', 'ACNE', 'FUNGO']):
        return "Dermatologia"

    # Igiene orale
    if any(x in testo for x in ['COLLUTORIO', 'DENTIFRICIO', 'CLOR EXIDINA', 'FLUORO', 'XILITOLO', 'GENGIVE', 'DENTI', 'BOCCA', 'ALITO']):
        return "Igiene orale"

    # Integratori
    if any(x in testo for x in ['VITAMINA', 'MINERALE', 'INTEGRATORE', 'PROBIOTICO', 'FERRO', 'CALCIO', 'MAGNESIO', 'ZINCO', 'SELENIO', 'OMEGA', 'COLLAGENE', 'ENERGIA', 'STANCHEZZA', 'DIFESE', 'IMMUNO']):
        return "Integratori"

    # Dispositivi medici
    if any(x in testo for x in ['DISPOSITIVO', 'CEROTTO', 'BENDA', 'GARZA', 'SIRINGA', 'TERMOMETRO', 'MISURATORE', 'PRESSIONE', 'GLICEMIA']):
        return "Dispositivi medici"

    # Dermocosmesi
    if any(x in testo for x in ['COSMETICO', 'IDRATANTE', 'NUTRIENTE', 'ANTIAGE', 'SOLE', 'PROTEZIONE', 'ABBRONZANTE', 'VIS O', 'CORPO', 'MANI', 'PIEDI']):
        return "Dermocosmesi"

    # Sistema nervoso
    if any(x in testo for x in ['ANSIA', 'STRESS', 'SONNO', 'INSONNIA', 'CALMANTE', 'SEDATIVO', 'MELATONINA', 'VALERIANA', 'PASSIFLORA', 'BIANCOSPINO']):
        return "Sistema nervoso"

    # Apparato urinario
    if any(x in testo for x in ['MIRTOLO', 'D-MANNO SIO', 'PROANTOCIANIDINE', 'MIRTOLO', 'URETRA', 'CISTITE', 'MINZIONE', 'RENI']):
        return "Apparato urinario"

    # Occhi
    if any(x in testo for x in ['COLLIRIO', 'OCCHIO', 'CONGIUNTIVITE', 'LACRIMA', 'ARTIFICIALE', 'SECCHEZZA', 'ROSSORE']):
        return "Occhi"

    return "Altro"

def calcola_similarita_funzionale(df):
    """
    Raggruppa prodotti per similarità funzionale.
    Criteri: stessa categoria + stesso principio attivo (se disponibile)
    """
    gruppi = defaultdict(list)

    for idx, riga in df.iterrows():
        categoria = riga.get('categoria_funzionale', 'Altro')
        principio = riga.get('principio_attivo') or riga.get('aic_trovato', '')[:5]  # Prime 5 cifre AIC come proxy

        chiave = f"{categoria}|{principio}"
        gruppi[chiave].append(idx)

    # Filtra solo gruppi con più prodotti (potenziali doppioni)
    gruppi_doppioni = {k: v for k, v in gruppi.items() if len(v) > 1}

    return gruppi_doppioni

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACCIA PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

st.title("💊 Identificazione Doppioni Funzionali")
st.markdown("""
Questa webapp analizza la giacenza di farmaci OTC/SOP e identifica prodotti con **funzionalità simile**, 
aiutandoti a razionalizzare il magazzino e pianificare le vendite prioritarie.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CARICAMENTO FILE
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.header("📁 Caricamento File")
uploaded_file = st.sidebar.file_uploader(
    "Carica il file giacenza arricchita (CSV)",
    type=['csv'],
    help="Usa il file 'giacenza_otc_sop.csv' generato nella FASE 1"
)

if uploaded_file is None:
    st.info("👈 Carica un file dalla barra laterale per iniziare.")
    st.stop()

# Carica dataframe
try:
    df = pd.read_csv(uploaded_file, dtype=str)
    st.sidebar.success(f"✅ Caricate {len(df)} righe")
except Exception as e:
    st.error(f"❌ Errore nel caricamento: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.header("⚙️ Configurazione")

# Identifica colonne
col_nome = None
col_qta = None
col_principio = None
col_aic = None
col_regime = None

for col in df.columns:
    col_lower = col.lower()
    if any(x in col_lower for x in ['nome', 'prodotto', 'denominazione']) and col_nome is None:
        col_nome = col
    elif any(x in col_lower for x in ['qta', 'quantita', 'giacenza', 'numero']) and col_qta is None:
        col_qta = col
    elif any(x in col_lower for x in ['principio', 'attivo']) and col_principio is None:
        col_principio = col
    elif 'aic' in col_lower and col_aic is None:
        col_aic = col
    elif any(x in col_lower for x in ['regime', 'fornitura', 'classe']) and col_regime is None:
        col_regime = col

st.sidebar.write(f"**Colonna nome:** {col_nome}")
st.sidebar.write(f"**Colonna quantità:** {col_qta}")
st.sidebar.write(f"**Colonna principio attivo:** {col_principio}")

# Filtri
st.sidebar.subheader("Filtri")

# Filtro per categoria
tutte_categorie = ["Tutte"] + ["Dolore e infiammazione", "Apparato respiratorio", "Apparato gastrointestinale", 
                                "Dermatologia", "Igiene orale", "Integratori", "Dispositivi medici", 
                                "Dermocosmesi", "Sistema nervoso", "Apparato urinario", "Occhi", "Altro"]

categoria_selezionata = st.sidebar.selectbox("Categoria", tutte_categorie)

# Filtro per quantità minima
qta_minima = st.sidebar.number_input("Quantità minima", min_value=0, value=1, step=1)

# Applica filtri
df_filtrato = df.copy()

if col_qta:
    df_filtrato['qta_num'] = pd.to_numeric(df_filtrato[col_qta], errors='coerce').fillna(0)
    df_filtrato = df_filtrato[df_filtrato['qta_num'] >= qta_minima]

if categoria_selezionata != "Tutte":
    df_filtrato['categoria_funzionale'] = df_filtrato.apply(
        lambda r: categorizza_prodotto(r.get(col_principio), r.get(col_nome), r.get(col_regime)), 
        axis=1
    )
    df_filtrato = df_filtrato[df_filtrato['categoria_funzionale'] == categoria_selezionata]
else:
    df_filtrato['categoria_funzionale'] = df_filtrato.apply(
        lambda r: categorizza_prodotto(r.get(col_principio), r.get(col_nome), r.get(col_regime)), 
        axis=1
    )

st.sidebar.info(f"📊 Prodotti filtrati: {len(df_filtrato)}")

# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFICAZIONE DOPPIONI
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔍 Identificazione Doppioni")

if len(df_filtrato) == 0:
    st.warning("Nessun prodotto corrisponde ai filtri selezionati.")
    st.stop()

# Calcola gruppi di doppioni
with st.spinner("Analisi in corso..."):
    gruppi_doppioni = calcola_similarita_funzionale(df_filtrato)

st.success(f"✅ Trovati **{len(gruppi_doppioni)} gruppi** di prodotti con funzionalità simile")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZZAZIONE DOPPIONI
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📦 Gruppi di Doppioni")

if len(gruppi_doppioni) == 0:
    st.info("Nessun doppione identificato con i filtri attuali.")
else:
    # Inizializza sessione per selezione prodotti
    if 'prodotti_selezionati' not in st.session_state:
        st.session_state.prodotti_selezionati = []

    # Itera sui gruppi
    for i, (chiave, indici) in enumerate(gruppi_doppioni.items()):
        categoria, principio = chiave.split('|', 1)

        with st.expander(f"📦 Gruppo {i+1}: {categoria} ({len(indici)} prodotti)", expanded=False):
            st.markdown(f"**Principio attivo / Codice:** `{principio}`")

            # Mostra tabella prodotti
            df_gruppo = df_filtrato.loc[indici].copy()

            # Colonne da mostrare
            colonne_visibili = []
            if col_nome:
                colonne_visibili.append(col_nome)
            if col_principio:
                colonne_visibili.append(col_principio)
            if col_qta:
                colonne_visibili.append(col_qta)
            if col_aic:
                colonne_visibili.append(col_aic)

            # Aggiungi checkbox per selezione
            df_gruppo['seleziona'] = False

            # Mostra tabella con checkbox
            for idx, riga in df_gruppo.iterrows():
                cols = st.columns([0.5, 3, 2, 1, 1])

                checkbox = cols[0].checkbox(f"Seleziona", key=f"gruppo_{i}_riga_{idx}")
                cols[1].write(f"**{riga.get(col_nome, 'N/A')}**")
                cols[2].write(riga.get(col_principio, '-'))
                cols[3].write(f"Qta: {riga.get(col_qta, 0)}")
                cols[4].write(riga.get(col_aic, '-'))

                if checkbox:
                    if idx not in st.session_state.prodotti_selezionati:
                        st.session_state.prodotti_selezionati.append(idx)
                else:
                    if idx in st.session_state.prodotti_selezionati:
                        st.session_state.prodotti_selezionati.remove(idx)

    # ═══════════════════════════════════════════════════════════════════════════
    # RIEPILOGO SELEZIONE
    # ═══════════════════════════════════════════════════════════════════════════

    st.header("✅ Prodotti Selezionati")

    if len(st.session_state.prodotti_selezionati) == 0:
        st.info("Nessun prodotto selezionato. Usa le checkbox per scegliere i prodotti da vendere con priorità.")
    else:
        df_selezionati = df_filtrato.loc[st.session_state.prodotti_selezionati].copy()

        st.write(f"**{len(df_selezionati)} prodotti selezionati**")

        # Mostra tabella riepilogativa
        colonne_export = []
        if col_nome:
            colonne_export.append(col_nome)
        if col_qta:
            colonne_export.append(col_qta)
        if col_aic:
            colonne_export.append(col_aic)

        # Cerca colonna codice ministeriale
        col_codice_min = None
        for col in df.columns:
            if any(x in col.lower() for x in ['ministeriale', 'codice_min', 'farmacode']):
                col_codice_min = col
                break

        if col_codice_min:
            colonne_export.append(col_codice_min)

        st.dataframe(df_selezionati[colonne_export], use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════════
        # EXPORT
        # ═══════════════════════════════════════════════════════════════════════

        st.header("💾 Export Lista")

        # Prepara dataframe per export
        df_export = df_selezionati[colonne_export].copy()
        df_export.columns = ['Nome_Prodotto', 'Quantita', 'AIC', 'Codice_Ministeriale'][:len(colonne_export)]

        # CSV per download
        csv_export = df_export.to_csv(index=False, encoding='utf-8').encode('utf-8')

        st.download_button(
            label="📥 Scarica CSV (prodotti selezionati)",
            data=csv_export,
            file_name='lista_da_vendere.csv',
            mime='text/csv',
            key='download_csv'
        )

        # Mostra anteprima
        with st.expander("👁️ Anteprima CSV"):
            st.code(df_export.to_csv(index=False), language='csv')

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("""
**Istruzioni:**
1. Carica il file `giacenza_otc_sop.csv` generato nella FASE 1
2. Filtra per categoria e quantità minima
3. Esplora i gruppi di doppioni identificati
4. Seleziona i prodotti da vendere con priorità
5. Scarica la lista semplificata per il farmacista
""")
