---
title: Mesa de triagem CS2
emoji: 📈
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Mesa de triagem de eventos anômalos — CS2 / Steam Community Market

Aplicativo interativo do TCC **"Detecção de padrões anômalos como possíveis sinais de manipulação em itens de CS2 no
Steam Community Market"**, de Matheus Lima Ribeiro (Ciência de Dados e IA — IESB), sob orientação do
Prof. Sérgio da Costa Côrtes.

O app reproduz o pipeline de triagem do trabalho sobre uma cesta de 5 skins Field-Tested (01/01/2024 a 15/06/2026):
baseline robusto (MAD), mudanças de regime (PELT), Isolation Forest, ranking integrado e estudo de evento da
atualização Trade Up (22/10/2025). A aba **Mercado & Regra** mostra o índice de mercado (mediana diária de 1.412
skins) e a amplitude, o calendário de eventos públicos, os 34 episódios com as quatro condições da regra de
possível sinal de manipulação, o baseline ajustado ao mercado e o piso de detecção medido com anomalias sintéticas.
Com os parâmetros padrão, os números coincidem com as tabelas do TCC.

> **Triagem, não prova.** Os dados são agregados diários (skin × dia), sem livro de ofertas nem identificação de
> contas. Um evento anômalo indica prioridade de investigação contextual, não manipulação intencional.

## Rodar localmente

```bash
pip install -r requirements.txt
python server.py --open        # Terminal de triagem (versão principal) → http://127.0.0.1:8765
streamlit run app.py           # versão alternativa em Streamlit, mesmos números
```

Na nuvem (Hugging Face Spaces / Render), o `Dockerfile` sobe o Terminal de triagem.

## Dados

`output/cesta_2024_2026.csv` é um recorte do dataset público de Devyn Pruden no Kaggle (versão 4, licença Apache 2.0).
`output/mercado_indice_diario.csv` (índice e amplitude por dia) e `output/anomalias_sinteticas_recuperacao.csv`
(taxas de recuperação) são resultados pré-calculados a partir do dataset completo, que não cabe no plano grátis.
