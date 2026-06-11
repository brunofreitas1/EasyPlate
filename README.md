# EasyPlate 🚗

Sistema de reconhecimento de placas veiculares para controle de acesso em condomínios residenciais.

## Funcionalidades

- **Detecção automática** de placas Mercosul (`ABC1D23`) e padrão antigo (`ABC1234`)
- **Pré-processamento inteligente** com 5+ pipelines de OCR e votação majoritária
- **Fallback robusto** para imagens de baixa qualidade (deblur, correção de contraste, busca combinada)
- **Correção automática** de confusões comuns de OCR (0↔6, C↔G, S↔5, etc.)
- **Interface Streamlit** com 3 abas: Detectar, Cadastrar, Histórico
- **Banco SQLite** local para cadastro de moradores e log de acessos

## Requisitos

- Python 3.9+
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Streamlit, OpenCV, Pillow

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

## Estrutura do projeto

| Arquivo | Função |
|---------|--------|
| `app.py` | Interface web (Streamlit) |
| `detector.py` | Pipeline de detecção com EasyOCR |
| `database.py` | CRUD SQLite (moradores + histórico) |
| `utils.py` | Validação, correção, similaridade entre placas |
| `easy_plate.ipynb` | Notebook para apresentação acadêmica |
| `requirements.txt` | Dependências do projeto |

## Como funciona a detecção

1. **Contornos** — busca por região retangular com proporção de placa na imagem original
2. **ROI** — aplica 5 pipelines de pré-processamento no recorte encontrado (Otsu, adaptativo, bilateral, CLAHE, gamma)
3. **Fallback full-image** — se o ROI falhar, processa a imagem completa com 3 pipelines adicionais
4. **Busca combinada** — une partes de texto separadas pelo OCR (ex: `"MG"` + `"3164"` → `"CMG3164"`)
5. **Votação** — seleciona o resultado mais frequente entre todos os pipelines

## Formatos suportados

| Formato | Padrão | Exemplo |
|---------|--------|---------|
| Mercosul | `LLLNLDD` | `RIO2A18` |
| Antiga | `LLLDDDD` | `FZJ5102` |

## Licença

Projeto acadêmico.
