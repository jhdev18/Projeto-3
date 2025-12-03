# Projeto 3 🚗 SPIA - Detecção e Leitura de Placas (LPR)

Este projeto é um sistema de **Reconhecimento Automático de Placas (ALPR/LPR)** em tempo real. Ele utiliza Inteligência Artificial para detectar a localização da placa no vídeo e OCR (Reconhecimento Óptico de Caracteres) para ler o texto.

O sistema foi otimizado para rodar com aceleração de **GPU (CUDA)** tanto na detecção quanto no reconhecimento de texto.

## Componentes
Almerinda Barros
João Henrique
José Maúricio
Getúlio josé

## ✨ Funcionalidades

  * **Detecção de Objetos:** Utiliza **YOLOv8** (Ultralytics) para encontrar placas de veículos na imagem.
  * **OCR (Leitura de Texto):** Utiliza **EasyOCR** para extrair os caracteres da placa.
  * **Processamento de Texto:** Filtra os resultados usando **Regex** para garantir que o formato corresponda a uma placa válida.
  * **Otimização de Performance:** Realiza a detecção pesada (YOLO) apenas a cada *N* frames, mantendo o rastreamento leve nos intervalos.
  * **Salvamento Automático:** Salva automaticamente a imagem recortada da placa na pasta `plates/` quando uma leitura válida é feita.
  * **Suporte a GPU:** Configurado para rodar no dispositivo `cuda` para máxima velocidade.

## 🛠️ Pré-requisitos

Para executar este projeto, você precisa ter o **Python 3.8+** instalado.

### 1\. Dependências do Python

Instale as bibliotecas necessárias executando:

```bash
pip install opencv-python numpy ultralytics easyocr torch torchvision torchaudio
```

> **Nota sobre GPU:** Para que o `detector.to('cuda')` e o `easyocr(gpu=True)` funcionem, você deve ter uma placa de vídeo NVIDIA e instalar a versão do **PyTorch** compatível com o seu CUDA. Verifique em: [pytorch.org](https://pytorch.org/get-started/locally/).

### 2\. Arquivos Necessários

Você precisa colocar o arquivo de pesos treinado do YOLO na raiz do projeto:

  * `license_plate_detector.pt` (O modelo treinado para detectar placas).

### 3\. Estrutura de Pastas

O script tenta salvar as imagens em uma pasta específica. Crie uma pasta chamada `plates` na raiz do projeto:

```bash
mkdir plates
```

## ⚙️ Configuração

No início do código, você encontrará a seção `----- CONFIG -----` onde pode ajustar os parâmetros:

| Variável | Descrição | Padrão |
| :--- | :--- | :--- |
| `YOLO_MODEL` | Nome/Caminho do arquivo de modelo treinado. | `"license_plate_detector.pt"` |
| `CONF_THRESHOLD` | Confiança mínima (0 a 1) para considerar uma detecção válida. | `0.35` |
| `OCR_LANGS` | Lista de idiomas para o EasyOCR. | `['en']` |
| `PLATE_REGEX` | Expressão regular para validar o formato da placa. | `r'[A-Z0-9]{4,8}'` |
| `VIDEO_SOURCE` | `0` para Webcam ou caminho de um arquivo de vídeo (ex: `"video.mp4"`). | `0` |
| `RUN_EVERY_N_FRAMES` | Executa o YOLO a cada X frames para economizar recursos. | `10` |

## 🚀 Como Executar

Com as dependências instaladas e o modelo na pasta, execute:

```bash
python seu_script.py
```

  * Pressione **ESC** para encerrar a execução.

## 🧠 Como Funciona o Código

1.  **Captura:** O vídeo é lido frame a frame.
2.  **Detecção (YOLO):** A cada 10 frames (configurável), o YOLO escaneia a imagem completa em busca de placas. As coordenadas das caixas detectadas são salvas em cache.
3.  **Recorte e OCR:** Para cada caixa detectada (usando o cache nos frames intermediários), o código recorta a região da placa e passa para o EasyOCR.
4.  **Pós-processamento:** O texto retornado é limpo (remove caracteres especiais) e validado contra o Regex.
5.  **Display e Save:** Se uma placa válida for lida, ela é desenhada na tela e a imagem recortada é salva na pasta `plates/` (com um delay de 1 segundo entre salvamentos para evitar duplicatas excessivas).

## ⚠️ Solução de Problemas Comuns

  * **Erro `AttributeError: 'NoneType' object has no attribute 'group'`:**
      * Isso ocorre se o OCR não detectar nada. O código já possui tratamento para isso na função `postprocess_text`.
  * **Lentidão:**
      * Verifique se o PyTorch está realmente usando a GPU: `import torch; print(torch.cuda.is_available())`. Se retornar `False`, reinstale o PyTorch com suporte a CUDA.
  * **Erro ao salvar imagem:**
      * Certifique-se de que a pasta `plates/` existe no diretório.
