flowchart TB
  S[start.bat / go-chat.bat\n(venv, detect HW, choose backend)]
  CHOICE{Choose runtime}
  CLI[CLI\n(chat_cli.py)]
  API[API\n(api.py / FastAPI)]
  GRADIO[Optional Gradio\n(app.py demo)]
  BACKEND[Backend Abstraction\n(model_loader.py)]
  TF[infer_transformers.py\n(Transformers + bnb + PEFT)]
  LL[infer_llamacpp.py\n(llama.cpp / ggml / gguf)]
  MODELS[Model files\n(HF / GPTQ / gguf / LoRA)]
  RAG[RAGManager\n(rag_manager.py / Chroma or in-mem)]
  LORAUTIL[lora_utils.py\n(load/merge adapters)]
  SCRIPTS[scripts/\n(train_lora, create_dataset, convert)]
  POST[Postprocess & Emotion\n(-> Live2D / UI)]

  S --> CHOICE
  CHOICE --> CLI
  CHOICE --> API
  CHOICE --> GRADIO

  CLI --> BACKEND
  API --> BACKEND
  GRADIO --> BACKEND

  BACKEND --> TF
  BACKEND --> LL
  TF --> MODELS
  LL --> MODELS

  BACKEND --> RAG
  BACKEND --> LORAUTIL
  RAG --> BACKEND
  BACKEND --> POST
  CLI --> POST
  API --> POST

  SCRIPTS -.-> LORAUTIL
  SCRIPTS -.-> MODELS

  classDef core fill:#bbf,stroke:#333,stroke-width:1px;
  class BACKEND,TF,LL,RAG,LORAUTIL core;
