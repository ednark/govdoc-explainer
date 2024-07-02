# govdoc-explainer

Setup configs by editing files within the ./config directory

Optionally, if using a local llm, turn on and login to Ollama, download the models of interest.
```
docker compose up -d
open http://localhost:3000/
```

Execute the main script to build out the html
```
python extract_content.py
```

Read and search
```
open http://127.0.0.1:5500/govdoc-explainer
```
