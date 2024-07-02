# govdoc-explainer

Setup configs by editing files within the ./config directory
- List of sources in csv format lives in `./config/sources.csv`
- Choice of which llm to use lives in `./config/llm.txt`
- Details of each prompt in use lives in `./config/prompts/*.txt`
- List of perspective to consider lives in `./config/perspectives.csv`


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
