import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

let extractor = null;
let embeddingsData = [];

async function initExtractor() {
    if (extractor) return extractor;
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    return extractor;
}

async function loadEmbeddings(path) {
    const response = await fetch(path);
    return await response.json();
}

async function generateQueryEmbedding(text) {
    const extractor = await initExtractor();
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

function cosineSimilarity(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

async function semanticSearch(query) {
    const queryEmbedding = await generateQueryEmbedding(query);
    const results = embeddingsData.map(item => {
        const embedding = item.embedding.length === 1 ? item.embedding[0] : item.embedding;
        const similarity = cosineSimilarity(queryEmbedding, embedding);
        return { id: item.id, title: item.title, text: item.body || item.text, similarity };
    });
    results.sort((a, b) => b.similarity - a.similarity);
    return results.filter(x => x.similarity >= 0.3);
}

document.addEventListener('DOMContentLoaded', async function () {
    const embedQuery = document.getElementById('embed-query');
    if (!embedQuery) return;

    const embedInput = document.getElementById('embed-query-input');
    const embedButton = document.getElementById('embed-query-button');
    const embedMessage = document.getElementById('embed-query-message');

    if (!embedButton || !embedInput) return;

    const isSourcePage = window.location.pathname.includes('/sources/');
    const isMainPage = !isSourcePage;

    const embeddingPath = isSourcePage ? './embedding.json' : './assets/embedding.json';

    try {
        embeddingsData = await loadEmbeddings(embeddingPath);
    } catch (e) {
        console.log('No embeddings found at', embeddingPath);
        return;
    }

    embedButton.addEventListener('click', async () => {
        const query = embedInput.value.trim();
        if (!query) return;

        embedMessage.textContent = 'Searching...';
        embedMessage.style.display = 'inline';

        try {
            const results = await semanticSearch(query);

            if (isMainPage) {
                if (results.length === 0) {
                    embedMessage.innerHTML = 'No results found.';
                } else {
                    embedMessage.innerHTML = results.slice(0, 10).map(r =>
                        `<a href="./sources/${r.id}/index.html">${r.title}</a>`
                    ).join(' | ');
                }
            } else {
                const resultDivs = document.querySelectorAll('.text-chunk');
                resultDivs.forEach(div => div.classList.remove('found'));

                if (results.length === 0) {
                    embedMessage.innerHTML = 'No results found.';
                } else {
                    embedMessage.innerHTML = `${results.length} chunk(s) found.`;
                    results.forEach(r => {
                        const chunkDiv = document.getElementById(`chunk-${r.id}`);
                        if (chunkDiv) {
                            chunkDiv.classList.add('found');
                        }
                    });
                }
            }
        } catch (e) {
            embedMessage.innerHTML = 'Search error.';
            console.error(e);
        }
    });

    embedInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') embedButton.click();
    });
});
