import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

let extractor = null;
let pagefind = null;
let embeddingsData = [];

async function initExtractor() {
    if (extractor) return extractor;
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    return extractor;
}

async function initPagefind() {
    if (pagefind !== null) return pagefind;
    try {
        pagefind = await import('/assets/pagefind/pagefind.js');
        return pagefind;
    } catch (e) {
        console.log('Pagefind index not available:', e.message);
        pagefind = false;
        return pagefind;
    }
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

function normalizeUrl(url) {
    return String(url || '').replace(/^\.?\//, '').split('#')[0];
}

function rrfMerge(semanticResults, keywordResults, k = 60, limit = 8) {
    const scores = new Map();
    const addItem = (item, rank) => {
        const key = normalizeUrl(item.url);
        if (!scores.has(key)) {
            scores.set(key, { score: 0, channels: new Set(), item: item });
        }
        const entry = scores.get(key);
        entry.score += 1 / (k + rank + 1);
        entry.channels.add(item.channel);
        if (item.excerpt && !entry.item.excerpt) entry.item.excerpt = item.excerpt;
    };
    semanticResults.forEach((item, rank) => addItem(item, rank));
    keywordResults.forEach((item, rank) => addItem(item, rank));
    return Array.from(scores.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map(entry => ({
            url: entry.item.url,
            title: entry.item.title,
            excerpt: entry.item.excerpt || '',
            channels: Array.from(entry.channels),
            score: entry.score,
        }));
}

async function semanticDocSearch(query) {
    const queryEmbedding = await generateQueryEmbedding(query);
    const results = embeddingsData.map(item => {
        const embedding = item.embedding.length === 1 ? item.embedding[0] : item.embedding;
        const similarity = cosineSimilarity(queryEmbedding, embedding);
        return { id: item.id, title: item.title, text: item.body || item.text, similarity };
    });
    results.sort((a, b) => b.similarity - a.similarity);
    return results
        .filter(x => x.similarity >= 0.3)
        .slice(0, 25)
        .map(x => ({
            url: './sources/' + x.id + '/index.html',
            title: x.title,
            similarity: x.similarity,
            channel: 'semantic',
        }));
}

async function keywordSearch(query) {
    const pf = await initPagefind();
    if (!pf) return [];
    const search = await pf.search(query);
    const results = await Promise.all(
        search.results.slice(0, 25).map(r => r.data())
    );
    return results.map(d => ({
        url: d.url,
        title: d.meta && d.meta.title ? d.meta.title : d.url,
        excerpt: d.excerpt || '',
        channel: 'keyword',
    }));
}

function renderResults(results) {
    const container = document.getElementById('embed-results');
    if (!container) return '';
    container.hidden = false;
    if (results.length === 0) {
        container.innerHTML = '<p class="embed-results-empty">No results found.</p>';
        return;
    }
    const channelLabel = { semantic: 'semantic', keyword: 'keyword' };
    container.innerHTML = '<ol class="embed-results-list">' + results.map(r => {
        const badges = r.channels.map(c => `<span class="result-channel channel-${c}">${channelLabel[c]}</span>`).join('');
        const excerpt = r.excerpt ? `<p class="result-excerpt">${r.excerpt}</p>` : '';
        return `<li class="result-item"><a href="${r.url}">${r.title}</a> ${badges}${excerpt}</li>`;
    }).join('') + '</ol>';
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
        embeddingsData = [];
    }

    embedButton.addEventListener('click', async () => {
        const query = embedInput.value.trim();
        if (!query) return;

        embedMessage.textContent = 'Searching...';
        embedMessage.style.display = 'inline';

        try {
            if (isMainPage) {
                let extractorPromise = null;
                if (embeddingsData.length) {
                    if (!extractor) embedMessage.textContent = 'Loading search model (one-time download)...';
                    extractorPromise = semanticDocSearch(query).catch(e => {
                        console.error(e);
                        return [];
                    });
                } else {
                    extractorPromise = Promise.resolve([]);
                }

                const [semanticResults, keywordResults] = await Promise.all([
                    extractorPromise,
                    keywordSearch(query).catch(e => {
                        console.error(e);
                        return [];
                    }),
                ]);

                embedMessage.textContent = '';
                embedMessage.style.display = 'none';

                const merged = rrfMerge(semanticResults, keywordResults);
                renderResults(merged);
            } else {
                const queryEmbedding = await generateQueryEmbedding(query);
                const results = embeddingsData.map(item => {
                    const embedding = item.embedding.length === 1 ? item.embedding[0] : item.embedding;
                    const similarity = cosineSimilarity(queryEmbedding, embedding);
                    return { id: item.id, similarity };
                });
                results.sort((a, b) => b.similarity - a.similarity);
                const matches = results.filter(x => x.similarity >= 0.3);

                const resultDivs = document.querySelectorAll('.text-chunk');
                resultDivs.forEach(div => div.classList.remove('found'));

                if (matches.length === 0) {
                    embedMessage.innerHTML = 'No results found.';
                } else {
                    embedMessage.innerHTML = `${matches.length} chunk(s) found.`;
                    matches.forEach(r => {
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
