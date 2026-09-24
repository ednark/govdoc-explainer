// Semantic + keyword search with Pagefind RRF fusion.
//
// Vectors load lazily on first search interaction (not on page load) and the
// client consumes the int8-quantized embedding_q8.json written at build time
// (~4x smaller than fp32 JSON; ranking on raw int8 is benchmark-verified to
// match fp32 cosine). Falls back to the fp32 embedding.json if the q8 file is
// missing. Vectors are L2-normalized at build time, so the fp32 dot product
// equals cosine similarity.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

const Q8_FORMAT = 'govdoc-int8-v1';

let extractor = null;
let pagefind = null;
let embeddingsData = [];
let vectorsPromise = null;

async function initExtractor() {
    if (extractor) return extractor;
    // dtype pinned: q8 is the smallest usable variant for this model
    // (118MB int8 + ~32MB tokenizer; q4 is larger for this architecture).
    extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', { dtype: 'q8' });
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

function b64ToInt8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const i8 = new Int8Array(u8.length);
    for (let i = 0; i < u8.length; i++) i8[i] = (u8[i] << 24) >> 24; // sign-extend
    return i8;
}

function cosineSimilarityF32(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot; // vectors are L2-normalized at build time
}

// Cosine on raw int8 vectors: per-vector scale cancels out, so no
// dequantization pass is needed for ranking.
function cosineSimilarityInt8(queryI8, docI8) {
    let dot = 0, qn = 0, dn = 0;
    for (let i = 0; i < queryI8.length; i++) {
        const q = queryI8[i], d = docI8[i];
        dot += q * d;
        qn += q * q;
        dn += d * d;
    }
    const denom = Math.sqrt(qn) * Math.sqrt(dn);
    return denom ? dot / denom : 0;
}

function quantizeQuery(f32) {
    let maxAbs = 0;
    for (let i = 0; i < f32.length; i++) {
        const m = Math.abs(f32[i]);
        if (m > maxAbs) maxAbs = m;
    }
    const i8 = new Int8Array(f32.length);
    if (!maxAbs) return i8;
    const scale = maxAbs / 127;
    for (let i = 0; i < f32.length; i++) {
        i8[i] = Math.max(-127, Math.min(127, Math.round(f32[i] / scale)));
    }
    return i8;
}

// Normalize fp32-legacy or int8 entries to { id, title, text, vec }.
function normalizeEntry(entry) {
    return {
        id: entry.id,
        title: entry.title ?? entry.id,
        text: entry.body || entry.text || '',
        vec: entry.scale !== undefined
            ? { kind: 'i8', data: b64ToInt8(entry.embedding) }
            : {
                kind: 'f32',
                data: (entry.embedding.length === 1 ? entry.embedding[0] : entry.embedding),
            },
    };
}

async function loadVectors(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${response.status} ${path}`);
    const data = await response.json();
    const entries = data && data.format === Q8_FORMAT ? data.entries : data;
    return entries.map(normalizeEntry);
}

// Lazy vector loading: deferred until the first search interaction. Tries the
// int8 file first, falling back to the fp32 legacy file for older builds.
function ensureVectors(path) {
    if (!vectorsPromise) {
        vectorsPromise = loadVectors(path)
            .catch(() => loadVectors(path.replace('embedding_q8.json', 'embedding.json')))
            .then(entries => { embeddingsData = entries || []; })
            .catch(e => {
                console.log('No embeddings found at', path, e.message);
                embeddingsData = [];
            });
    }
    return vectorsPromise;
}

async function generateQueryEmbedding(text) {
    const model = await initExtractor();
    const output = await model(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

function similarityTo(entry, queryEmbedding, queryI8) {
    if (entry.vec.kind === 'i8') return cosineSimilarityInt8(queryI8, entry.vec.data);
    return cosineSimilarityF32(queryEmbedding, entry.vec.data);
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

function normalizeUrl(url) {
    return String(url || '').replace(/^\.?\//, '').split('#')[0];
}

async function semanticDocSearch(query) {
    const queryEmbedding = await generateQueryEmbedding(query);
    const queryI8 = embeddingsData.length && embeddingsData[0].vec.kind === 'i8'
        ? quantizeQuery(queryEmbedding) : null;
    const results = embeddingsData.map(item => {
        const similarity = queryI8
            ? cosineSimilarityInt8(queryI8, item.vec.data)
            : cosineSimilarityF32(queryEmbedding, item.vec.data);
        return { id: item.id, title: item.title, text: item.text, similarity };
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

    const embeddingPath = isSourcePage ? './embedding_q8.json' : './assets/embedding_q8.json';

    // Start the vector download as soon as the user shows search intent.
    embedInput.addEventListener('focus', () => ensureVectors(embeddingPath));

    embedButton.addEventListener('click', async () => {
        const query = embedInput.value.trim();
        if (!query) return;

        embedMessage.textContent = 'Searching...';
        embedMessage.style.display = 'inline';

        try {
            await ensureVectors(embeddingPath);

            if (isMainPage) {
                const semanticPromise = embeddingsData.length
                    ? semanticDocSearch(query).catch(e => {
                        console.error(e);
                        return [];
                    })
                    : Promise.resolve([]);
                if (embeddingsData.length && !extractor) {
                    embedMessage.textContent = 'Loading search model (one-time download)...';
                }

                const [semanticResults, keywordResults] = await Promise.all([
                    semanticPromise,
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
                const queryI8 = embeddingsData.length && embeddingsData[0].vec.kind === 'i8'
                    ? quantizeQuery(queryEmbedding) : null;
                const results = embeddingsData.map(item => {
                    const similarity = queryI8
                        ? cosineSimilarityInt8(queryI8, item.vec.data)
                        : cosineSimilarityF32(queryEmbedding, item.vec.data);
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